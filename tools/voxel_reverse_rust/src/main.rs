//! Reverse-projection voxel reconstruction in Rust.
//!
//! Same algorithm as `voxel_reconstruct.py --reverse` and the JAX-CPU
//! port: iterate voxels, project each onto every frame, classify as
//! colour / air / discard, then apply the ratio threshold +
//! min-color-count gate. Parallelised over x slabs with rayon — each
//! ix slab gets its own slice of the global counters, so there are no
//! atomics or merging passes.
//!
//! Frame parsing matches the Python `tools/serve.py:parse_frame` exactly
//! (224-byte little-endian header: 48 f32 matrices + 8 trailing fields,
//! followed by the depth then colour payloads).

use std::env;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use glam::{DMat4, DVec3, DVec4};
use half::f16;
use rayon::prelude::*;
use serde_json::json;

const FRAME_HEADER_SIZE: usize = 224;
const DEPTH_FMT_UINT16_LA: u32 = 0;
const DEPTH_FMT_FLOAT32:   u32 = 1;
const COLOR_FMT_RGBA8:     u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq)]
enum DepthSource {
    /// WebXR phone depth from the .bin file at depth-buffer resolution,
    /// projected via Bd. (Default; matches voxel_reconstruct.py.)
    Phone,
    /// On-the-fly Gaussian detail-injection blend (phone low-frequency +
    /// model_raw high-frequency, in metres) at colour-image resolution.
    /// (u, v) maps directly to colour-grid pixels — no Bd indirection.
    Blend,
}

/// Which family of monocular-depth cache the blend pulls from. The on-disk
/// directory layout matches `tools/cache_model_raw.py`:
///   V2 → `<session>/model_raw/`
///   V3 → `<session>/model_raw_v3/`
/// The cache contract (`index.json` + per-frame `frame_NNNNNN.f16` at
/// colour-image resolution) is identical across versions, so the only
/// thing that changes here is which directory we read from and what
/// suffix we apply to the output JSON files.
#[derive(Clone, Copy, Debug, PartialEq)]
enum ModelVersion {
    V2,
    V3,
}

impl ModelVersion {
    fn cache_subdir(self) -> &'static str {
        match self {
            ModelVersion::V2 => "model_raw",
            ModelVersion::V3 => "model_raw_v3",
        }
    }
}

struct Frame {
    v_inv: DMat4,
    p: DMat4,
    bd: DMat4,
    cam_origin: DVec3,
    /// Depth source on a (h, w) row-major grid. For Phone the grid is the
    /// depth buffer; for Blend it is the colour-image grid.
    depth: Vec<f32>,
    color: Vec<u8>,
    w: usize,
    h: usize,
    cw: usize,
    ch: usize,
    /// True when `(u, v)` should be mapped through Bd to land in the
    /// depth source grid (Phone). False when `(u, v)` already names a
    /// colour-grid pixel (Blend).
    uses_bd: bool,
}

#[inline]
fn read_f32(buf: &[u8], off: usize) -> f32 {
    f32::from_le_bytes(buf[off..off + 4].try_into().unwrap())
}

#[inline]
fn read_u32(buf: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(buf[off..off + 4].try_into().unwrap())
}

fn parse_frame(body: &[u8]) -> Result<Option<Frame>, String> {
    if body.len() < FRAME_HEADER_SIZE {
        return Err(format!("frame too short: {} < {}", body.len(), FRAME_HEADER_SIZE));
    }

    // 48 f32 floats: viewMatrix, projectionMatrix, normDepthBufferFromNormView
    // (each 16 column-major).
    let mut view  = [0.0f32; 16];
    let mut proj  = [0.0f32; 16];
    let mut nd_m  = [0.0f32; 16];
    for i in 0..16 { view[i] = read_f32(body, i * 4); }
    for i in 0..16 { proj[i] = read_f32(body, 64 + i * 4); }
    for i in 0..16 { nd_m[i] = read_f32(body, 128 + i * 4); }

    let dw      = read_u32(body, 192) as usize;
    let dh      = read_u32(body, 196) as usize;
    let raw_to_m = read_f32(body, 200);
    let dfmt    = read_u32(body, 204);
    let cw      = read_u32(body, 208) as usize;
    let ch      = read_u32(body, 212) as usize;
    let cfmt    = read_u32(body, 216);
    let clen    = read_u32(body, 220) as usize;

    let depth_bpp = match dfmt {
        DEPTH_FMT_UINT16_LA => 2,
        DEPTH_FMT_FLOAT32   => 4,
        _ => return Err(format!("unknown depth format {dfmt}")),
    };
    let depth_expected = dw * dh * depth_bpp;
    let body_after = &body[FRAME_HEADER_SIZE..];
    if body_after.len() < depth_expected {
        return Err(format!(
            "depth payload truncated: got {}, expected {}",
            body_after.len(), depth_expected,
        ));
    }
    let depth_payload = &body_after[..depth_expected];

    let mut depth = vec![0.0f32; dw * dh];
    if dfmt == DEPTH_FMT_UINT16_LA {
        for i in 0..(dw * dh) {
            let raw = u16::from_le_bytes(
                depth_payload[i * 2..i * 2 + 2].try_into().unwrap()
            ) as f32;
            depth[i] = raw * raw_to_m;
        }
    } else {
        for i in 0..(dw * dh) {
            let raw = f32::from_le_bytes(
                depth_payload[i * 4..i * 4 + 4].try_into().unwrap()
            );
            depth[i] = raw * raw_to_m;
        }
    }

    if cfmt != COLOR_FMT_RGBA8 || clen == 0 {
        // Frames without colour are unusable for reverse-projection.
        return Ok(None);
    }
    let expected_color = cw * ch * 4;
    if expected_color != clen {
        return Err(format!(
            "RGBA8 declared {clen} bytes, expected {expected_color}"
        ));
    }
    let rest = &body_after[depth_expected..];
    if rest.len() < clen {
        return Err(format!("color payload truncated: got {}, expected {}", rest.len(), clen));
    }
    let color = rest[..clen].to_vec();

    // glam's DMat4::from_cols_array takes 16 f64s in column-major order; each
    // 4 consecutive values are one column. The Python code reads the same
    // layout via `_mat4_from_column_major`.
    let v  = DMat4::from_cols_array(&view.map(|x| x as f64));
    let p  = DMat4::from_cols_array(&proj.map(|x| x as f64));
    let bd = DMat4::from_cols_array(&nd_m.map(|x| x as f64));
    let v_inv = v.inverse();
    let cam_origin = DVec3::new(v.col(3).x, v.col(3).y, v.col(3).z);

    Ok(Some(Frame {
        v_inv, p, bd, cam_origin,
        depth, color,
        w: dw, h: dh, cw, ch,
        uses_bd: true,
    }))
}

/// Build a Frame whose depth source is the on-the-fly blend at colour
/// resolution. Reads the model_raw cache for `idx` from
/// `<session>/model_raw/frame_NNNNNN.f16` (cw_c × ch_c float16) and uses
/// the same hole-aware Gaussian detail-injection as the Python server
/// (see `_compute_blend_metres` in tools/serve.py): low-frequency content
/// from phone (in metres, sparse), high-frequency content from
/// `a · model_raw + b` where (a, b) is OLS-fit on the overlap.
///
/// Returns `Ok(None)` if the model_raw cache file is missing for this
/// frame (caller drops the frame, like a missing colour buffer).
fn parse_frame_with_blend(
    body: &[u8],
    model_raw_path: &Path,
    cw_c: usize,
    ch_c: usize,
    sigma_frac: f64,
) -> Result<Option<Frame>, String> {
    let phone = match parse_frame(body)? {
        Some(f) => f,
        None    => return Ok(None),
    };
    if cw_c != phone.cw || ch_c != phone.ch {
        // index.json's stored dims should always match the bin's; if they
        // don't this frame's cache was built against a different capture.
        return Err(format!(
            "model_raw dims ({cw_c}×{ch_c}) don't match colour ({}×{})",
            phone.cw, phone.ch,
        ));
    }
    let bytes = match fs::read(model_raw_path) {
        Ok(b) => b,
        Err(e) => return Err(format!("model_raw read failed: {e}")),
    };
    if bytes.len() != cw_c * ch_c * 2 {
        return Err(format!(
            "model_raw size {} != {} expected",
            bytes.len(), cw_c * ch_c * 2,
        ));
    }
    let model_raw: Vec<f32> = bytes
        .chunks_exact(2)
        .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
        .collect();
    let blend = compute_blend_metres(&phone, &model_raw, cw_c, ch_c, sigma_frac);

    // The blend grid is in GL row order (row 0 = scene-bottom = norm-view
    // v=0), matching how `project_voxel_to_frame` already samples the
    // colour buffer. Drop the original phone depth buffer to free RAM.
    Ok(Some(Frame {
        v_inv: phone.v_inv,
        p: phone.p,
        bd: phone.bd,
        cam_origin: phone.cam_origin,
        depth: blend,
        color: phone.color,
        w: cw_c, h: ch_c,
        cw: phone.cw, ch: phone.ch,
        uses_bd: false,
    }))
}

// ----------------------------------------------------------------------
// Blend depth: hole-aware Gaussian detail injection in pure Rust.
// ----------------------------------------------------------------------

/// Compute the blend on the colour-image grid, returned in GL row order
/// (row 0 = scene-bottom). Algorithm matches `_compute_blend_metres` in
/// tools/serve.py.
fn compute_blend_metres(
    frame: &Frame,
    model_raw: &[f32],
    cw_c: usize, ch_c: usize,
    sigma_frac: f64,
) -> Vec<f32> {
    let n = cw_c * ch_c;

    // Build phone and model on the same colour grid in NATURAL display
    // orientation (yo=0 = top of view = norm-view v=1). NaN where the
    // sample would land outside the source grid or on a zero/negative
    // depth pixel (phone holes).
    let mut phone_grid = vec![f32::NAN; n];
    let mut model_grid = vec![f32::NAN; n];
    for yo in 0..ch_c {
        let v_v = 1.0 - (yo as f64 + 0.5) / ch_c as f64;
        for xo in 0..cw_c {
            let u_v = (xo as f64 + 0.5) / cw_c as f64;

            // Phone via Bd (norm-view → norm-depth-buffer) → pixel.
            // Matches `nd = nv @ Bd.T` in tools/serve.py
            // _sample_phone_model_on_color_grid (and the same convention
            // in project_voxel_to_frame above). The pre-fix code applied
            // Bd⁻¹ here, which inverted the per-pixel phone-depth
            // assignment and produced an OLS fit whose `a` was off by a
            // large factor — invisible at 5 cm voxels but obviously wrong
            // at 2 cm.
            let nv = DVec4::new(u_v, v_v, 0.0, 1.0);
            let nd = frame.bd * nv;
            if nd.w.abs() > 1e-12 {
                let u_d = nd.x / nd.w;
                let v_d = nd.y / nd.w;
                let bx = (1.0 - u_d) * frame.w as f64 - 0.5;
                let by = v_d * frame.h as f64 - 0.5;
                if let Some(p) = bilinear_sample(&frame.depth, frame.w, frame.h, bx, by) {
                    if p.is_finite() && p > 0.0 {
                        phone_grid[yo * cw_c + xo] = p;
                    }
                }
            }

            // Model: direct colour-grid sample (model_raw stores row 0 =
            // scene bottom too — cache_model_raw.py does NOT vertical-
            // flip, so the row index for v_v is v_v · ch_c).
            let sx = u_v * cw_c as f64 - 0.5;
            let sy = v_v * ch_c as f64 - 0.5;
            if let Some(m) = bilinear_sample(model_raw, cw_c, ch_c, sx, sy) {
                if m.is_finite() && m > 1e-3 {
                    model_grid[yo * cw_c + xo] = m;
                }
            }
        }
    }

    // OLS fit a, b: phone ≈ a · model_raw + b on the overlap.
    let (a, b) = ols_affine(&phone_grid, &model_grid);

    let mut model_metres = vec![f32::NAN; n];
    for i in 0..n {
        if model_grid[i].is_finite() {
            model_metres[i] = (a * model_grid[i] as f64 + b) as f32;
        }
    }

    // Hole-aware Gaussian via 3 iterations of a box filter — variance of
    // 3·box(r) ≈ r², so sigma ≈ box-radius. Each box pass is a running-
    // sum sweep, O(N) regardless of radius.
    let diag = ((cw_c * cw_c + ch_c * ch_c) as f64).sqrt();
    let radius = (sigma_frac * diag).max(1.0).round() as usize;
    let low_phone = blur_with_holes(&phone_grid, cw_c, ch_c, radius, 3);
    let low_model = blur_with_holes(&model_metres, cw_c, ch_c, radius, 3);

    // blend = low_phone + (model_metres − low_model). Pixels with no
    // valid phone neighbour fall back to plain model_metres so the
    // result has no NaN holes beyond what model_raw can't cover.
    let mut blend_natural = vec![0.0f32; n];
    for i in 0..n {
        let lp = low_phone[i];
        let lm = low_model[i];
        let mm = model_metres[i];
        let v = if lp.is_finite() && lm.is_finite() && mm.is_finite() {
            lp + (mm - lm)
        } else if mm.is_finite() {
            mm
        } else {
            0.0
        };
        blend_natural[i] = v;
    }

    // Flip to GL row order so row 0 = scene-bottom = norm-view v=0.
    let mut blend_gl = vec![0.0f32; n];
    for yo in 0..ch_c {
        let src_row = (ch_c - 1 - yo) * cw_c;
        let dst_row = yo * cw_c;
        blend_gl[dst_row..dst_row + cw_c]
            .copy_from_slice(&blend_natural[src_row..src_row + cw_c]);
    }
    blend_gl
}

/// Sample a (h, w) row-major image with NaN-on-out-of-bounds. Returns
/// `None` only when (x, y) is well outside the grid; pixels just past
/// the right/bottom edge clamp to the last valid index.
fn bilinear_sample(img: &[f32], w: usize, h: usize, x: f64, y: f64) -> Option<f32> {
    let w_max = (w - 1) as f64;
    let h_max = (h - 1) as f64;
    if x < -0.5 || x > w_max + 0.5 || y < -0.5 || y > h_max + 0.5 { return None; }
    let xc = x.clamp(0.0, w_max - 1e-3);
    let yc = y.clamp(0.0, h_max - 1e-3);
    let x0 = xc.floor() as usize;
    let y0 = yc.floor() as usize;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let fx = (xc - x0 as f64) as f32;
    let fy = (yc - y0 as f64) as f32;
    let a = img[y0 * w + x0];
    let b = img[y0 * w + x1];
    let c = img[y1 * w + x0];
    let d = img[y1 * w + x1];
    Some(((a * (1.0 - fx) + b * fx) * (1.0 - fy)
        + (c * (1.0 - fx) + d * fx) * fy) as f32)
}

fn ols_affine(phone: &[f32], model: &[f32]) -> (f64, f64) {
    let mut n = 0u64;
    let mut sum_m = 0.0f64;
    let mut sum_p = 0.0f64;
    let mut sum_mm = 0.0f64;
    let mut sum_mp = 0.0f64;
    for i in 0..phone.len() {
        let p = phone[i] as f64;
        let m = model[i] as f64;
        if p.is_finite() && m.is_finite() {
            n += 1;
            sum_m  += m;
            sum_p  += p;
            sum_mm += m * m;
            sum_mp += m * p;
        }
    }
    if n < 100 { return (1.0, 0.0); }
    let nf = n as f64;
    let mean_m = sum_m / nf;
    let mean_p = sum_p / nf;
    let var_m = sum_mm / nf - mean_m * mean_m;
    let cov   = sum_mp / nf - mean_m * mean_p;
    if var_m.abs() < 1e-9 { return (1.0, 0.0); }
    let a = cov / var_m;
    let b = mean_p - a * mean_m;
    (a, b)
}

/// Hole-aware box-iterated blur. NaN inputs are zeroed and the matching
/// mask is blurred too; the output is `blur(data) / blur(mask)` where the
/// mask blur is above a small floor (otherwise the result is NaN — the
/// caller's "no valid phone neighbour" fallback). Three iterations of a
/// box filter of radius r approximate a Gaussian with σ ≈ r.
fn blur_with_holes(data: &[f32], w: usize, h: usize, radius: usize, iters: usize) -> Vec<f32> {
    let n = w * h;
    let mut data_z = vec![0.0f32; n];
    let mut mask   = vec![0.0f32; n];
    for i in 0..n {
        if data[i].is_finite() {
            data_z[i] = data[i];
            mask[i]   = 1.0;
        }
    }
    for _ in 0..iters {
        box_blur_separable_inplace(&mut data_z, w, h, radius);
        box_blur_separable_inplace(&mut mask,   w, h, radius);
    }
    let mut out = vec![f32::NAN; n];
    for i in 0..n {
        if mask[i] > 0.05 {
            out[i] = data_z[i] / mask[i].max(1e-6);
        }
    }
    out
}

/// One iteration of a separable box filter (mean of a (2r+1)×(2r+1)
/// neighbourhood) implemented with running sums — O(N) regardless of r.
/// Edge windows shrink to whatever is in-bounds, so the mask blur keeps
/// the right normalisation against partial windows at the borders.
fn box_blur_separable_inplace(data: &mut [f32], w: usize, h: usize, radius: usize) {
    let mut tmp = vec![0.0f32; w * h];
    // Horizontal: each row → tmp.
    for y in 0..h {
        let row_in  = &data[y * w..(y + 1) * w];
        let row_out = &mut tmp[y * w..(y + 1) * w];
        let mut sum: f64 = 0.0;
        // Pre-roll: window initially starts at x=0 with right edge at
        // index radius (clamped to w-1).
        let init_hi = radius.min(w - 1);
        for i in 0..=init_hi { sum += row_in[i] as f64; }
        let mut lo: i64 = 0;
        let mut hi: i64 = init_hi as i64;
        for x in 0..w {
            let want_hi = (x + radius) as i64;
            let want_lo = x as i64 - radius as i64;
            // Slide right edge.
            while hi < want_hi && hi + 1 < w as i64 {
                hi += 1;
                sum += row_in[hi as usize] as f64;
            }
            // Slide left edge.
            while lo < want_lo {
                sum -= row_in[lo as usize] as f64;
                lo += 1;
            }
            let count = (hi - lo + 1) as f64;
            row_out[x] = (sum / count) as f32;
        }
    }
    // Vertical: tmp → data.
    for x in 0..w {
        let mut sum: f64 = 0.0;
        let init_hi = radius.min(h - 1);
        for i in 0..=init_hi { sum += tmp[i * w + x] as f64; }
        let mut lo: i64 = 0;
        let mut hi: i64 = init_hi as i64;
        for y in 0..h {
            let want_hi = (y + radius) as i64;
            let want_lo = y as i64 - radius as i64;
            while hi < want_hi && hi + 1 < h as i64 {
                hi += 1;
                sum += tmp[hi as usize * w + x] as f64;
            }
            while lo < want_lo {
                sum -= tmp[lo as usize * w + x] as f64;
                lo += 1;
            }
            let count = (hi - lo + 1) as f64;
            data[y * w + x] = (sum / count) as f32;
        }
    }
}

enum Hit {
    Color([f64; 3]),
    Air,
}

#[inline]
fn project_voxel_to_frame(
    pt: DVec3,
    frame: &Frame,
    near: f64, far: f64, tol: f64,
) -> Option<Hit> {
    // World → view → clip.
    let pt4 = DVec4::new(pt.x, pt.y, pt.z, 1.0);
    let view_h = frame.v_inv * pt4;
    if view_h.z >= 0.0 { return None; }            // behind camera (-z is forward)
    let clip_h = frame.p * view_h;
    if clip_h.w.abs() < 1e-9 { return None; }
    let ndc_x = clip_h.x / clip_h.w;
    let ndc_y = clip_h.y / clip_h.w;
    let u = 0.5 * (ndc_x + 1.0);
    let v = 0.5 * (ndc_y + 1.0);
    if !(0.0..=1.0).contains(&u) || !(0.0..=1.0).contains(&v) {
        return None;
    }

    let dist = (pt - frame.cam_origin).length();
    if !(near..=far + tol).contains(&dist) { return None; }

    // (u, v) → depth-source pixel coords. Phone goes through Bd to land
    // on the depth-buffer grid; blend names colour-grid pixels directly
    // (the blend is computed at colour resolution and stored in the
    // frame.depth slot, also in GL row order).
    let (bx, by) = if frame.uses_bd {
        let nv = DVec4::new(u, v, 0.0, 1.0);
        let nd = frame.bd * nv;
        if nd.w.abs() < 1e-9 { return None; }
        let u_d = nd.x / nd.w;
        let v_d = nd.y / nd.w;
        ((1.0 - u_d) * frame.w as f64, v_d * frame.h as f64)
    } else {
        (u * frame.w as f64 - 0.5, v * frame.h as f64 - 0.5)
    };
    let w_max = (frame.w - 1) as f64;
    let h_max = (frame.h - 1) as f64;
    if bx < 0.0 || bx > w_max || by < 0.0 || by > h_max { return None; }

    // Bilinear depth sample.
    let bxc = bx.clamp(0.0, w_max - 1e-3);
    let byc = by.clamp(0.0, h_max - 1e-3);
    let bx0 = bxc.floor() as usize;
    let by0 = byc.floor() as usize;
    let bx1 = (bx0 + 1).min(frame.w - 1);
    let by1 = (by0 + 1).min(frame.h - 1);
    let fx  = bxc - bx0 as f64;
    let fy  = byc - by0 as f64;

    let row0 = by0 * frame.w;
    let row1 = by1 * frame.w;
    let d00 = frame.depth[row0 + bx0] as f64;
    let d10 = frame.depth[row0 + bx1] as f64;
    let d01 = frame.depth[row1 + bx0] as f64;
    let d11 = frame.depth[row1 + bx1] as f64;
    let depth_est = (d00 * (1.0 - fx) + d10 * fx) * (1.0 - fy)
                  + (d01 * (1.0 - fx) + d11 * fx) * fy;
    if depth_est <= near || depth_est >= far { return None; }

    let delta = dist - depth_est;
    if delta > tol { return None; }                // behind the surface
    if delta < -tol { return Some(Hit::Air); }

    // Bilinear colour sample (RGBA buffer, ignore A).
    let cx_pix = u * frame.cw as f64 - 0.5;
    let cy_pix = v * frame.ch as f64 - 0.5;
    let cw_max = (frame.cw - 1) as f64;
    let ch_max = (frame.ch - 1) as f64;
    let cxc = cx_pix.clamp(0.0, cw_max - 1e-3);
    let cyc = cy_pix.clamp(0.0, ch_max - 1e-3);
    let cx0 = cxc.floor() as usize;
    let cy0 = cyc.floor() as usize;
    let cx1 = (cx0 + 1).min(frame.cw - 1);
    let cy1 = (cy0 + 1).min(frame.ch - 1);
    let bxf = cxc - cx0 as f64;
    let byf = cyc - cy0 as f64;

    let pix = |x: usize, y: usize| -> [f64; 3] {
        let off = (y * frame.cw + x) * 4;
        [
            frame.color[off]     as f64,
            frame.color[off + 1] as f64,
            frame.color[off + 2] as f64,
        ]
    };
    let c00 = pix(cx0, cy0);
    let c10 = pix(cx1, cy0);
    let c01 = pix(cx0, cy1);
    let c11 = pix(cx1, cy1);

    let mut color = [0.0f64; 3];
    for k in 0..3 {
        color[k] = (c00[k] * (1.0 - bxf) + c10[k] * bxf) * (1.0 - byf)
                 + (c01[k] * (1.0 - bxf) + c11[k] * bxf) * byf;
    }
    Some(Hit::Color(color))
}

// ----------------------------------------------------------------------
// Blend context: shared per-session state for a Blend pass.
// ----------------------------------------------------------------------

struct BlendContext {
    model_dir: PathBuf,
    /// idx → (cw, ch) of the cached prediction; loaded from
    /// `<session>/model_raw/index.json`.
    model_dims: std::collections::HashMap<usize, (usize, usize)>,
    /// Gaussian sigma as a fraction of the colour-image diagonal.
    sigma_frac: f64,
}

impl BlendContext {
    fn from_session(session_dir: &Path, sigma_frac: f64,
                     model_version: ModelVersion) -> Result<Self, String> {
        let sub = model_version.cache_subdir();
        let model_dir = session_dir.join(sub);
        let index_path = model_dir.join("index.json");
        let bytes = fs::read(&index_path)
            .map_err(|e| format!("{sub}/index.json read failed: {e}"))?;
        let v: serde_json::Value = serde_json::from_slice(&bytes)
            .map_err(|e| format!("{sub}/index.json parse failed: {e}"))?;
        let obj = v.as_object()
            .ok_or_else(|| format!("{sub}/index.json must be an object"))?;
        let mut model_dims = std::collections::HashMap::new();
        for (k, entry) in obj.iter() {
            let idx: usize = match k.parse() {
                Ok(n) => n,
                Err(_) => continue,
            };
            let w = entry.get("w").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
            let h = entry.get("h").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
            if w > 0 && h > 0 {
                model_dims.insert(idx, (w, h));
            }
        }
        Ok(BlendContext { model_dir, model_dims, sigma_frac })
    }
}

// ----------------------------------------------------------------------
// CLI / driver.
// ----------------------------------------------------------------------

struct Args {
    // Two ways to specify input:
    //   --session <id> [--frames-root <dir>]   → run twice
    //                                            (frames/ + frames_refined/),
    //                                            outputs go into the session dir
    //   --frames-dir <dir> [--out <path>]      → single ad-hoc run
    session: Option<String>,
    frames_root: PathBuf,
    frames_dir: Option<PathBuf>,
    out: Option<PathBuf>,
    voxel_size: f64,
    world_min: [f64; 3],
    world_max: [f64; 3],
    near: f64,
    far: f64,
    tol: f64,
    threshold: f64,
    min_color_count: u32,
    max_frames: Option<usize>,
    /// Which depth source feeds the voxel projection.
    depth_source: DepthSource,
    /// Gaussian sigma (fraction of the colour-image diagonal) for the
    /// blend low-pass — only used when `depth_source == Blend`.
    blend_sigma: f64,
    /// Which model_raw cache directory to read from. Only meaningful
    /// when `depth_source == Blend`. V2 is the default for backward
    /// compatibility.
    model_version: ModelVersion,
}

impl Args {
    fn parse() -> Args {
        let mut a = Args {
            session: None,
            frames_root: PathBuf::from("captured_frames"),
            frames_dir: None,
            out: None,
            voxel_size: 0.02,
            world_min: [-2.5, -0.3, -2.5],
            world_max: [ 2.5,  4.7,  2.5],
            near: 0.05,
            far: 8.0,
            tol: 0.03,
            threshold: 0.20,
            min_color_count: 3,
            max_frames: None,
            depth_source: DepthSource::Phone,
            blend_sigma: 0.03,
            model_version: ModelVersion::V2,
        };
        let args: Vec<String> = env::args().skip(1).collect();
        let mut i = 0;
        while i < args.len() {
            match args[i].as_str() {
                "--session"     => { a.session     = Some(args[i+1].clone());        i += 2; }
                "--frames-root" => { a.frames_root = PathBuf::from(args[i+1].clone()); i += 2; }
                "--frames-dir"  => { a.frames_dir  = Some(PathBuf::from(args[i+1].clone())); i += 2; }
                "--out"         => { a.out         = Some(PathBuf::from(args[i+1].clone())); i += 2; }
                "--voxel-size"  => { a.voxel_size  = args[i+1].parse().unwrap();     i += 2; }
                "--near"        => { a.near        = args[i+1].parse().unwrap();     i += 2; }
                "--far"         => { a.far         = args[i+1].parse().unwrap();     i += 2; }
                "--tol"         => { a.tol         = args[i+1].parse().unwrap();     i += 2; }
                "--threshold"   => { a.threshold   = args[i+1].parse().unwrap();     i += 2; }
                "--min-color-count" => { a.min_color_count = args[i+1].parse().unwrap(); i += 2; }
                "--max-frames"  => { a.max_frames  = Some(args[i+1].parse().unwrap()); i += 2; }
                "--depth-source" => {
                    a.depth_source = match args[i+1].as_str() {
                        "phone" => DepthSource::Phone,
                        "blend" => DepthSource::Blend,
                        other => {
                            eprintln!("--depth-source must be 'phone' or 'blend', got {other:?}");
                            std::process::exit(2);
                        }
                    };
                    i += 2;
                }
                "--blend-sigma" => { a.blend_sigma = args[i+1].parse().unwrap();      i += 2; }
                "--model-version" => {
                    a.model_version = match args[i+1].as_str() {
                        "v2" => ModelVersion::V2,
                        "v3" => ModelVersion::V3,
                        other => {
                            eprintln!("--model-version must be 'v2' or 'v3', got {other:?}");
                            std::process::exit(2);
                        }
                    };
                    i += 2;
                }
                "--world-min"   => {
                    a.world_min = [
                        args[i+1].parse().unwrap(),
                        args[i+2].parse().unwrap(),
                        args[i+3].parse().unwrap(),
                    ];
                    i += 4;
                }
                "--world-max"   => {
                    a.world_max = [
                        args[i+1].parse().unwrap(),
                        args[i+2].parse().unwrap(),
                        args[i+3].parse().unwrap(),
                    ];
                    i += 4;
                }
                "-h" | "--help" => {
                    eprintln!("usage: voxel-reverse \\
        --session <id> [--frames-root DIR]      # session mode (writes both variants) \\
      | --frames-dir DIR [--out PATH]           # ad-hoc mode \\
        [--voxel-size M] [--world-min X Y Z] [--world-max X Y Z] \\
        [--near M] [--far M] [--tol M] \\
        [--threshold R] [--min-color-count N] [--max-frames N] \\
        [--depth-source phone|blend] [--blend-sigma 0.03] \\
        [--model-version v2|v3]                  # which model_raw cache to read");
                    std::process::exit(0);
                }
                other => {
                    eprintln!("unknown arg: {other}");
                    std::process::exit(2);
                }
            }
        }
        a
    }
}

/// Returns true if a JSON file was written, false if the input dir was empty
/// or yielded no kept voxels.
// blend_ctx: when `Some`, depth source is the blend at colour resolution;
// the context holds a path to `<session>/model_raw/` and an idx → (cw, ch)
// map so frames whose model_raw cache is missing can be dropped.
fn run_pass(
    label: &str,
    frames_dir: &Path,
    out_path: &Path,
    args: &Args,
    blend_ctx: Option<&BlendContext>,
) -> bool {
    println!("=== {label} ===");
    println!("frames-dir: {:?}", frames_dir);
    println!("out:        {:?}", out_path);

    // Load + sort frame paths.
    let mut frame_paths: Vec<PathBuf> = match fs::read_dir(frames_dir) {
        Ok(rd) => rd
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.file_name()
                    .and_then(|s| s.to_str())
                    .map(|s| s.starts_with("frame_") && s.ends_with(".bin"))
                    .unwrap_or(false)
            })
            .collect(),
        Err(e) => {
            eprintln!("cannot read frames-dir {:?}: {}", frames_dir, e);
            return false;
        }
    };
    frame_paths.sort();
    if let Some(n) = args.max_frames {
        frame_paths.truncate(n);
    }
    if frame_paths.is_empty() {
        eprintln!("no frames found in {:?}", frames_dir);
        return false;
    }

    // Decode frames in parallel.
    let mode_label = if blend_ctx.is_some() { "phone+model→blend" } else { "phone" };
    println!("Decoding {} frames ({mode_label})…", frame_paths.len());
    let t_dec = Instant::now();
    let frames: Vec<Frame> = frame_paths.par_iter()
        .filter_map(|p| {
            // Extract idx from "frame_NNNNNN.bin" so we can look up the
            // matching model_raw cache file when blending.
            let stem = p.file_stem().and_then(|s| s.to_str())?;
            let digits = stem.trim_start_matches("frame_");
            let idx: usize = digits.parse().ok()?;
            let bytes = match fs::read(p) {
                Ok(b) => b,
                Err(e) => {
                    eprintln!("  read error on {:?}: {}", p.file_name().unwrap(), e);
                    return None;
                }
            };
            let parsed = match blend_ctx {
                None => parse_frame(&bytes).map(|f| f),
                Some(ctx) => {
                    let dims = match ctx.model_dims.get(&idx) {
                        Some(d) => *d,
                        None => {
                            eprintln!("  no model_raw cache for frame {idx}, skipping");
                            return None;
                        }
                    };
                    let raw_path = ctx.model_dir.join(format!("frame_{idx:06}.f16"));
                    parse_frame_with_blend(&bytes, &raw_path, dims.0, dims.1, ctx.sigma_frac)
                }
            };
            match parsed {
                Ok(Some(f)) => Some(f),
                Ok(None)    => None,
                Err(e)      => {
                    eprintln!("  parse error on {:?}: {}", p.file_name().unwrap(), e);
                    None
                }
            }
        })
        .collect();
    println!("  decoded {}/{} frames in {:.2} s",
             frames.len(), frame_paths.len(),
             t_dec.elapsed().as_secs_f64());
    if frames.is_empty() { return false; }
    if let Some(f0) = frames.first() {
        let depth_label = if f0.uses_bd { "phone-depth" } else { "blend-depth" };
        println!("  {} {}×{}, color {}×{}",
                 depth_label, f0.h, f0.w, f0.ch, f0.cw);
    }

    // Voxel grid.
    let wmin = DVec3::from_array(args.world_min);
    let wmax = DVec3::from_array(args.world_max);
    let nx = ((wmax.x - wmin.x) / args.voxel_size).ceil() as usize;
    let ny = ((wmax.y - wmin.y) / args.voxel_size).ceil() as usize;
    let nz = ((wmax.z - wmin.z) / args.voxel_size).ceil() as usize;
    let n_total = nx * ny * nz;
    let n_slab  = ny * nz;
    println!("Voxel grid: ({nx}, {ny}, {nz}) = {n_total} voxels @ {} cm/edge",
             args.voxel_size * 100.0);

    let mut color_count = vec![0u32; n_total];
    let mut air_count   = vec![0u32; n_total];
    let mut color_sum   = vec![[0.0f64; 3]; n_total];

    let near = args.near; let far = args.far; let tol = args.tol;

    println!("Processing voxels (parallel over {nx} ix slabs)…");
    let t_proc = Instant::now();
    color_count.par_chunks_mut(n_slab)
        .zip(air_count.par_chunks_mut(n_slab))
        .zip(color_sum.par_chunks_mut(n_slab))
        .enumerate()
        .for_each(|(ix, ((cc, ac), cs))| {
            let cx = wmin.x + (ix as f64 + 0.5) * args.voxel_size;
            for iy in 0..ny {
                let cy = wmin.y + (iy as f64 + 0.5) * args.voxel_size;
                let row_off = iy * nz;
                for iz in 0..nz {
                    let cz = wmin.z + (iz as f64 + 0.5) * args.voxel_size;
                    let pt = DVec3::new(cx, cy, cz);
                    let off = row_off + iz;
                    for frame in &frames {
                        match project_voxel_to_frame(pt, frame, near, far, tol) {
                            Some(Hit::Color(c)) => {
                                cc[off] += 1;
                                cs[off][0] += c[0];
                                cs[off][1] += c[1];
                                cs[off][2] += c[2];
                            }
                            Some(Hit::Air) => { ac[off] += 1; }
                            None => {}
                        }
                    }
                }
            }
        });
    let dt_proc = t_proc.elapsed();
    let total_color: u64 = color_count.iter().map(|&v| v as u64).sum();
    let total_air:   u64 = air_count.iter().map(|&v| v as u64).sum();
    println!("Processed in {:.2} s (color {} / air {})",
             dt_proc.as_secs_f64(), total_color, total_air);

    // Threshold + average.
    let mut indices: Vec<[i32; 3]> = Vec::new();
    let mut colors:  Vec<[u8; 3]>  = Vec::new();
    for ix in 0..nx {
        for iy in 0..ny {
            for iz in 0..nz {
                let i = ix * n_slab + iy * nz + iz;
                let cc = color_count[i];
                let ac = air_count[i];
                if cc < args.min_color_count { continue; }
                let total = cc as f64 + ac as f64;
                let ratio = if total > 0.0 { cc as f64 / total } else { 0.0 };
                if ratio < args.threshold { continue; }
                let cs = color_sum[i];
                let safe = cc.max(1) as f64;
                let r = (cs[0] / safe).clamp(0.0, 255.0) as u8;
                let g = (cs[1] / safe).clamp(0.0, 255.0) as u8;
                let b = (cs[2] / safe).clamp(0.0, 255.0) as u8;
                indices.push([ix as i32, iy as i32, iz as i32]);
                colors.push([r, g, b]);
            }
        }
    }
    let n_kept = indices.len();
    println!("Kept {n_kept} of {n_total} voxels (ratio >= {}, color_count >= {})",
             args.threshold, args.min_color_count);
    if n_kept == 0 { return false; }

    // JSON output (matches the Python schema).
    if let Some(parent) = out_path.parent() {
        if !parent.as_os_str().is_empty() {
            let _ = fs::create_dir_all(parent);
        }
    }
    let payload = json!({
        "voxel_size": args.voxel_size,
        "world_min": [wmin.x, wmin.y, wmin.z],
        "world_max": [
            wmin.x + nx as f64 * args.voxel_size,
            wmin.y + ny as f64 * args.voxel_size,
            wmin.z + nz as f64 * args.voxel_size,
        ],
        "shape": [nx, ny, nz],
        "threshold": args.threshold,
        "n_voxels": n_kept,
        "indices": indices,
        "colors":  colors,
    });
    let s = serde_json::to_string(&payload).unwrap();
    let mut f = fs::File::create(out_path).unwrap();
    f.write_all(s.as_bytes()).unwrap();
    let size_mb = fs::metadata(out_path).map(|m| m.len() as f64 / 1e6).unwrap_or(0.0);
    println!("Wrote {:?}  ({:.1} MB)", out_path, size_mb);
    true
}

fn main() {
    let args = Args::parse();
    println!("rayon threads: {}", rayon::current_num_threads());

    // Output filename suffix per (depth_source, model_version).
    //   Phone           → ""
    //   Blend + V2      → "_blended"      (unchanged: keeps existing filenames)
    //   Blend + V3      → "_blended_v3"   (parallel to the V2 outputs)
    let blend_suffix: &str = match (args.depth_source, args.model_version) {
        (DepthSource::Phone, _)                => "",
        (DepthSource::Blend, ModelVersion::V2) => "_blended",
        (DepthSource::Blend, ModelVersion::V3) => "_blended_v3",
    };

    if let Some(session_id) = &args.session {
        // Session mode: produce voxels_original.json from frames/, plus
        // voxels_refined.json (from frames_refined/) and voxels_aligned.json
        // (from frames_feature_ba/, the bundle-adjustment output) if those
        // sibling directories exist.
        //
        // Note: voxels_aligned.json now sources from frames_feature_ba/ —
        // i.e. the per-frame SE(3) corrections produced by
        // tools/feature_pose_align.py — rather than the depth-ICP-derived
        // frames_aligned/ that loop_closure_analyze.py used to write.
        // BA on feature reprojection error gives sub-cm pose corrections
        // that are tighter than ICP-on-depth, especially on captures with
        // good parallax (orthogonal-to-motion handheld scans).
        //
        // When --depth-source=blend, every variant gets a `_blended`
        // suffix on its output JSON and is fed the per-frame Gaussian
        // detail-injection blend instead of the WebXR depth.
        let session_dir = args.frames_root.join(session_id);
        let blend_ctx = match args.depth_source {
            DepthSource::Phone => None,
            DepthSource::Blend => Some(
                BlendContext::from_session(&session_dir, args.blend_sigma,
                                            args.model_version)
                    .unwrap_or_else(|e| {
                        eprintln!("blend mode: {e}");
                        std::process::exit(1);
                    })
            ),
        };
        let blend_ref = blend_ctx.as_ref();
        let mut wrote_any = false;
        let original_dir = session_dir.join("frames");
        if original_dir.exists() {
            let out = session_dir.join(format!("voxels_original{blend_suffix}.json"));
            if run_pass("original", &original_dir, &out, &args, blend_ref) {
                wrote_any = true;
            }
        } else {
            eprintln!("session has no frames/ subdir at {:?}", original_dir);
        }
        let refined_dir = session_dir.join("frames_refined");
        if refined_dir.exists() {
            let out = session_dir.join(format!("voxels_refined{blend_suffix}.json"));
            if run_pass("refined", &refined_dir, &out, &args, blend_ref) {
                wrote_any = true;
            }
        } else {
            println!("(no frames_refined/ — run tools/depth_refine.py first to add it)");
        }
        let aligned_dir = session_dir.join("frames_feature_ba");
        if aligned_dir.exists() {
            let out = session_dir.join(format!("voxels_aligned{blend_suffix}.json"));
            if run_pass("aligned (BA)", &aligned_dir, &out, &args, blend_ref) {
                wrote_any = true;
            }
        } else {
            println!("(no frames_feature_ba/ — run \
                     tools/feature_ray_reconstruct.py --session <id> and then \
                     tools/feature_pose_align.py --session <id> to produce it)");
        }
        // Refined depth + BA poses: depth_refine.py with
        // --frames-variant frames_feature_ba --anchor features writes
        // here. The depth maps are dense (colour-resolution) Depth-Anything
        // outputs that have been per-frame-affine-aligned to the
        // BA-triangulated 3D feature points — i.e. a metric scale that
        // matches the BA reconstruction rather than ARCore's own depth.
        let refined_aligned_dir = session_dir.join("frames_refined_feature_ba");
        if refined_aligned_dir.exists() {
            let out = session_dir.join(format!("voxels_refined_aligned{blend_suffix}.json"));
            if run_pass("refined+aligned (model depth, BA poses, feature anchors)",
                        &refined_aligned_dir, &out, &args, blend_ref) {
                wrote_any = true;
            }
        } else {
            println!("(no frames_refined_feature_ba/ — run \
                     tools/depth_refine.py --session <id> --frames-variant \
                     frames_feature_ba --anchor features to produce it)");
        }
        if !wrote_any {
            std::process::exit(1);
        }
    } else {
        // Ad-hoc mode (single dir → single output).
        // Note: blend mode in ad-hoc mode is unsupported — there's no
        // session_dir to pull the model_raw cache from. Reject early.
        if args.depth_source == DepthSource::Blend {
            eprintln!("--depth-source=blend requires --session (model_raw cache lives in <session>/model_raw/)");
            std::process::exit(2);
        }
        let frames_dir = args.frames_dir.clone()
            .unwrap_or_else(|| PathBuf::from("captured_frames"));
        let out = args.out.clone()
            .unwrap_or_else(|| PathBuf::from("web/out/voxels.json"));
        if !run_pass("ad-hoc", &frames_dir, &out, &args, None) {
            std::process::exit(1);
        }
    }
}
