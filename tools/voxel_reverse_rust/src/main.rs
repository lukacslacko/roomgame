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
use std::path::PathBuf;
use std::time::Instant;

use glam::{DMat4, DVec3, DVec4};
use rayon::prelude::*;
use serde_json::json;

const FRAME_HEADER_SIZE: usize = 224;
const DEPTH_FMT_UINT16_LA: u32 = 0;
const DEPTH_FMT_FLOAT32:   u32 = 1;
const COLOR_FMT_RGBA8:     u32 = 1;

struct Frame {
    v_inv: DMat4,
    p: DMat4,
    bd: DMat4,
    cam_origin: DVec3,
    depth: Vec<f32>,    // (h, w) row-major
    color: Vec<u8>,     // (ch, cw) RGBA, row-major
    w: usize,
    h: usize,
    cw: usize,
    ch: usize,
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
    }))
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

    // (u, v) → depth-buffer pixel coords via Bd.
    let nv = DVec4::new(u, v, 0.0, 1.0);
    let nd = frame.bd * nv;
    if nd.w.abs() < 1e-9 { return None; }
    let u_d = nd.x / nd.w;
    let v_d = nd.y / nd.w;
    let bx = (1.0 - u_d) * frame.w as f64;
    let by = v_d * frame.h as f64;
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
// CLI / driver.
// ----------------------------------------------------------------------

struct Args {
    frames_dir: PathBuf,
    out: PathBuf,
    voxel_size: f64,
    world_min: [f64; 3],
    world_max: [f64; 3],
    near: f64,
    far: f64,
    tol: f64,
    threshold: f64,
    min_color_count: u32,
    max_frames: Option<usize>,
}

impl Args {
    fn parse() -> Args {
        let mut a = Args {
            frames_dir: PathBuf::from("captured_frames"),
            out: PathBuf::from("web/out/voxels.json"),
            voxel_size: 0.05,
            world_min: [-2.5, -0.3, -2.5],
            world_max: [ 2.5,  4.7,  2.5],
            near: 0.05,
            far: 8.0,
            tol: 0.20,
            threshold: 0.10,
            min_color_count: 1,
            max_frames: None,
        };
        let mut args: Vec<String> = env::args().skip(1).collect();
        let mut i = 0;
        while i < args.len() {
            match args[i].as_str() {
                "--frames-dir" => { a.frames_dir = PathBuf::from(args[i+1].clone()); i += 2; }
                "--out"        => { a.out        = PathBuf::from(args[i+1].clone()); i += 2; }
                "--voxel-size" => { a.voxel_size = args[i+1].parse().unwrap();        i += 2; }
                "--near"       => { a.near       = args[i+1].parse().unwrap();        i += 2; }
                "--far"        => { a.far        = args[i+1].parse().unwrap();        i += 2; }
                "--tol"        => { a.tol        = args[i+1].parse().unwrap();        i += 2; }
                "--threshold"  => { a.threshold  = args[i+1].parse().unwrap();        i += 2; }
                "--min-color-count" => { a.min_color_count = args[i+1].parse().unwrap(); i += 2; }
                "--max-frames" => { a.max_frames = Some(args[i+1].parse().unwrap()); i += 2; }
                "--world-min"  => {
                    a.world_min = [
                        args[i+1].parse().unwrap(),
                        args[i+2].parse().unwrap(),
                        args[i+3].parse().unwrap(),
                    ];
                    i += 4;
                }
                "--world-max"  => {
                    a.world_max = [
                        args[i+1].parse().unwrap(),
                        args[i+2].parse().unwrap(),
                        args[i+3].parse().unwrap(),
                    ];
                    i += 4;
                }
                "-h" | "--help" => {
                    eprintln!("usage: voxel-reverse [--frames-dir DIR] [--out PATH] \\
        [--voxel-size M] [--world-min X Y Z] [--world-max X Y Z] \\
        [--near M] [--far M] [--tol M] \\
        [--threshold R] [--min-color-count N] [--max-frames N]");
                    std::process::exit(0);
                }
                other => {
                    eprintln!("unknown arg: {other}");
                    std::process::exit(2);
                }
            }
            args.truncate(args.len()); // silence unused_assignments lints
        }
        a
    }
}

fn main() {
    let args = Args::parse();

    println!("rayon threads: {}", rayon::current_num_threads());

    // Load frame paths.
    let mut frame_paths: Vec<PathBuf> = match fs::read_dir(&args.frames_dir) {
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
            eprintln!("cannot read frames-dir {:?}: {}", args.frames_dir, e);
            std::process::exit(1);
        }
    };
    frame_paths.sort();
    if let Some(n) = args.max_frames {
        frame_paths.truncate(n);
    }
    if frame_paths.is_empty() {
        eprintln!("no frames found in {:?}", args.frames_dir);
        std::process::exit(1);
    }

    // Decode frames in parallel.
    println!("Decoding {} frames…", frame_paths.len());
    let t_dec = Instant::now();
    let frames: Vec<Frame> = frame_paths.par_iter()
        .filter_map(|p| {
            match fs::read(p) {
                Ok(bytes) => match parse_frame(&bytes) {
                    Ok(Some(f)) => Some(f),
                    Ok(None)    => None,
                    Err(e)      => {
                        eprintln!("  parse error on {:?}: {}", p.file_name().unwrap(), e);
                        None
                    }
                }
                Err(e) => {
                    eprintln!("  read error on {:?}: {}", p.file_name().unwrap(), e);
                    None
                }
            }
        })
        .collect();
    println!("  decoded {}/{} frames in {:.2} s",
             frames.len(), frame_paths.len(),
             t_dec.elapsed().as_secs_f64());
    if frames.is_empty() { return; }
    if let Some(f0) = frames.first() {
        println!("  depth {}×{}, color {}×{}", f0.h, f0.w, f0.ch, f0.cw);
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
    if n_kept == 0 { return; }

    // JSON output (matches the Python schema).
    if let Some(parent) = args.out.parent() {
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
    let mut f = fs::File::create(&args.out).unwrap();
    f.write_all(s.as_bytes()).unwrap();
    let size_mb = fs::metadata(&args.out).map(|m| m.len() as f64 / 1e6).unwrap_or(0.0);
    println!("Wrote {:?}  ({:.1} MB)", args.out, size_mb);
}
