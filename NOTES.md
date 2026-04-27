# Roomgame — debugging notes

A running list of concrete reproductions worth checking against any
algorithmic change to the depth-refinement / pose-graph / voxelisation
pipeline. Each entry pairs an observable symptom with the simplest way
to reproduce it, so we can tell whether a fix actually helps.

## Regression cases

### Kitchen-counter drift — session `20260427_123301`, frames 149 vs 122

**Symptom.** Frames 149 and 122 view the same patch of kitchen counter
but their per-frame voxel projections place it noticeably far apart in
world space. Frame 149 additionally shows the floor voxels as strongly
slanted (not horizontal).

**Persists in.** `refined`, `refined_aligned` (and presumably the
unaligned `refined`/`original` too).

**What this rules out.** Loop closure fixes pose drift; the
disagreement persisting in `refined_aligned` means the residual error
is **not** primarily pose drift on this pair.

**Likely culprits, in order of suspicion.**

1. **Per-frame model-depth scale `(a, b)`.** Depth Anything's affine
   was fit per frame against phone depth in `tools/depth_refine.py`.
   On comparable sessions `a` ranged 0.39–0.89 — large enough that
   neighbouring frames can encode the same world surface at meaningfully
   different metric distances. The slanted floor on 149 specifically
   suggests a depth profile whose scale isn't constant across the image
   (a per-frame *scalar* affine can't compensate that).
2. **Lens distortion not modelled.** `tools/camera_check.py` showed
   ~4 px reprojection error in the outer 25 % radial bin (mild
   distortion). Floor on 149 might be an edge-of-frame effect from this.
3. **Colour / depth alignment via `Bd`.** Trusted as correct so far;
   if it's slightly off, depth values get attributed to the wrong
   colour-pixel positions during voxelisation.

**How to use this case.**

1. Start `tools/serve.py`, open `voxelview.html`, pick session
   `20260427_123301`.
2. Cycle through the four variants (`original`, `refined`, `aligned`,
   `refined_aligned`). For each, open the frame-debug panel and select
   frames 149 and 122. The two highlight clouds should overlap on the
   kitchen counter; today they don't.
3. After any change to the depth or intrinsics pipeline, reload and
   repeat — the disagreement on this exact pair should shrink for the
   change to count as a real win.

**Diagnostic next steps for the underlying issue.**

- Print the per-frame `(a, b)` for frames 149 and 122 from
  `tools/depth_refine.py`'s log. If they differ a lot, that's
  hypothesis (1).
- Use `tools/loop_closure_analyze.py` (or its ICP machinery) to
  pairwise-register 149 against 122 directly. Whatever residual
  remains after the rigid (R, t) correction is the part **not**
  explained by pose — i.e. depth or intrinsics error.
- If chasing intrinsics: extend `tools/camera_check.py` to fit a
  `(k1, k2)` radial distortion model from the same reprojection
  residuals, then rebuild the voxel pipeline against OpenCV-undistorted
  colour + depth and re-check this pair.
