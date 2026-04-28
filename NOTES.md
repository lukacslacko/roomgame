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

### Spurious feature track 147–150 + 190 — session `20260427_123301`, `features` variant

**Snapshot.** As of commit `6d09afc` (with `--min-depth 0.0` cheirality
on by default and `--max-sensitivity 0.10`), a feature track linking
frames 147, 148, 149, 150 and 190 still triangulates to a floating
mid-air voxel. It survives every track filter currently in
`tools/feature_ray_reconstruct.py`.

**Symptom.** A clearly mid-air voxel in the `features` variant whose
contributing observations are frames 147–150 (a contiguous run) plus
the disconnected frame 190. The voxel's `obs` UVs land on visually
unrelated patches in the thumbnails — i.e. it is a genuine
mismatch, not an ill-conditioned-but-real feature.

**What this rules out.**

- Cheirality alone — every contributing ray sees the (incorrect)
  point in front of its camera, so `tᵢ > 0` doesn't catch it.
- 1-px sensitivity at the 0.10 m cutoff — the spurious geometry is
  consistent enough across the five views that `σ_1px` falls below
  the threshold. So whatever's happening, it's *not* an
  underdetermined triangulation: the false rays really do
  near-intersect.

**Likely culprits, in order of suspicion.**

1. **Repeated texture matched across non-adjacent timestamps.** A
   147–150 contiguous run plus 190 alone is the signature of an ORB
   descriptor that matches one physical patch in the close run *and*
   a visually similar but different patch back at frame 190. Lowe
   ratio (currently `0.75`) plus pure descriptor matching can't tell
   the two apart.
2. **Reflective / view-dependent surface.** A specular highlight or
   monitor patch can shift slightly between frames yet remain
   ratio-test-stable; the false 3D point is then the apparent
   intersection of the highlight rays.
3. **Dense same-object texture wrapping a real edge.** Less likely
   given the disjoint time-of-observation pattern.

**How to use this case.**

1. Re-run the pipeline at the snapshot:
   `git checkout 6d09afc && .venv/bin/python tools/feature_ray_reconstruct.py --session 20260427_123301 --max-sensitivity 0.10`.
2. Open the voxel viewer, pick that session, switch the variant
   dropdown to `features`.
3. Look for the floater near where frames 147–150's frusta cluster
   (open the frames panel and click those four to render the frusta —
   they'll point at the same room region).
4. Click the floater. The panel should auto-select 147–150 + 190 and
   draw yellow rings on each thumbnail at the alleged source
   keypoint. Eyeball the rings: the ones on 147–150 will land in one
   place and the one on 190 will land somewhere visibly different.

**Diagnostic next steps.**

- Add a *temporal-coherence* filter: after triangulation, reject
  tracks where the observed frame indices fall into ≥2 disjoint
  clusters separated by a gap larger than some threshold (e.g. 20+
  frames). The 147–150 + 190 split is exactly this signature.
- Tighten the Lowe ratio (`--ratio 0.65`) and re-run; if the floater
  goes away the matching is borderline rather than structurally
  wrong.
- Add a per-pair geometric verification step (essential matrix +
  RANSAC inliers) before unioning matches into tracks. The 147↔190
  pair would likely fail this where 147↔148 wouldn't.
- Use the per-voxel `features[].sensitivity` and `n_views` in
  `features_meta.json` to confirm the spurious track has *low*
  sensitivity (i.e. high confidence) — that's what makes it
  pernicious; it can't be rejected on conditioning alone.
