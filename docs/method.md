# Method and measurements

DoubleSharp integrates the released Apple SHARP and Depth Anything 3 models at inference time. The reference photograph supplies SHARP image features; both photographs supply DA3 depth and camera-pose estimates.

## Depth preparation

DA3 Giant processes the two photos jointly at 504 × 336. Camera reprojection compares predicted depths across views. Pixels must agree within 8% and pass a confidence threshold of 1.4 in both views. Scale anchoring uses supported pixels in the lower 30% of the image and the central 70% of its width, where this example's pedestal lies. The median ratio of SHARP depth to DA3 depth estimates one global scale. These are estimated scene units, without an independent physical measurement.

The full-frame scaled depth is bilinearly resized to SHARP's 1536 × 1536 network input. SHARP's existing external-depth argument passes it into the learned depth-alignment module, which supplies Gaussian initialization. The feature network, prediction head and Gaussian composer then produce a fresh Gaussian scene. Standard camera unprojection maps that prediction into the exported scene coordinates.

## Verified run

Reconstructed from the exact published source-image copies on an Apple M4 Pro with 48 GB memory. Both copies have decoded pixels identical to the local originals; sensitive metadata fields were removed. Timings below are individual end-to-end script-stage measurements with cached checkpoints, including model loading and export where performed.

| Stage | Seconds |
| --- | ---: |
| Baseline SHARP | 17.15 |
| Joint DA3 depth and pose | 11.71 |
| Depth-conditioned SHARP | 19.35 |

- Baseline and DoubleSharp each contain **1,179,648 Gaussians**.
- Initializer, feature model, prediction head and Gaussian composer each ran exactly once in the conditioned pass.
- Every Gaussian output field is finite; all scales are positive.
- All five Gaussian output fields change from baseline: positions, sizes, orientations, colors and opacities.
- No geometric warp is applied to a completed baseline scene.
- The input depth is estimated jointly from both photographs; the SHARP checkpoint remains unchanged.

See [provenance.json](../examples/tree/provenance.json) for source hashes, model revisions, camera parameters and the full measured reconstruction report.

## Screenshot protocol

The paired images are original 1280 × 720 browser screenshots, with 45° horizontal orbit, zero pitch, target `(0, 0, -2)` and camera distance `4` in the estimated scene units. Each scene uses exactly the same camera and field of view. The camera is pulled back equally to keep the baseline's displaced canopy in frame. The WebGL preview uses the exported splat's color, opacity, scale and rotation parameters. There is no retouching or separate per-scene framing.

The 45° view reveals the baseline canopy/pedestal depth discontinuity. DoubleSharp reduces the separation on this capture. Both scenes have visible incomplete surfaces, floaters and disocclusion artifacts. A 45° orbit is a diagnostic stress view; small viewpoint changes are the more suitable presentation range for the underlying monocular reconstruction.

## Limits

This is a qualitative, capture-specific experiment with no depth ground truth or held-out-view benchmark. The second photo influences the depth prior; SHARP still predicts appearance from the reference image's features. The scale anchor assumes this example's pedestal framing and must be adapted for other subjects. Learned alignment can retain some monocular depth errors or suppress correct depth variation. Large orbits reveal missing surfaces.

Supplying external depth to SHARP uses an existing released model capability. This repository documents an integration and example, without claiming a new reconstruction architecture, newly trained weights or a novel depth-injection method.
