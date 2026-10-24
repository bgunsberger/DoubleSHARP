# DoubleSharp

An experimental two-photo depth-conditioning pipeline for [Apple SHARP](https://github.com/apple/ml-sharp), running locally on Apple Silicon.

[Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) estimates depth jointly from both photos. DoubleSharp supplies that depth to SHARP **before Gaussian initialization**, then reruns its learned Gaussian prediction. SHARP's image features still come from the reference photograph.

## Tree example at 45°

Both images below use the same camera: 45° horizontal orbit, zero pitch, target `(0, 0, -2)`, distance `4` in the estimated scene units. These are unretouched WebGL screenshots of the reconstructed scenes.

| Baseline SHARP | DoubleSharp |
| --- | --- |
| ![Baseline SHARP at 45 degrees](docs/images/sharp-baseline-45.png) | ![DoubleSharp at 45 degrees](docs/images/doublesharp-45.png) |

Baseline SHARP places much of the canopy too far behind the pedestal, producing a visible separation when viewed from the side. In this capture, conditioning SHARP on joint two-view depth brings the canopy closer to the pedestal and reduces that separation. Floaters, incomplete surfaces, and disocclusion artifacts remain visible at this angle. This is one qualitative example with no depth ground truth.

### Source photographs

| Reference photo | Second view |
| --- | --- |
| [![Reference photograph](examples/tree/reference-preview.jpg)](examples/tree/2026_10_L1101215.jpg) | [![Second-view photograph](examples/tree/second-view-preview.jpg)](examples/tree/2026_10_L1101219.jpg) |

Both full-resolution **8368 × 5584** JPEGs are included in [examples/tree](examples/tree). Location and device-serial metadata have been removed from the published copies. Decoded image pixels and the focal-length metadata used by SHARP are preserved.

## Run locally

Tested on an Apple M4 Pro with 48 GB unified memory, Python 3.12, and PyTorch MPS. Install [uv](https://docs.astral.sh/uv/getting-started/installation/), Git, and Apple's command-line tools, then:

```bash
git clone https://github.com/bgunsberger/DoubleSharp.git
cd DoubleSharp
./setup-sharp-multiview.sh
./run-doublesharp.sh \
  examples/tree/2026_10_L1101215.jpg \
  examples/tree/2026_10_L1101219.jpg
./preview.sh
```

Open [the local comparison](http://127.0.0.1:8873/web/sharp-conditioned-comparison.html). Choose **45° comparison** to reproduce the illustrated angle. The [45° permalink](http://127.0.0.1:8873/web/sharp-conditioned-comparison.html?yaw=45&dist=4) also restores that camera.

Setup clones pinned upstream sources and installs the Python environment. The first inference downloads roughly 7.5 GB of SHARP and DA3 Giant checkpoints from their publishers. Neural inference runs on MPS. The WebGL preview runs in the browser. Input photographs stay local during reconstruction.

For your own experiment, replace the two file arguments with overlapping photographs of a static scene. The pedestal-based scale anchor is specific to this example; adapt `scripts/multiview_prior.py` for different framing or subjects.

Outputs are written under `output/sharp-multiview/`, including baseline and conditioned PLY/`.splat` files, reusable tensors, DA3 depth/pose predictions, and a reconstruction report. Running again replaces those experiment outputs. Generated scene files and checkpoints are excluded from this repository.

## How it works

1. Predict baseline SHARP geometry from the reference image and retain its depth estimate for scale anchoring.
2. Run DA3 Giant jointly on both images, at 504 × 336 per view, to estimate depth, confidence and relative camera poses.
3. Filter pedestal pixels using confidence and cross-view depth consistency, then estimate one global scale between DA3 and SHARP depth.
4. Pass the full-frame scaled depth into SHARP's existing `depth=` argument. Its learned alignment module determines the depth reaching Gaussian initialization.
5. Rerun the feature network, Gaussian prediction head, and composer; export the newly predicted scene.

This is inference-time integration of released models. It uses their existing learned weights. See [the method and measurements](docs/method.md) and [the example provenance](examples/tree/provenance.json).

## License

Original DoubleSharp code is **Apache 2.0**. The complete pipeline has separate model restrictions: **SHARP's checkpoint is restricted to noncommercial scientific research and excludes product development**, and **DA3 Giant is CC BY-NC 4.0**. The code license does not expand those model permissions. See [THIRD_PARTY.md](THIRD_PARTY.md), [NOTICE](NOTICE), and [licenses](licenses).

The example photographs and rendered screenshots are excluded from the source-code license. This is an independent research experiment using Apple SHARP and ByteDance's Depth Anything 3.
