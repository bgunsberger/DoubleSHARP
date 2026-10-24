# Third-party components and model licenses

The root Apache 2.0 license covers original DoubleSharp code and documentation.
Example photographs and rendered screenshots are excluded from that grant.
Each upstream component retains its own license. The complete reconstruction
pipeline is a research experiment with restricted model licenses.

| Component | Pinned source | Terms |
| --- | --- | --- |
| Apple SHARP source | [`aed6527`](https://github.com/apple/ml-sharp/tree/aed6527499ef91cba3b54c18d49a870f25947190) | [Apple software license](licenses/APPLE_SHARP_SOFTWARE.txt) and upstream acknowledgements |
| Apple SHARP checkpoint | `sharp_2572gikvuh.pt` from Apple's official CDN | [Apple Machine Learning Research Model License](licenses/APPLE_SHARP_MODEL.txt): noncommercial scientific research; product development is excluded |
| Depth Anything 3 source | [`3d835ec`](https://github.com/ByteDance-Seed/Depth-Anything-3/tree/3d835ec1a5802d64a8b8b15f817a1ab54809bfe4) | [Apache 2.0](licenses/DEPTH_ANYTHING_3_CODE.txt), with upstream component notices |
| DA3 Giant checkpoint | [`7cd62ae`](https://huggingface.co/depth-anything/DA3-GIANT/tree/7cd62ae9315b9dff094d2d300e4ad012640607dd) | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/); see the [model card](https://huggingface.co/depth-anything/DA3-GIANT) |
| WebGL viewer | [`free-splatter.cpp bf636c4`](https://github.com/localai-org/free-splatter.cpp/tree/bf636c4fc2cf4117649eccab64909d3937bbbcdb) | Apache 2.0; attribution in [NOTICE](NOTICE) |

Apple Machine Learning Research Model is licensed under the Apple Machine Learning
Research Model License Agreement. DoubleSharp supplies external depth through
SHARP's existing inference argument and reruns the released prediction network.
The SHARP checkpoint is used without retraining or changing its weights.

DA3 Giant is developed by ByteDance Seed Team. DoubleSharp uses its jointly
predicted depths, confidences and camera poses. The only local DA3 source patch
makes optional export imports lazy; see `patches/da3-lazy-export.patch`.

This independent project has no affiliation with or endorsement from Apple,
ByteDance, or the upstream project authors. Consult upstream acknowledgements
for dependencies downloaded during setup. Source/model checkouts, checkpoints,
and generated Gaussian scene files are excluded from the Git repository.
