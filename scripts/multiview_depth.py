"""Two-view DA3 depth/pose inference on MPS; no Gaussian/CUDA renderer used."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'vendor/Depth-Anything-3/src'))
import time
import json
import numpy as np
import torch
from depth_anything_3.api import DepthAnything3

OUT=ROOT/'output/sharp-multiview'
OUT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(8)
paths=sys.argv[1:]
assert len(paths)==2
start=time.perf_counter()
from huggingface_hub import snapshot_download
checkpoint=Path(snapshot_download('depth-anything/DA3-GIANT',revision='7cd62ae9315b9dff094d2d300e4ad012640607dd',allow_patterns=['config.json','model.safetensors']))
print(f'Loading cached DA3 Giant: {checkpoint}',flush=True)
model=DepthAnything3.from_pretrained(str(checkpoint),local_files_only=True).eval().to('mps')
# Call the core network in float32: the high-level forward selects CUDA-oriented
# autocast defaults. The architecture/weights and preprocessing stay unchanged.
images,_,_=model.input_processor(paths,process_res=504,process_res_method='upper_bound_resize',sequential=True)
with torch.inference_mode():
    raw=model.model(images[None].to('mps'),infer_gs=False,ref_view_strategy='first')
    prediction=model.output_processor(raw)
np.savez_compressed(OUT/'multiview-depth.npz',depth=prediction.depth,confidence=prediction.conf,
                   intrinsics=prediction.intrinsics,extrinsics=prediction.extrinsics,
                   images=images.numpy())
(OUT/'depth-meta.json').write_text(json.dumps({'model':'DA3-GIANT','device':'mps','precision':'float32','seconds':time.perf_counter()-start,'sources':paths,'depth_shape':list(prediction.depth.shape)},indent=2))
print('Depth/pose complete:',prediction.depth.shape,time.perf_counter()-start,flush=True)
