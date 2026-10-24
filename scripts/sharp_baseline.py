"""Unmodified SHARP prediction on the reference photograph, with reusable tensors."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'vendor/ml-sharp/src'))
import time
import json
import numpy as np
import torch
import torch.nn.functional as F
from sharp.models import PredictorParams, create_predictor
from sharp.utils.io import load_rgb
from sharp.utils.gaussians import save_ply, unproject_gaussians

OUT = ROOT / 'output/sharp-multiview'
OUT.mkdir(exist_ok=True, parents=True)
source = Path(sys.argv[1])
started = time.perf_counter()
torch.set_num_threads(8)
device = torch.device('mps')
assert torch.backends.mps.is_available()
image, _, focal = load_rgb(source)
focal = float(focal)
h, w = image.shape[:2]
print(f'Image {w}x{h}, EXIF focal {focal:.3f}px', flush=True)
model = create_predictor(PredictorParams())
checkpoint=Path.home()/'.cache/torch/hub/checkpoints/sharp_2572gikvuh.pt'
if checkpoint.exists():
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
else:
    state = torch.hub.load_state_dict_from_url('https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt',map_location='cpu')
model.load_state_dict(state)
del state
model.eval().to(device)
captured = {}
def capture_init(module, args):
    captured['monodepth'] = args[1].detach().cpu()
handle = model.init_model.register_forward_pre_hook(capture_init)
with torch.inference_mode():
    rgb = torch.from_numpy(image.copy()).float().permute(2,0,1)[None].to(device)/255
    rgb = F.interpolate(rgb, size=(1536,1536), mode='bilinear', align_corners=True)
    ndc = model(rgb, torch.tensor([focal/w], device=device, dtype=torch.float32))
    K = torch.tensor([[focal*1536/w,0,768,0],[0,focal*1536/h,768,0],[0,0,1,0],[0,0,0,1]],device=device,dtype=torch.float32)
    g = unproject_gaussians(ndc, torch.eye(4,device=device), K, (1536,1536)).to('cpu')
handle.remove()
torch.save({**g._asdict(), **captured, 'focal':focal,'height':h,'width':w,'source':str(source)}, OUT/'baseline.pt')
save_ply(g, focal, (h,w), OUT/'sharp-baseline.ply')
np.save(OUT/'sharp-monodepth.npy', captured['monodepth'].numpy())
elapsed=time.perf_counter()-started
(OUT/'baseline-meta.json').write_text(json.dumps({'source':str(source),'focal_px':focal,'width':w,'height':h,'seconds':elapsed,'gaussians':g.mean_vectors.shape[-2],'device':'mps'},indent=2))
print(f'SHARP complete in {elapsed:.1f}s, means {g.mean_vectors.shape}',flush=True)
