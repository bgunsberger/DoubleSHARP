"""Regenerate SHARP Gaussians with multi-view depth inside its forward pass.

No Gaussian positions or sizes are edited after prediction. Native mode uses
SHARP's released learned external-depth alignment. Direct mode replaces its depth
tensor immediately before Gaussian initialization and reruns all Gaussian heads.
"""
import sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'vendor/ml-sharp/src'))
import numpy as np,torch
import torch.nn.functional as F
from sharp.models import PredictorParams,create_predictor
from sharp.utils.io import load_rgb
from sharp.utils.gaussians import unproject_gaussians,save_ply
from splat_export import export_splat
from multiview_prior import prepare_prior

OUT=ROOT/'output/sharp-multiview'
mode=sys.argv[1] if len(sys.argv)>1 else 'native'
assert mode in ('native','direct')
torch.set_num_threads(8)
start=time.perf_counter()
baseline=torch.load(OUT/'baseline.pt',weights_only=True)
data=np.load(OUT/'multiview-depth.npz')
scaled_depth,support,scale,anchor_count=prepare_prior(baseline,data)
# Full-frame depth from joint two-view inference, aligned to the pedestal's scale.
# This input does not contain the earlier post-warp correction or canopy fit.
prior=torch.from_numpy(scaled_depth.copy())[None,None]
prior=F.interpolate(prior,size=(1536,1536),mode='bilinear',align_corners=True)
assert torch.isfinite(prior).all() and prior.min()>0
np.save(OUT/'multiview-depth-input.npy',prior.numpy())
device=torch.device('mps')
model=create_predictor(PredictorParams())
weights=torch.load(Path.home()/'.cache/torch/hub/checkpoints/sharp_2572gikvuh.pt',map_location='cpu',weights_only=True)
model.load_state_dict(weights);del weights
model.eval().to(device)
rgb,_,focal=load_rgb(Path(baseline['source']));focal=float(focal)
h,w=rgb.shape[:2]
captured={};calls={'initializer':0,'feature_model':0,'prediction_head':0,'composer':0}
def init_hook(module,args):
    calls['initializer']+=1
    if mode=='direct':
        # Replace both visible-surface depth layers. Preserve relative occlusion
        # gaps from the mono predictor, rather than collapsing its hidden layer.
        mono=args[1]
        visible=prior.to(mono)
        depth=visible*(mono/mono[:,:1].clamp(min=1e-4))
        captured['depth']=depth.detach().cpu()
        return (args[0],depth)
    captured['depth']=args[1].detach().cpu()
def feature_hook(module,args):
    calls['feature_model']+=1
    captured['feature_input']=args[0].detach().cpu()
def count_head(module,args): calls['prediction_head']+=1
def count_composer(module,args): calls['composer']+=1
handles=[model.init_model.register_forward_pre_hook(init_hook),
         model.feature_model.register_forward_pre_hook(feature_hook),
         model.prediction_head.register_forward_pre_hook(count_head),
         model.gaussian_composer.register_forward_pre_hook(count_composer)]
with torch.inference_mode():
    image=torch.from_numpy(rgb.copy()).permute(2,0,1).float()[None].to(device)/255
    image=F.interpolate(image,size=(1536,1536),mode='bilinear',align_corners=True)
    disparity=torch.tensor([focal/w],dtype=torch.float32,device=device)
    print(f'Running SHARP {mode} conditioning on MPS',flush=True)
    prediction=model(image,disparity,depth=prior.to(device) if mode=='native' else None)
    K=torch.tensor([[focal*1536/w,0,768,0],[0,focal*1536/h,768,0],[0,0,1,0],[0,0,0,1]],device=device,dtype=torch.float32)
    gaussians=unproject_gaussians(prediction,torch.eye(4,device=device),K,(1536,1536)).to('cpu')
for handle in handles:handle.remove()
assert all(v==1 for v in calls.values()),calls
state={**gaussians._asdict(),'monodepth':captured['depth'],'focal':focal,'width':w,'height':h,'source':baseline['source']}
for key in gaussians._fields:assert torch.isfinite(state[key]).all(),key
assert (state['singular_values']>0).all()
torch.save(state,OUT/f'conditioned-{mode}.pt')
save_ply(gaussians,focal,(h,w),OUT/f'sharp-conditioned-{mode}.ply')
export_splat(state,OUT/f'sharp-conditioned-{mode}.splat')
depth=captured['depth'];original=baseline['monodepth']
H,W=data['depth'][0].shape
small=F.interpolate(depth[:,:1],size=(H,W),mode='bilinear',align_corners=True)[0,0].numpy()
ys,xs=np.mgrid[:H,:W]
canopy=support&(ys>H*.15)&(ys<H*.65);base=support&(ys>H*.7)
report={'mode':mode,'device':'mps','seconds':time.perf_counter()-start,'gaussians':state['mean_vectors'].shape[1],
 'post_prediction_geometric_edits':False,'forward_call_counts':calls,
 'scale_anchor_pixels':anchor_count,'two_view_to_sharp_scale':scale,
 'prior':'Full-frame DA3 joint two-view depth, scaled by pedestal anchor; no post-warp field or canopy fit',
 'initializer_depth_mean_abs_log_change':float((depth/original).log().abs().mean()),
 'initializer_depth_canopy_median':float(np.median(small[canopy])),
 'initializer_depth_pedestal_median':float(np.median(small[base])),
 'output_changes':{k:float((state[k]-baseline[k]).abs().mean()) for k in gaussians._fields},
 'limitation':'Released SHARP remains a monocular appearance/feature model. Conditioning is inference-time and untrained on this DA3 prior.'}
(OUT/f'conditioned-{mode}-report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
