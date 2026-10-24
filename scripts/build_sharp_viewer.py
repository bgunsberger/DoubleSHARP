from pathlib import Path
import json
root=Path(__file__).resolve().parents[1]
s=(root/'vendor/free-splatter.cpp/web/index.html').read_text()
s=s.replace('<title>free-splatter.cpp — gaussian splatting demo</title>','<title>DoubleSHARP viewer</title>')
s=s.replace('<h1>free-splatter.cpp</h1>','<h1 id="view-title">SHARP</h1>')
s=s.replace('3D gaussian scene from 2 photos — CPU/Vulkan engine, WebGL render.','Local SHARP · full resolution Gaussians')
s=s.replace('</style>', '''#panel {top:8px;left:8px;width:auto;padding:5px 9px;border-radius:6px;}
#panel>* {display:none;} #panel #status {display:block;margin:0;font-size:10px;min-height:0;}
#hint {display:none;}
</style>''')
s=s.replace('let framed = false;','let framed = true;')
s=s.replace('??"0")];','??"-2")];') if False else s
s=s.replace('parseFloat(_p.get("tz")??"0")','parseFloat(_p.get("tz")??"-2")')
s=s.replace('?? "0.05"','?? "0.0"')
s=s.replace('  const fovy=1.0;','  const imageAspect = 8368 / 5584;\n  const fovy = 2*Math.atan(canvas.height / (2 * Math.min(canvas.width/8368, canvas.height/5584) * 6510.36));')
s=s.replace('perspective(fovy, aspect, 0.05, 100)','perspective(fovy, aspect, 0.05, 100000)')
s=s.replace('const keys = {};','''const keys = {};
$("view-title").textContent = _p.get("label") || "SHARP";
addEventListener("message", e => {
  if(e.origin!==location.origin || e.data.type!=="sharp-camera")return;
  yaw=e.data.yaw; pitch=e.data.pitch; target=e.data.target; dist=e.data.dist;sortDirty=true;
});''')
s=s.replace('  if(sortDirty){ requestSort(view); sortDirty=false; }','''  if(sortDirty){
    if(parent!==window) parent.postMessage({type:"sharp-camera-state",yaw,pitch,target,dist},location.origin);
    requestSort(view); sortDirty=false;
  }''')
meta=json.loads((root/'output/sharp-multiview/baseline-meta.json').read_text())
s=s.replace('8368',str(meta['width'])).replace('5584',str(meta['height'])).replace('6510.36',str(meta['focal_px']))
from PIL import Image
sources=json.loads((root/'output/sharp-multiview/depth-meta.json').read_text())['sources']
for i,path in enumerate(sources):
    image=Image.open(path).convert('RGB');image.thumbnail((1200,1200))
    image.save(root/f'output/sharp-multiview/input-{i}.jpg',quality=90)
# Parent drives shared controls; individual iframe gestures stay local.
(root/'web/sharp-viewer.html').write_text('<!-- Adapted from localai-org/free-splatter.cpp (Apache-2.0). See NOTICE. -->\n'+s)
