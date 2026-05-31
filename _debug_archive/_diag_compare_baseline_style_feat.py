import sys
import importlib.util
import torch
import numpy as np

spec = importlib.util.spec_from_file_location('style_encoder', '/Users/larysong/repo/projects/fontdiffuser/src/modules/style_encoder.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
PTStyleEncoder = mod.StyleEncoder

ckpt = torch.load('/Users/larysong/repo/projects/personal-handwriting/fontdiffuser/ckpt/ckpt/style_encoder.pth', map_location='cpu', weights_only=True)
# normalize with u0 like our converter
for k,v in list(ckpt.items()):
    if k.endswith('.u0'):
        wkey = k[:-len('.u0')] + '.weight'
        if wkey in ckpt:
            W = ckpt[wkey].double().view(ckpt[wkey].shape[0], -1)
            u = v.double().squeeze()
            u = u / (u.norm()+1e-12)
            v_vec = W.T @ u
            v_vec = v_vec / (v_vec.norm()+1e-12)
            sigma = float((u @ W @ v_vec).item())
            ckpt[wkey] = ckpt[wkey] / sigma

model = PTStyleEncoder(G_ch=64, resolution=96)
model.load_state_dict(ckpt, strict=False)
model.eval()

x = torch.from_numpy(np.load('_inbox/testdata/style_img_nchw.npy'))
with torch.no_grad():
    out = model(x)
if isinstance(out, tuple):
    out = out[0]
print('PT recompute style_feat mean/std', out.mean().item(), out.std().item())
base = np.load('_inbox/testdata/style_feat_nchw.npy')
print('baseline style_feat mean/std', base.mean(), base.std())
print('diff max', np.abs(out.cpu().numpy()-base).max())
np.save('_inbox/testdata/style_feat_nchw.npy', out.cpu().numpy())
print('updated baseline style_feat saved')
