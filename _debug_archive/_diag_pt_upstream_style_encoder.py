import sys, importlib.util, torch, numpy as np

# Load content_encoder first because style_encoder imports it from package __init__
for name, path in [
    ('content_encoder', '/Users/larysong/repo/projects/fontdiffuser/src/modules/content_encoder.py'),
    ('style_encoder', '/Users/larysong/repo/projects/fontdiffuser/src/modules/style_encoder.py'),
]:
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f'modules.{name}'] = mod
    spec.loader.exec_module(mod)

from modules.style_encoder import StyleEncoder

model = StyleEncoder(G_ch=64, resolution=96)
state = torch.load('/Users/larysong/repo/projects/personal-handwriting/fontdiffuser/ckpt/ckpt/style_encoder.pth', map_location='cpu', weights_only=True)
missing = model.load_state_dict(state, strict=False)
print('missing', missing.missing_keys[:10], 'unexpected', len(missing.unexpected_keys))
model.eval()
x = torch.from_numpy(np.load('_inbox/testdata/style_img_nchw.npy'))
with torch.no_grad():
    out = model(x)
if isinstance(out, tuple):
    out = out[0]
print('upstream style_feat', out.mean().item(), out.std().item())
np.save('_inbox/testdata/style_feat_nchw.npy', out.cpu().numpy())
print('saved upstream baseline style_feat')
