import sys, importlib.util, torch, numpy as np

# Load content_encoder first because style_encoder imports it
ce_spec = importlib.util.spec_from_file_location('modules.content_encoder', '/Users/larysong/repo/projects/fontdiffuser/src/modules/content_encoder.py')
ce_mod = importlib.util.module_from_spec(ce_spec)
sys.modules['modules.content_encoder'] = ce_mod
ce_spec.loader.exec_module(ce_mod)

se_spec = importlib.util.spec_from_file_location('modules.style_encoder', '/Users/larysong/repo/projects/fontdiffuser/src/modules/style_encoder.py')
se_mod = importlib.util.module_from_spec(se_spec)
sys.modules['modules.style_encoder'] = se_mod
se_spec.loader.exec_module(se_mod)

from modules.style_encoder import StyleEncoder

model = StyleEncoder(G_ch=64, resolution=96)
state = torch.load('/Users/larysong/repo/projects/personal-handwriting/fontdiffuser/ckpt/ckpt/style_encoder.pth', map_location='cpu', weights_only=True)

mapped = {}
for k, v in state.items():
    parts = k.split('.')
    if len(parts) >= 4 and parts[0] == 'blocks' and parts[2] == '0':
        k2 = f"blocks.{parts[1]}.{'.'.join(parts[3:])}"
        mapped[k2] = v
    else:
        mapped[k] = v

missing = model.load_state_dict(mapped, strict=False)
print('missing', missing.missing_keys[:10], 'unexpected', len(missing.unexpected_keys))
model.eval()
x = torch.from_numpy(np.load('_inbox/testdata/style_img_nchw.npy'))
with torch.no_grad():
    out = model(x)
if isinstance(out, tuple):
    out = out[0]
print('upstream2 style_feat', out.mean().item(), out.std().item())
np.save('_inbox/testdata/style_feat_nchw.npy', out.cpu().numpy())
print('saved upstream2 baseline style_feat')
