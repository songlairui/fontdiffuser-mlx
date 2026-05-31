import sys, types, importlib.util, torch, numpy as np

# setup upstream modules without triggering __init__ imports
pkg = types.ModuleType('modules')
pkg.__path__ = ['/Users/larysong/repo/projects/fontdiffuser/src/modules']
sys.modules['modules'] = pkg
for mod_name, path in [
    ('modules.content_encoder', '/Users/larysong/repo/projects/fontdiffuser/src/modules/content_encoder.py'),
    ('modules.style_encoder', '/Users/larysong/repo/projects/fontdiffuser/src/modules/style_encoder.py'),
]:
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)

from modules.style_encoder import StyleEncoder

model = StyleEncoder(G_ch=64, resolution=96)
state = torch.load('/Users/larysong/repo/projects/personal-handwriting/fontdiffuser/ckpt/ckpt/style_encoder.pth', map_location='cpu', weights_only=True)
model.load_state_dict(state, strict=False)
model.eval()
x = torch.from_numpy(np.load('_inbox/testdata/style_img_nchw.npy'))

# forward with hooks to capture block outputs
outs = {}

def make_hook(name):
    def hook(module, inp, out):
        outs[name] = out.detach() if isinstance(out, torch.Tensor) else out[0].detach()
    return hook

for i, block in enumerate(model.blocks):
    block.register_forward_hook(make_hook(f'block{i}'))

# Reproduce upstream forward which calls last block twice
with torch.no_grad():
    h = x
    residual_features = [h]
    for index, blocklist in enumerate(model.blocks):
        for block in blocklist:
            h = block(h)
        if index in model.save_featrues[:-1]:
            residual_features.append(h)
    h = model.blocks[-1](h)
    style_feat = h

outs['style_feat'] = style_feat.detach()

for name in ['block0','block1','block2','block3','block4','block5','style_feat']:
    if name not in outs:
        continue
    t = outs[name]
    print(name, t.shape, t.mean().item(), t.std().item())
