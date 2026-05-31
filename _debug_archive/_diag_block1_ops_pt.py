import torch, numpy as np

CKPT = '/Users/larysong/repo/projects/personal-handwriting/fontdiffuser/ckpt/ckpt/style_encoder.pth'

def get_weight(state, prefix):
    W = state[f'{prefix}.weight']
    u = state[f'{prefix}.u0'].squeeze()
    Wm = W.view(W.shape[0], -1).double()
    u = u.double(); v = Wm.T @ u; v = v / (v.norm()+1e-12)
    u = Wm @ v; u = u / (u.norm()+1e-12)
    sigma = float((u @ (Wm @ v)).item())
    return (W / sigma).float()

state = torch.load(CKPT, map_location='cpu', weights_only=True)
mapped = {}
for k,v in state.items():
    parts = k.split('.')
    if len(parts)>=4 and parts[0]=='blocks' and parts[2]=='0':
        mapped[f"blocks.{parts[1]}.{'.'.join(parts[3:])}"] = v
    else:
        mapped[k] = v
state = mapped
x = torch.from_numpy(np.load('_inbox/testdata/style_img_nchw.npy'))
# block0
prefix='blocks.0'
conv1W=get_weight(state,f'{prefix}.conv1'); conv1b=state[f'{prefix}.conv1.bias']
conv2W=get_weight(state,f'{prefix}.conv2'); conv2b=state[f'{prefix}.conv2.bias']
scW=get_weight(state,f'{prefix}.conv_sc'); scb=state[f'{prefix}.conv_sc.bias']
h = torch.nn.functional.conv2d(x, conv1W, conv1b, padding=1)
h = torch.relu(h)
h = torch.nn.functional.conv2d(h, conv2W, conv2b, padding=1)
h = torch.nn.functional.avg_pool2d(h,2)
shortcut = torch.nn.functional.conv2d(x, scW, scb)
shortcut = torch.nn.functional.avg_pool2d(shortcut,2)
pre = h + shortcut

# block1 ops
prefix='blocks.1'
conv1W=get_weight(state,f'{prefix}.conv1'); conv1b=state[f'{prefix}.conv1.bias']
conv2W=get_weight(state,f'{prefix}.conv2'); conv2b=state[f'{prefix}.conv2.bias']
scW=get_weight(state,f'{prefix}.conv_sc'); scb=state[f'{prefix}.conv_sc.bias']

def stats(t):
    return t.mean().item(), t.std().item(), t.norm().item()

print('pre', stats(pre))
c1 = torch.nn.functional.conv2d(pre, conv1W, conv1b, padding=1)
print('conv1', stats(c1))
r1 = torch.relu(c1)
print('relu', stats(r1))
c2 = torch.nn.functional.conv2d(r1, conv2W, conv2b, padding=1)
print('conv2', stats(c2))
p2 = torch.nn.functional.avg_pool2d(c2,2)
print('pool', stats(p2))
sc = torch.nn.functional.conv2d(torch.nn.functional.avg_pool2d(pre,2), scW, scb)
print('shortcut', stats(sc))
out = p2 + sc
print('block1', stats(out))
