import torch, numpy as np
from pathlib import Path

CKPT = Path('/Users/larysong/repo/projects/personal-handwriting/fontdiffuser/ckpt/ckpt/style_encoder.pth')

def power_iteration_sigma(W, u, num_itrs=1, eps=1e-12):
    # W: [out, in], u: [out]
    u = u / (u.norm() + eps)
    v = W.T @ u
    v = v / (v.norm() + eps)
    for _ in range(num_itrs):
        u = W @ v
        u = u / (u.norm() + eps)
        v = W.T @ u
        v = v / (v.norm() + eps)
    return (u @ (W @ v))

def get_weight(state, prefix):
    W = state[f'{prefix}.weight']
    u = state[f'{prefix}.u0'].squeeze()
    Wm = W.view(W.shape[0], -1).double()
    sigma = power_iteration_sigma(Wm, u.double())
    return (W / sigma).float()

def forward_block(state, prefix, x, downsample=True):
    conv1W = get_weight(state, f'{prefix}.conv1')
    conv2W = get_weight(state, f'{prefix}.conv2')
    b1 = state[f'{prefix}.conv1.bias']
    b2 = state[f'{prefix}.conv2.bias']
    shortcut = x
    h = torch.nn.functional.conv2d(x, conv1W, b1, padding=1)
    h = torch.relu(h)
    h = torch.nn.functional.conv2d(h, conv2W, b2, padding=1)
    if downsample:
        h = torch.nn.functional.avg_pool2d(h, 2)
    # shortcut
    if f'{prefix}.conv_sc.weight' in state:
        scW = get_weight(state, f'{prefix}.conv_sc')
        scb = state[f'{prefix}.conv_sc.bias']
        shortcut = torch.nn.functional.conv2d(shortcut, scW, scb)
    if downsample:
        shortcut = torch.nn.functional.avg_pool2d(shortcut, 2)
    return h + shortcut

def main():
    state = torch.load(str(CKPT), map_location='cpu', weights_only=True)
    # remap keys blocks.{i}.0.x -> blocks.{i}.x
    mapped = {}
    for k,v in state.items():
        parts = k.split('.')
        if len(parts)>=4 and parts[0]=='blocks' and parts[2]=='0':
            k2 = f"blocks.{parts[1]}.{'.'.join(parts[3:])}"
            mapped[k2]=v
        else:
            mapped[k]=v
    state = mapped
    x = torch.from_numpy(np.load('_inbox/testdata/style_img_nchw.npy'))
    h = x
    print('input', h.mean().item(), h.std().item())
    for i in range(5):
        h = forward_block(state, f'blocks.{i}', h, downsample=True)
        print(f'block{i}', h.shape, h.mean().item(), h.std().item())
    # last norm + relu + conv
    mean = h.mean(dim=(2,3), keepdim=True)
    var = h.var(dim=(2,3), keepdim=True, unbiased=False)
    h = (h - mean) / torch.sqrt(var + 1e-5)
    print('last_norm', h.mean().item(), h.std().item())
    h = torch.relu(h)
    print('relu', h.mean().item(), h.std().item())
    convW = state.get('last_conv.weight', state.get('blocks.5.2.weight'))
    convb = state.get('last_conv.bias', state.get('blocks.5.2.bias'))
    h = torch.nn.functional.conv2d(h, convW, convb)
    print('style_feat', h.mean().item(), h.std().item())
    np.save('_inbox/testdata/style_feat_nchw.npy', h.detach().cpu().numpy())
    print('saved baseline style_feat')

if __name__ == '__main__':
    main()
