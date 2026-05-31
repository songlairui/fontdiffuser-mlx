import torch
import torch.nn.functional as F
from pathlib import Path

PT_CKPT = Path('/Users/larysong/repo/projects/personal-handwriting/fontdiffuser/ckpt/ckpt/style_encoder.pth')

def power_iteration(W, u, num_itrs=1, eps=1e-12):
    # W: [out, in]
    v = torch.randn(W.shape[1])
    v = v / (v.norm() + eps)
    for _ in range(num_itrs):
        u_ = W @ v
        u_ = u_ / (u_.norm() + eps)
        v = W.T @ u_
        v = v / (v.norm() + eps)
    sigma = torch.dot(u_, W @ v)
    return sigma, u_, v

def main():
    state = torch.load(str(PT_CKPT), map_location='cpu', weights_only=True)
    keys = sorted([k for k in state if k.endswith('.sv0')])
    for k in keys:
        wkey = k[:-len('.sv0')] + '.weight'
        if wkey not in state:
            continue
        W = state[wkey].float().view(state[wkey].shape[0], -1)
        u = state[k[:-len('.sv0')] + '.u0'].float().squeeze(0)
        # compute with 1 iteration like upstream default
        sigma, u_new, v_new = power_iteration(W, u, num_itrs=1, eps=1e-12)
        stored = state[k].float().item()
        print(f'{k}: stored={stored:.6f}, recomputed={sigma.item():.6f}, diff={abs(stored - sigma.item()):.6f}')

if __name__ == '__main__':
    main()
