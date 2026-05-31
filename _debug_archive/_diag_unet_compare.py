import sys, importlib.util, torch, numpy as np, mlx.core as mx
# Load upstream UNet directly from file to avoid package init chain
spec = importlib.util.spec_from_file_location('unet', '/Users/larysong/repo/projects/fontdiffuser/src/modules/unet.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
PTUNet = mod.UNet

pt = PTUNet(sample_size=96, in_channels=3, out_channels=3, block_out_channels=(64,128,256,512), layers_per_block=2, cross_attention_dim=1024, attention_head_dim=1, content_encoder_downsample_size=3, content_start_channel=64)
sd = torch.load('/Users/larysong/repo/projects/personal-handwriting/fontdiffuser/ckpt/ckpt/unet.pth', map_location='cpu', weights_only=True)
missing = pt.load_state_dict(sd, strict=False)
print('PT missing', missing.missing_keys[:10], 'unexpected', len(missing.unexpected_keys))
pt.eval()

# load mlx
from mlx_fd.unet import UNet
unet = UNet(sample_size=96, in_channels=3, out_channels=3, block_out_channels=(64,128,256,512), layers_per_block=2, cross_attention_dim=1024, attention_head_dim=1, content_encoder_downsample_size=3, content_start_channel=64)
unet.load_weights({k:mx.array(v) for k,v in np.load('mlx_weights/unet.npz').items()})

# build encoder hidden states from saved npy
from mlx_fd.encoders import ContentEncoder, StyleEncoder
from mlx_fd.model import FontDiffuserModel
ce=ContentEncoder(64,96); se=StyleEncoder(64,96)
ce.load_weights({k:mx.array(v) for k,v in np.load('mlx_weights/content_encoder.npz').items()})
se.load_weights({k:mx.array(v) for k,v in np.load('mlx_weights/style_encoder.npz').items()}, strict=False)
content_img=mx.array(np.transpose(np.load('_inbox/testdata/content_img_nchw.npy'),(0,2,3,1)).astype(np.float32))
style_img=mx.array(np.transpose(np.load('_inbox/testdata/style_img_nchw.npy'),(0,2,3,1)).astype(np.float32))
x_t=mx.array(np.transpose(np.load('_inbox/testdata/x_t_nchw.npy'),(0,2,3,1)).astype(np.float32))
sf,_,_=se(style_img)
cf,cres=ce(content_img)
scf,scre=ce(style_img)
enc=[sf, list(cres)+[cf], sf.reshape(sf.shape[0],sf.shape[1]*sf.shape[2],sf.shape[3]), list(scre)+[scf]]

# mlx forward
mlx_np, _ = unet(x_t, mx.array([999]), encoder_hidden_states=enc, content_encoder_downsample_size=3)
mlx_np = np.array(mlx_np)
if mlx_np.shape[-1] != 3:
    pass
# pt forward
with torch.no_grad():
    pt_sf = torch.from_numpy(np.array(sf).transpose(0,3,1,2))
    pt_cf = torch.from_numpy(np.array(cf).transpose(0,3,1,2))
    pt_cres = [torch.from_numpy(np.array(x).transpose(0,3,1,2)) for x in cres]+[pt_cf]
    pt_x = torch.from_numpy(np.array(x_t).transpose(0,3,1,2))
    pt_scf = torch.from_numpy(np.array(scf).transpose(0,3,1,2))
    pt_scre = [torch.from_numpy(np.array(x).transpose(0,3,1,2)) for x in scre]+[pt_scf]
    pt_hidden = [pt_sf, pt_cres, pt_sf.flatten(2).permute(0,2,1), pt_scre]
    out = pt_x
    # call upstream unet directly
    pt_out = pt(out, torch.tensor([999]), encoder_hidden_states=pt_hidden, content_encoder_downsample_size=3)
    if isinstance(pt_out, tuple):
        pt_out = pt_out[0]
    pt_np = pt_out.cpu().numpy().transpose(0,2,3,1)

print('PT mean/std', pt_np.mean(), pt_np.std())
print('MLX mean/std', mlx_np.mean(), mlx_np.std())
print('diff max', np.abs(pt_np-mlx_np).max())
