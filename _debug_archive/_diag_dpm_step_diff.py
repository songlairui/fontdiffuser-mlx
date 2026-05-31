import sys, types, importlib.util, torch, numpy as np, mlx.core as mx
from mlx_fd.encoders import ContentEncoder, StyleEncoder
from mlx_fd.unet import UNet
from mlx_fd.model import FontDiffuserModelDPM
from mlx_fd.scheduler import DDPMScheduler, DPMSolverPipeline

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
from modules.style_encoder import StyleEncoder as PTStyleEncoder
from modules.content_encoder import ContentEncoder as PTContentEncoder

# load PT models
pt_ce = PTContentEncoder(G_ch=64, resolution=96)
pt_se = PTStyleEncoder(G_ch=64, resolution=96)
pt_ce.load_state_dict(torch.load('/Users/larysong/repo/projects/personal-handwriting/fontdiffuser/ckpt/ckpt/content_encoder.pth', map_location='cpu', weights_only=True), strict=False)
pt_se.load_state_dict(torch.load('/Users/larysong/repo/projects/personal-handwriting/fontdiffuser/ckpt/ckpt/style_encoder.pth', map_location='cpu', weights_only=True), strict=False)
pt_ce.eval(); pt_se.eval()

# load inputs
content_img = torch.from_numpy(np.load('_inbox/testdata/content_img_nchw.npy'))
style_img = torch.from_numpy(np.load('_inbox/testdata/style_img_nchw.npy'))
x_t = torch.from_numpy(np.load('_inbox/testdata/x_t_nchw.npy')).float()

with torch.no_grad():
    pt_sf, _, _ = pt_se(style_img)
    pt_cf, pt_cres = pt_ce(content_img)
    pt_scf, pt_scre = pt_ce(style_img)

# build encoder_hidden_states same as our pipeline (simplified)
# We'll just use our MLX model for DPM step comparison
# Load MLX models
content_enc = ContentEncoder(64, 96)
style_enc = StyleEncoder(64, 96)
unet = UNet(sample_size=96, in_channels=3, out_channels=3, block_out_channels=(64,128,256,512), layers_per_block=2, cross_attention_dim=1024, attention_head_dim=1, content_encoder_downsample_size=3, content_start_channel=64)
content_enc.load_weights({k: mx.array(v) for k, v in np.load('mlx_weights/content_encoder.npz').items()})
style_enc.load_weights({k: mx.array(v) for k, v in np.load('mlx_weights/style_encoder.npz').items()}, strict=False)
unet.load_weights({k: mx.array(v) for k, v in np.load('mlx_weights/unet.npz').items()})

content_img_mlx = mx.array(content_img.numpy().transpose(0,2,3,1))
style_img_mlx = mx.array(style_img.numpy().transpose(0,2,3,1))
x = mx.array(x_t.numpy().transpose(0,2,3,1))

sf, _, _ = style_enc(style_img_mlx)
cf, cres = content_enc(content_img_mlx)
scf, scre = content_enc(style_img_mlx)
enc = [sf, list(cres)+[cf], sf.reshape(sf.shape[0], sf.shape[1]*sf.shape[2], sf.shape[3]), list(scre)+[scf]]
model = FontDiffuserModelDPM(unet=unet, style_encoder=style_enc, content_encoder=content_enc)
scheduler = DDPMScheduler(1000, 0.0001, 0.02, 'scaled_linear')
pipeline = DPMSolverPipeline(model=model, ddpm_train_scheduler=scheduler, guidance_type='classifier-free', guidance_scale=1.0)

# Run DPM loop manually to capture intermediates
timesteps = np.linspace(999, 0, 21, dtype=np.int32)
cond = (content_img_mlx, style_img_mlx)
print('step,t,t_next,x_mean,x_std')
for i in range(len(timesteps)-1):
    t = int(timesteps[i]); t_next = int(timesteps[i+1])
    noise_pred = model(x, mx.array([t]), cond, 3)
    alpha_t = scheduler.alphas_cumprod[t]
    alpha_next = scheduler.alphas_cumprod[t_next]
    x0 = (x - mx.sqrt(1 - alpha_t) * noise_pred) / mx.sqrt(alpha_t)
    x0 = mx.clip(x0, -1, 1)
    x = mx.sqrt(alpha_next) * x0 + mx.sqrt(1 - alpha_next) * noise_pred
    mx.eval(x)
    print(f'{i},{t},{t_next},{float(np.array(x).mean()):.6f},{float(np.array(x).std()):.6f}')
print('final fg', float((np.array(x[0]) > 0).mean()))
