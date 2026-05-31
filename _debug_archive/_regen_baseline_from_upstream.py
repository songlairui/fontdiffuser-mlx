import sys, types, importlib.util, torch, numpy as np
from PIL import Image

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
from modules.content_encoder import ContentEncoder

# load models
content_enc = ContentEncoder(G_ch=64, resolution=96)
style_enc = StyleEncoder(G_ch=64, resolution=96)
ce_state = torch.load('/Users/larysong/repo/projects/personal-handwriting/fontdiffuser/ckpt/ckpt/content_encoder.pth', map_location='cpu', weights_only=True)
se_state = torch.load('/Users/larysong/repo/projects/personal-handwriting/fontdiffuser/ckpt/ckpt/style_encoder.pth', map_location='cpu', weights_only=True)
unet_state = torch.load('/Users/larysong/repo/projects/personal-handwriting/fontdiffuser/ckpt/ckpt/unet.pth', map_location='cpu', weights_only=True)
content_enc.load_state_dict(ce_state, strict=False)
style_enc.load_state_dict(se_state, strict=False)

# load inputs
content_img = torch.from_numpy(np.load('_inbox/testdata/content_img_nchw.npy'))
style_img = torch.from_numpy(np.load('_inbox/testdata/style_img_nchw.npy'))
x_t = torch.from_numpy(np.load('_inbox/testdata/x_t_nchw.npy'))

# encode
with torch.no_grad():
    sf, _, _ = style_enc(style_img)
    cf, cres = content_enc(content_img)
    scf, scre = content_enc(style_img)

# UNet forward at t=999
import importlib.util as ilu
unet_spec = ilu.spec_from_file_location('modules.unet', '/Users/larysong/repo/projects/fontdiffuser/src/modules/unet.py')
unet_mod = ilu.module_from_spec(unet_spec)
sys.modules['modules.unet'] = unet_mod
unet_spec.loader.exec_module(unet_mod)
from modules.unet import UNet
unet = UNet(sample_size=96, in_channels=3, out_channels=3, block_out_channels=(64,128,256,512), layers_per_block=2, cross_attention_dim=1024, attention_head_dim=1, content_encoder_downsample_size=3, content_start_channel=64)
unet.load_state_dict(unet_state, strict=False)
unet.eval()
b,h,w,c = sf.shape
style_hidden = sf.reshape(b,h*w,c)
encoder_hidden_states = [sf, list(cres)+[cf], style_hidden, list(scre)+[scf]]
with torch.no_grad():
    noise_pred, _ = unet(x_t, torch.tensor([999]), encoder_hidden_states=encoder_hidden_states, content_encoder_downsample_size=3)
np.save('_inbox/testdata/noise_pred_t999_nchw.npy', noise_pred.cpu().numpy())
print('saved upstream noise_pred_t999_nchw.npy')

# DPM-Solver++ sampling (simplified, guidance_scale=1.0)
# Use same initial noise x_t
x = x_t.clone().float()
scheduler_betas = np.linspace(0.0001**0.5, 0.02**0.5, 1000, dtype=np.float32)**2
alphas = 1.0 - scheduler_betas
alphas_cumprod = np.cumprod(alphas, dtype=np.float64).astype(np.float32)
timesteps = np.linspace(999, 0, 21, dtype=np.int32)
cond = (content_img.float(), style_img.float())
for i in range(len(timesteps)-1):
    t = int(timesteps[i])
    t_next = int(timesteps[i+1])
    with torch.no_grad():
        noise_pred, _ = unet(x, torch.tensor([t]), encoder_hidden_states=encoder_hidden_states, content_encoder_downsample_size=3)
    alpha_t = alphas_cumprod[t]
    alpha_next = alphas_cumprod[t_next]
    x0 = (x - np.sqrt(1-alpha_t)*noise_pred) / np.sqrt(alpha_t)
    x0 = torch.clamp(x0, -1, 1)
    x = np.sqrt(alpha_next)*x0 + np.sqrt(1-alpha_next)*noise_pred
np.save('_inbox/testdata/dpm_output_nchw.npy', x.cpu().numpy())
arr = ((x[0].permute(1,2,0).cpu().numpy()+1)*127.5).clip(0,255).astype(np.uint8)
Image.fromarray(arr).convert('L').save('_inbox/testdata/expected_output.png')
print('saved upstream dpm_output and expected_output.png')
