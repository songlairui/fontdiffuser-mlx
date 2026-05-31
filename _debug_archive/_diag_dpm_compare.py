import numpy as np, mlx.core as mx, torch
from diffusers import DPMSolverMultistepScheduler
from mlx_fd.encoders import ContentEncoder, StyleEncoder
from mlx_fd.unet import UNet
from mlx_fd.model import FontDiffuserModelDPM
from mlx_fd.scheduler import DDPMScheduler, DPMSolverPipeline

# load mlx models
content_enc = ContentEncoder(64, 96)
style_enc = StyleEncoder(64, 96)
unet = UNet(sample_size=96, in_channels=3, out_channels=3, block_out_channels=(64,128,256,512), layers_per_block=2, cross_attention_dim=1024, attention_head_dim=1, content_encoder_downsample_size=3, content_start_channel=64)
content_enc.load_weights({k: mx.array(v) for k, v in np.load('mlx_weights/content_encoder.npz').items()})
style_enc.load_weights({k: mx.array(v) for k, v in np.load('mlx_weights/style_encoder.npz').items()}, strict=False)
unet.load_weights({k: mx.array(v) for k, v in np.load('mlx_weights/unet.npz').items()})

content_img = mx.array(np.transpose(np.load('_inbox/testdata/content_img_nchw.npy'), (0,2,3,1)).astype(np.float32))
style_img = mx.array(np.transpose(np.load('_inbox/testdata/style_img_nchw.npy'), (0,2,3,1)).astype(np.float32))
x_t = mx.array(np.transpose(np.load('_inbox/testdata/x_t_nchw.npy'), (0,2,3,1)).astype(np.float32))

sf, _, _ = style_enc(style_img)
cf, cres = content_enc(content_img)
scf, scre = content_enc(style_img)
enc = [sf, list(cres)+[cf], sf.reshape(sf.shape[0], sf.shape[1]*sf.shape[2], sf.shape[3]), list(scre)+[scf]]
model = FontDiffuserModelDPM(unet=unet, style_encoder=style_enc, content_encoder=content_enc)

# capture noise predictions at each timestep
timesteps = np.linspace(999, 0, 21, dtype=np.int32)
cond = (content_img, style_img)
preds = []
x = x_t
for t in timesteps[:-1]:
    noise_pred = model(x, mx.array([int(t)]), cond, 3)
    preds.append(np.array(noise_pred))
    mx.eval(noise_pred)
print('captured preds', len(preds), preds[0].mean(), preds[0].std())

# setup upstream dpm solver
dpm = DPMSolverMultistepScheduler.from_config({
    'num_train_timesteps': 1000,
    'beta_start': 0.0001,
    'beta_end': 0.02,
    'beta_schedule': 'scaled_linear',
    'solver_order': 2,
    'prediction_type': 'epsilon',
    'thresholding': False,
    'dynamic_thresholding_ratio': 0.995,
    'sample_max_value': 1.0,
    'algorithm_type': 'dpmsolver++',
    'solver_type': 'midpoint',
    'lower_order_final': True,
    'use_karras_sigmas': False,
    'lambda_min_clipped': -float('inf'),
    'variance_type': None,
})
dpm.set_timesteps(20)
# convert to numpy timesteps
x_pt = np.array(x_t).transpose(0,3,1,2)  # NCHW
model_outputs = []
for i,p in enumerate(preds):
    model_outputs.append(np.array(p).transpose(0,3,1,2))  # NCHW
# step through
sample = x_pt.copy()
for i,t in enumerate(dpm.timesteps):
    model_output = model_outputs[i]
    sample = dpm.step(torch.from_numpy(model_output), torch.tensor(int(t)), torch.from_numpy(sample), return_dict=False)[0].numpy()
    print('upstream step',i,'t',int(t),'mean',sample.mean(),'std',sample.std())
print('upstream final mean/std', sample.mean(), sample.std())
