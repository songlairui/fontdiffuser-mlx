import numpy as np, mlx.core as mx
from mlx_fd.encoders import ContentEncoder, StyleEncoder
from mlx_fd.unet import UNet
from mlx_fd.model import FontDiffuserModel, FontDiffuserModelDPM
from mlx_fd.scheduler import DDPMScheduler, DPMSolverPipeline
from PIL import Image
mx.random.seed(42)

# load models
content_enc = ContentEncoder(64, 96)
style_enc = StyleEncoder(64, 96)
unet = UNet(sample_size=96, in_channels=3, out_channels=3, block_out_channels=(64,128,256,512), layers_per_block=2, cross_attention_dim=1024, attention_head_dim=1, content_encoder_downsample_size=3, content_start_channel=64)
content_enc.load_weights({k: mx.array(v) for k, v in np.load('mlx_weights/content_encoder.npz').items()})
style_enc.load_weights({k: mx.array(v) for k, v in np.load('mlx_weights/style_encoder.npz').items()}, strict=False)
unet.load_weights({k: mx.array(v) for k, v in np.load('mlx_weights/unet.npz').items()})

# load inputs
content_img = mx.array(np.transpose(np.load('_inbox/testdata/content_img_nchw.npy'), (0,2,3,1)).astype(np.float32))
style_img = mx.array(np.transpose(np.load('_inbox/testdata/style_img_nchw.npy'), (0,2,3,1)).astype(np.float32))
x_t = mx.array(np.transpose(np.load('_inbox/testdata/x_t_nchw.npy'), (0,2,3,1)).astype(np.float32))

sf, _, _ = style_enc(style_img)
cf, cres = content_enc(content_img)
scf, scre = content_enc(style_img)
enc = [sf, list(cres)+[cf], sf.reshape(sf.shape[0], sf.shape[1]*sf.shape[2], sf.shape[3]), list(scre)+[scf]]

noise_pred, _ = unet(x_t, mx.array([999]), encoder_hidden_states=enc, content_encoder_downsample_size=3)
mx.eval(noise_pred)
np.save('_inbox/testdata/noise_pred_t999_nchw.npy', np.array(noise_pred).transpose(0,3,1,2))
print('saved noise_pred_t999_nchw.npy')

model = FontDiffuserModel(unet=unet, style_encoder=style_enc, content_encoder=content_enc)
model_dpm = FontDiffuserModelDPM(unet=unet, style_encoder=style_enc, content_encoder=content_enc)
scheduler = DDPMScheduler(1000, 0.0001, 0.02, 'scaled_linear')
pipeline = DPMSolverPipeline(model=model_dpm, ddpm_train_scheduler=scheduler, guidance_type='classifier-free', guidance_scale=1.0)
result = pipeline.generate(content_images=content_img, style_images=style_img, batch_size=1, num_inference_step=20, content_encoder_downsample_size=3, dm_size=(96,96), initial_noise=x_t)
mx.eval(result)
np.save('_inbox/testdata/dpm_output_nchw.npy', np.array(result).transpose(0,3,1,2))
arr = np.array(result[0])
arr_uint8 = ((arr + 1.0) * 127.5).clip(0,255).astype(np.uint8)
Image.fromarray(arr_uint8).convert('L').save('_inbox/testdata/expected_output.png')
print('saved dpm_output and expected_output.png')
