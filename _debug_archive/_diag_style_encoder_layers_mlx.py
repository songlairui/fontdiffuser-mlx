import numpy as np, mlx.core as mx
import mlx.nn as nn
from mlx_fd.encoders import StyleEncoder
from mlx_fd.snconv import _power_iteration_sigma

enc = StyleEncoder(64, 96)
enc.load_weights({k: mx.array(v) for k, v in np.load('mlx_weights/style_encoder.npz').items()}, strict=False)

x = mx.array(np.transpose(np.load('_inbox/testdata/style_img_nchw.npy'), (0,2,3,1)).astype(np.float32))
print('input', float(np.array(x).mean()), float(np.array(x).std()))
h = x
for i, block in enumerate(enc.blocks):
    h = block(h)
    a = np.array(h)
    sigma = float(np.array(_power_iteration_sigma(enc.blocks[i].conv1.weight.reshape(enc.blocks[i].conv1.weight.shape[0], -1), enc.blocks[i].conv1.u0)))
    print(f'block{i}', a.mean(), a.std(), 'sigma0', sigma)
mean = mx.mean(h, axis=(1,2), keepdims=True)
var = mx.var(h, axis=(1,2), keepdims=True, ddof=0)
hn = (h - mean) / mx.sqrt(var + 1e-5)
print('last_norm', float(np.array(hn).mean()), float(np.array(hn).std()))
hr = nn.relu(hn)
print('relu', float(np.array(hr).mean()), float(np.array(hr).std()))
hc = enc.last_conv(hr)
print('last_conv', float(np.array(hc).mean()), float(np.array(hc).std()))
