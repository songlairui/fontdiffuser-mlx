import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx_fd.encoders import StyleEncoder

style_w = {k: mx.array(v) for k, v in np.load('mlx_weights/style_encoder.npz').items()}
enc = StyleEncoder(G_ch=64, resolution=96)
enc.load_weights(style_w, strict=False)

x = mx.array(np.transpose(np.load('_inbox/testdata/style_img_nchw.npy'), (0,2,3,1)).astype(np.float32))
h = x
print('input', np.array(h).mean(), np.array(h).std())
for i, block in enumerate(enc.blocks):
    h = block(h)
    arr = np.array(h)
    print(f'block{i}', arr.shape, arr.mean(), arr.std())
h = enc.last_norm(h)
arr = np.array(h)
print('last_norm', arr.shape, arr.mean(), arr.std())
h = nn.relu(h)
print('relu', np.array(h).mean(), np.array(h).std())
h = enc.last_conv(h)
arr = np.array(h)
print('last_conv', arr.shape, arr.mean(), arr.std())
