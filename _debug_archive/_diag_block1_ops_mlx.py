import numpy as np, mlx.core as mx, mlx.nn as nn
from mlx_fd.encoders import StyleEncoder

enc = StyleEncoder(64, 96)
enc.load_weights({k: mx.array(v) for k, v in np.load('mlx_weights/style_encoder.npz').items()}, strict=False)
x = mx.array(np.transpose(np.load('_inbox/testdata/style_img_nchw.npy'), (0,2,3,1)).astype(np.float32))

# run block0
h = enc.blocks[0](x)
b1 = enc.blocks[1]

# helper

def stats(t):
    a = np.array(t)
    return a.mean(), a.std(), np.linalg.norm(a)

pre = h
print('pre', stats(pre))
c1 = b1.conv1(pre)
print('conv1', stats(c1))
r1 = nn.relu(c1)
print('relu', stats(r1))
c2 = b1.conv2(r1)
print('conv2', stats(c2))
p2 = nn.AvgPool2d(2)(c2)
print('pool', stats(p2))
# shortcut path
sc = b1.conv_sc(nn.AvgPool2d(2)(pre))
print('shortcut', stats(sc))
out = p2 + sc
print('block1', stats(out))
