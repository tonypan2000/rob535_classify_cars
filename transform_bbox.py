from glob import glob
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


def rot(n):
  n = np.asarray(n).flatten()
  assert(n.size == 3)
  theta = np.linalg.norm(n)
  if theta:
    n /= theta
    K = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    return np.identity(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K
  else:
    return np.identity(3)


def get_bbox(p0, p1):
  v = np.array([
    [p0[0], p0[0], p0[0], p0[0], p1[0], p1[0], p1[0], p1[0]],
    [p0[1], p0[1], p1[1], p1[1], p0[1], p0[1], p1[1], p1[1]],
    [p0[2], p1[2], p0[2], p1[2], p0[2], p1[2], p0[2], p1[2]]
  ])
  e = np.array([
    [2, 3, 0, 0, 3, 3, 0, 1, 2, 3, 4, 4, 7, 7],
    [7, 6, 1, 2, 1, 2, 4, 5, 6, 7, 5, 6, 5, 6]
  ], dtype=np.uint8)
  return v, e


files = glob('trainval/*/*_image.jpg')
idx = np.random.randint(0, len(files))
snapshot = files[idx]
print(snapshot)

img = plt.imread(snapshot)

proj = np.fromfile(snapshot.replace('_image.jpg', '_proj.bin'), dtype=np.float32)
proj.resize([3, 4])

try:
  bbox = np.fromfile(snapshot.replace('_image.jpg', '_bbox.bin'), dtype=np.float32)
except FileNotFoundError:
  print('[*] bbox not found.')
  bbox = np.array([], dtype=np.float32)
bbox = bbox.reshape([-1, 11])

fig1 = plt.figure(1, figsize=(16, 9))
ax1 = fig1.add_subplot(1, 1, 1)
ax1.imshow(img)
ax1.axis('scaled')
fig1.tight_layout()

colors = ['C{:d}'.format(i) for i in range(10)]
for k, b in enumerate(bbox):
  R = rot(b[0:3])
  t = b[3:6]
  sz = b[6:9]
  vert_3D, edges = get_bbox(-sz / 2, sz / 2)
  vert_3D = R @ vert_3D + t[:, np.newaxis]
  vert_2D = proj @ np.vstack([vert_3D, np.ones(vert_3D.shape[1])])
  vert_2D = vert_2D / vert_2D[2, :]

  clr = colors[np.mod(k, len(colors))]
  for e in edges.T:
    ax1.plot(vert_2D[0, e], vert_2D[1, e], color=clr)

  x1 = np.min(vert_2D[0])
  y1 = np.min(vert_2D[1])
  x2 = np.max(vert_2D[0])
  y2 = np.max(vert_2D[1])
  ax1.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=None))

plt.show()
