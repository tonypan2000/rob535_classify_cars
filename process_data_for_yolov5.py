import cv2
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image


def rot(n):
    n = np.asarray(n).flatten()
    assert (n.size == 3)
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


def data_visualizer(img, bbox=None):
    img_copy = np.array(img).astype('uint8')
    if bbox is not None:
        cv2.rectangle(img_copy, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
    plt.imshow(img_copy)
    plt.axis('off')
    plt.show()


def extract_bbox(b, width, height):
    class_id = int(b[9])
    R = rot(b[0:3])
    t = b[3:6]
    sz = b[6:9]
    vert_3D, edges = get_bbox(-sz / 2, sz / 2)
    vert_3D = R @ vert_3D + t[:, np.newaxis]
    vert_2D = proj @ np.vstack([vert_3D, np.ones(vert_3D.shape[1])])
    vert_2D = vert_2D / vert_2D[2, :]

    x1 = np.min(vert_2D[0])
    y1 = np.min(vert_2D[1])
    x2 = np.max(vert_2D[0])
    y2 = np.max(vert_2D[1])

    center_x = (x1 + x2) / 2 / width
    center_y = (y1 + y2) / 2 / height
    box_width = (x2 - x1) / width
    box_height = (y2 - y1) / height
    return [class_id, center_x, center_y, box_width, box_height]


class_to_idx = {'Unknown': 0, 'Compacts': 1, 'Sedans': 2, 'SUVs': 3, 'Coupes': 4,
                'Muscle': 5, 'SportsClassics': 6, 'Sports': 7, 'Super': 8, 'Motorcycles': 9, 'OffRoad': 10,
                'Industrial': 11, 'Utility': 12, 'Vans': 13, 'Cycles': 14, 'Boats': 15,
                'Helicopters': 16, 'Planes': 17, 'Service': 18, 'Emergency': 19, 'Military': 20,
                'Commercial': 21, 'Trains': 22
                }
idx_to_class = {i: c for c, i in class_to_idx.items()}


def data_visualizer(img, annotations, width, height):
    img_copy = np.array(img).astype('uint8')
    for bbox_idx in range(bbox.shape[0]):
        one_bbox = annotations[1:]
        x1 = (one_bbox[0] - one_bbox[2] / 2) * width
        y1 = (one_bbox[1] - one_bbox[3] / 2) * height
        x2 = (one_bbox[0] + one_bbox[2] / 2) * width
        y2 = (one_bbox[1] + one_bbox[3] / 2) * height
        cv2.rectangle(img_copy, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        obj_cls = idx_to_class[annotations[0]]
        cv2.putText(img_copy, '%s' % obj_cls,
                    (int(x1), int(y1 + 15)),
                    cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), thickness=1)
    plt.imshow(img_copy)
    plt.axis('off')
    plt.show()


files = glob(os.path.join('./', 'trainval', '*/*_image.jpg'))
for i, img_file in enumerate(files):
    img = Image.open(img_file)
    img_name = os.path.split(os.path.split(img_file)[0])[1] + '_' + os.path.basename(img_file)
    split = 'train'  # if i < 6058 else 'valid'
    img.save(os.path.join('./', split, 'images', img_name))
    width = img.width
    height = img.height
    try:
        bbox = np.fromfile(img_file.replace('_image.jpg', '_bbox.bin'), dtype=np.float32)
    except FileNotFoundError:
        print('[*] bbox not found.')
        bbox = np.array([], dtype=np.float32)
    bbox = bbox.reshape([-1, 11])
    proj = np.fromfile(img_file.replace('_image.jpg', '_proj.bin'), dtype=np.float32)
    proj.resize([3, 4])

    label_name = img_name.replace('.jpg', '.txt')
    with open(os.path.join('./', split, 'labels', label_name), 'w') as f:
        for b in bbox:
            annotations = extract_bbox(b, width, height)
            # data_visualizer(img, annotations, width, height)
            f.write(' '.join(str(ann) for ann in annotations))
            f.write('\n')
