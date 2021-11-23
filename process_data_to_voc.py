import cv2
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

to_float = {'dtype': torch.float, 'device': 'cpu'}


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


class GTA5(Dataset):
    def __init__(self, root_dir, split='trainval', transform=None):
        self.transform = transform
        self.files = glob(os.path.join(root_dir, split, '*/*_image.jpg'))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        snapshot = self.files[idx]
        img = Image.open(snapshot)
        width = img.width
        height = img.height
        if self.transform:
            img = self.transform(img)
        try:
            bbox = np.fromfile(snapshot.replace('_image.jpg', '_bbox.bin'), dtype=np.float32)
        except FileNotFoundError:
            print('[*] bbox not found.')
            bbox = np.array([], dtype=np.float32)
        bbox = bbox.reshape([-1, 11])
        proj = np.fromfile(snapshot.replace('_image.jpg', '_proj.bin'), dtype=np.float32)
        proj.resize([3, 4])
        return img, bbox, proj, width, height


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

trainval_data = GTA5('./', 'trainval', transform)
test_data = GTA5('./', 'test', transform)
print('Number of training images {}, number of testing images {}'.format(len(trainval_data), len(test_data)))


def gta5_collate_fn(batch_lst, reshape_size=224):
    batch_size = len(batch_lst)
    img_batch = torch.zeros(batch_size, 3, reshape_size, reshape_size)
    max_num_box = max(len(batch_lst[i][1]) for i in range(batch_size))

    box_batch = torch.Tensor(batch_size, max_num_box, 5).fill_(-1.)
    w_list = []
    h_list = []

    for i in range(batch_size):
        img, ann, proj, width, height = batch_lst[i]
        w_list.append(width)
        h_list.append(height)
        img_batch[i] = img
        for bbox_idx, b in enumerate(ann):
            R = rot(b[0:3])
            t = b[3:6]
            sz = b[6:9]
            vert_3D = np.vstack([(-sz / 2), (sz / 2)]).T
            vert_3D = R @ vert_3D + t[:, np.newaxis]
            vert_2D = proj @ np.vstack([vert_3D, np.ones(2)])
            vert_2D = vert_2D / vert_2D[2, :]
            box_batch[i][bbox_idx] = torch.Tensor([vert_2D[0, 0], vert_2D[1, 0],
                                                   vert_2D[0, 1], vert_2D[1, 1], b[9]])

    h_batch = torch.tensor(h_list)
    w_batch = torch.tensor(w_list)

    return img_batch, box_batch, w_batch, h_batch


train_loader = DataLoader(trainval_data,
                          batch_size=10,
                          shuffle=False, pin_memory=True,
                          num_workers=0,
                          collate_fn=gta5_collate_fn)
test_loader = DataLoader(test_data,
                         batch_size=10,
                         shuffle=False, pin_memory=True,
                         num_workers=0,
                         collate_fn=gta5_collate_fn)

train_loader_iter = iter(train_loader)
img, ann, _, _ = train_loader_iter.next()

class_to_idx = {'Unknown': 0, 'Compacts': 1, 'Sedans': 2, 'SUVs': 3, 'Coupes': 4,
                'Muscle': 5, 'SportsClassics': 6, 'Sports': 7, 'Super': 8, 'Motorcycles': 9, 'OffRoad': 10,
                'Industrial': 11, 'Utility': 12, 'Vans': 13, 'Cycles': 14, 'Boats': 15,
                'Helicopters': 16, 'Planes': 17, 'Service': 18, 'Emergency': 19, 'Military': 20,
                'Commercial': 21, 'Trains': 22
                }
idx_to_class = {i: c for c, i in class_to_idx.items()}


def coord_trans(bbox, w_pixel, h_pixel, w_amap=7, h_amap=7, mode='a2p'):
    assert mode in ('p2a', 'a2p'), 'invalid coordinate transformation mode!'
    assert bbox.shape[-1] >= 4, 'the transformation is applied to the first 4 values of dim -1'

    if bbox.shape[0] == 0:  # corner cases
        return bbox

    resized_bbox = bbox.clone()
    # could still work if the first dim of bbox is not batch size
    # in that case, w_pixel and h_pixel will be scalars
    resized_bbox = resized_bbox.view(bbox.shape[0], -1, bbox.shape[-1])
    invalid_bbox_mask = (resized_bbox == -1)  # indicating invalid bbox

    if mode == 'p2a':
        # pixel to activation
        width_ratio = w_pixel * 1. / w_amap
        height_ratio = h_pixel * 1. / h_amap
        resized_bbox[:, :, [0, 2]] /= width_ratio.view(-1, 1, 1)
        resized_bbox[:, :, [1, 3]] /= height_ratio.view(-1, 1, 1)
    else:
        # activation to pixel
        width_ratio = w_pixel * 1. / w_amap
        height_ratio = h_pixel * 1. / h_amap
        resized_bbox[:, :, [0, 2]] *= width_ratio.view(-1, 1, 1)
        resized_bbox[:, :, [1, 3]] *= height_ratio.view(-1, 1, 1)

    resized_bbox.masked_fill_(invalid_bbox_mask, -1)
    resized_bbox.resize_as_(bbox)
    return resized_bbox


def data_visualizer(img, idx_to_class, bbox=None, pred=None):
    img_copy = np.array(img).astype('uint8')

    if bbox is not None:
        for bbox_idx in range(bbox.shape[0]):
            one_bbox = bbox[bbox_idx][:4]
            cv2.rectangle(img_copy, (int(one_bbox[0]), int(one_bbox[1])), (int(one_bbox[2]),
                                                                           int(one_bbox[3])), (255, 0, 0), 2)
            if bbox.shape[1] > 4:  # if class info provided
                obj_cls = idx_to_class[bbox[bbox_idx][4].item()]
                cv2.putText(img_copy, '%s' % (obj_cls),
                            (int(one_bbox[0]), int(one_bbox[1] + 15)),
                            cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), thickness=1)

    if pred is not None:
        for bbox_idx in range(pred.shape[0]):
            one_bbox = pred[bbox_idx][:4]
            cv2.rectangle(img_copy, (one_bbox[0], one_bbox[1]), (one_bbox[2],
                                                                 one_bbox[3]), (0, 255, 0), 2)

            if pred.shape[1] > 4:  # if class and conf score info provided
                obj_cls = idx_to_class[pred[bbox_idx][4].item()]
                conf_score = pred[bbox_idx][5].item()
                cv2.putText(img_copy, '%s, %.2f' % (obj_cls, conf_score),
                            (one_bbox[0], one_bbox[1] + 15),
                            cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), thickness=1)

    plt.imshow(img_copy)
    plt.axis('off')
    plt.show()


# default examples for visualization
batch_size = 3
sampled_idx = torch.linspace(0, len(trainval_data) - 1, steps=batch_size).long()

# get the size of each image first
h_list = []
w_list = []
img_list = []  # list of images
MAX_NUM_BBOX = 40
box_list = torch.LongTensor(batch_size, MAX_NUM_BBOX, 5).fill_(-1)  # PADDED GT boxes
files = glob(os.path.join('./', 'trainval', '*/*_image.jpg'))

for idx, i in enumerate(sampled_idx):
    # hack to get the original image so we don't have to load from local again...
    _, all_bbox, proj, width, height = trainval_data.__getitem__(i)
    img_list.append(Image.open(files[i]))

    for bbox_idx, b in enumerate(all_bbox):
        R = rot(b[0:3])
        t = b[3:6]
        sz = b[6:9]
        vert_3D = np.vstack([(-sz / 2), (sz / 2)]).T
        vert_3D = R @ vert_3D + t[:, np.newaxis]
        vert_2D = proj @ np.vstack([vert_3D, np.ones(2)])
        vert_2D = vert_2D / vert_2D[2, :]
        box_list[idx][bbox_idx] = torch.Tensor([vert_2D[0, 0], vert_2D[1, 0],
                                                vert_2D[0, 1], vert_2D[1, 1], b[9]])

    # get sizes
    img = np.array(img)
    w_list.append(width)
    h_list.append(height)

w_list = torch.as_tensor(w_list, **to_float)
h_list = torch.as_tensor(h_list, **to_float)
box_list = torch.as_tensor(box_list, **to_float)
resized_box_list = coord_trans(box_list, w_list, h_list, mode='p2a')  # on activation map coordinate system

for i in range(len(img_list)):
    valid_box = sum([1 if j != -1 else 0 for j in box_list[i][:, 0]])
    data_visualizer(img_list[i], idx_to_class, box_list[i][:valid_box])


def GenerateGrid(batch_size, w_amap=7, h_amap=7, dtype=torch.float32, device='cuda'):
    w_range = torch.arange(0, w_amap, dtype=dtype, device=device) + 0.5
    h_range = torch.arange(0, h_amap, dtype=dtype, device=device) + 0.5

    w_grid_idx = w_range.unsqueeze(0).repeat(h_amap, 1)
    h_grid_idx = h_range.unsqueeze(1).repeat(1, w_amap)
    grid = torch.stack([w_grid_idx, h_grid_idx], dim=-1)
    grid = grid.unsqueeze(0).repeat(batch_size, 1, 1, 1)

    return grid


grid_list = GenerateGrid(w_list.shape[0], device='cpu')

center = torch.cat((grid_list, grid_list), dim=-1)
grid_cell = center.clone()
grid_cell[:, :, :, [0, 1]] -= 1. / 2.
grid_cell[:, :, :, [2, 3]] += 1. / 2.
center = coord_trans(center, w_list, h_list)
grid_cell = coord_trans(grid_cell, w_list, h_list)

for img, anc, grid in zip(img_list, center, grid_cell):
    data_visualizer(img, idx_to_class, anc.reshape(-1, 4), grid.reshape(-1, 4))
