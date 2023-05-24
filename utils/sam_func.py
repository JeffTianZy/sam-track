import os
import cv2
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from PIL import Image, ImageFilter
from segment_anything import sam_model_registry, SamPredictor

sys.path.append("../..")


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def object_track(img_path, tracker=None):
    tracks = {}
    if tracker is None:
        tracker = cv2.legacy.TrackerCSRT_create()
    img_list = sorted(os.listdir(img_path))
    roi = None
    for img_file in tqdm(img_list):
        img_rpath = os.path.join(img_path, img_file)
        org_image = cv2.imread(img_rpath)
        image = cv2.cvtColor(org_image, cv2.COLOR_BGR2RGB)
        if roi is None:
            roi = cv2.selectROI("Frame", image, fromCenter=False, showCrosshair=True)
            tracker.init(image, roi)
        else:
            (_, roi) = tracker.update(image)
        (x, y, w, h) = [int(v) for v in roi]
        tracks[img_file] = [(2 * x + w) / 2, (2 * y + h) / 2]

    return tracks


def auto_seg(img_path, tracks, checkpoint='./ckpts/sam_vit_h_4b8939.pth', device='cuda', model_type='default'):
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    img_list = sorted(os.listdir(img_path))
    for img_file in tqdm(img_list):
        img_rpath = os.path.join(img_path, img_file)
        org_image = cv2.imread(img_rpath)
        image = cv2.cvtColor(org_image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image)

        axis = tracks[img_file]
        input_point = np.array([axis])
        input_label = np.array([1])
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        mask_img = masks[2] * 255
        plt.imsave(img_rpath.replace('images', 'masks'), mask_img, cmap='gray')


def post_process(mask_dir, out_dir):
    mask_list = os.listdir(mask_dir)
    mask_list = sorted(mask_list)
    for mask in mask_list:
        print(f'Processing on {mask}')
        msk = Image.open(os.path.join(mask_dir, mask))
        msk = msk.filter(ImageFilter.ModeFilter(size=13))
        msk = np.array(msk)[:, :, 0]
        kernel = np.ones((5, 5), dtype=np.uint8)
        msk = cv2.erode(msk, kernel, iterations=3)
        msk = cv2.dilate(msk, kernel, iterations=3)
        plt.imsave(os.path.join(out_dir, mask), msk, cmap='gray')
