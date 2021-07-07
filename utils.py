"""
# visualization code copied from: https://github.com/fregu856/deeplabv3/blob/master/visualization/run_on_seq.py
"""
import time

import numpy as np
import PIL
import torch

label_names = [
    "__background__",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

label_to_color = {
    0: [128, 64, 128],
    1: [244, 35, 232],
    2: [70, 70, 70],
    3: [102, 102, 156],
    4: [190, 153, 153],
    5: [153, 153, 153],
    6: [250, 170, 30],
    7: [220, 220, 0],
    8: [107, 142, 35],
    9: [152, 251, 152],
    10: [70, 130, 180],
    11: [220, 20, 60],
    12: [255, 0, 0],
    13: [0, 0, 142],
    14: [0, 0, 70],
    15: [0, 60, 100],
    16: [0, 80, 100],
    17: [0, 0, 230],
    18: [119, 11, 32],
    19: [81, 0, 81],
}


def get_class_labels(outputs):
    pred_label_imgs = np.argmax(outputs, axis=1)  # (shape: (batch_size, img_h, img_w))
    return pred_label_imgs.astype(np.uint8)


def convert_label_image_to_color(img):
    img_height, img_width = img.shape
    img_color = np.zeros((img_height, img_width, 3))
    for row in range(img_height):
        for col in range(img_width):
            label = img[row, col]
            img_color[row, col] = np.array(label_to_color[label])

    return img_color


def post_process_image(image_tensor):
    img = image_tensor  # imgs[i] # (shape: (3, img_h, img_w))

    # img = img.data.cpu().numpy()
    img = np.transpose(img, (1, 2, 0))  # (shape: (img_h, img_w, 3))
    img = img * np.array([0.229, 0.224, 0.225])
    img = img + np.array([0.485, 0.456, 0.406])
    img = img * 255.0
    img = img.astype(np.uint8)
    return img


def overlay_image(pred_label_img, img):
    pred_label_img_color = convert_label_image_to_color(pred_label_img)
    overlayed_img = 0.35 * img + 0.65 * pred_label_img_color
    return overlayed_img.astype(np.uint8)


def run_inference(model, input_tensor):
    t = time.time()
    # input_tensor_cuda = input_tensor.cuda()
    output = model(input_tensor)
    torch.cuda.synchronize()
    output = output.cpu()
    return (time.time() - t), output
