from application.matting.transforms import trimap_transform, groupnorm_normalise_image
from application.matting.models import build_model

import cv2
import numpy as np
import torch


def np_to_torch(x):
    return torch.from_numpy(x).permute(2, 0, 1)[None, :, :, :].float().cuda()


def scale_input(x: np.ndarray, scale: float, scale_type) -> np.ndarray:
    """ Scales inputs to multiple of 8. """
    h, w = x.shape[:2]
    h1 = int(np.ceil(scale * h / 8) * 8)
    w1 = int(np.ceil(scale * w / 8) * 8)
    x_scale = cv2.resize(x, (w1, h1), interpolation=scale_type)
    return x_scale


def pred(image_np: np.ndarray, trimap_np: np.ndarray, model) -> np.ndarray:
    """ Predict alpha, foreground and background.
        Parameters:
        image_np -- the image in rgb format between 0 and 1. Dimensions: (h, w, 3)
        trimap_np -- two channel trimap, first background then foreground. Dimensions: (h, w, 2)
        Returns:
        fg: foreground image in rgb format between 0 and 1. Dimensions: (h, w, 3)
        bg: background image in rgb format between 0 and 1. Dimensions: (h, w, 3)
        alpha: alpha matte image between 0 and 1. Dimensions: (h, w)
    """
    h, w = trimap_np.shape[:2]

    image_scale_np = scale_input(image_np, 1.0, cv2.INTER_LANCZOS4)
    trimap_scale_np = scale_input(trimap_np, 1.0, cv2.INTER_LANCZOS4)

    with torch.no_grad():

        image_torch = np_to_torch(image_scale_np)
        trimap_torch = np_to_torch(trimap_scale_np)

        trimap_transformed_torch = np_to_torch(trimap_transform(trimap_scale_np))
        image_transformed_torch = groupnorm_normalise_image(image_torch.clone(), format='nchw')

        output = model(image_torch, trimap_torch, image_transformed_torch, trimap_transformed_torch)

        output = cv2.resize(output[0].cpu().numpy().transpose((1, 2, 0)), (w, h), cv2.INTER_LANCZOS4)
    alpha = output[:, :, 0]
    fg = output[:, :, 1:4]
    bg = output[:, :, 4:7]

    alpha[trimap_np[:, :, 0] == 1] = 0
    alpha[trimap_np[:, :, 1] == 1] = 1
    fg[alpha == 1] = image_np[alpha == 1]
    bg[alpha == 0] = image_np[alpha == 0]
    return fg, bg, alpha


def trimap_for_matting(img_arr):
    trimap_im = img_arr / 255.0
    h, w = trimap_im.shape
    trimap = np.zeros((h, w, 2))
    trimap[trimap_im == 1, 1] = 1
    trimap[trimap_im == 0, 0] = 1
    return trimap


def image_for_matting(img_arr):
    return (img_arr / 255.0)[:, :, ::-1]


def perform_matting(image, trimap, model_path):
    path = 'models' if model_path is None else model_path

    class Args:
        encoder = 'resnet50_GN_WS'
        decoder = 'fba_decoder'
        weights = f'{path}/fba_matting.pth'

    args = Args()
    model = build_model(args)
    model.eval()
    fixed_image = image_for_matting(image)
    fixed_trimap = trimap_for_matting(trimap)
    return pred(fixed_image, fixed_trimap, model)
