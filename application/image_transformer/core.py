import cv2
import torch
import numpy as np

from PIL import Image, ImageOps
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

from application.image_transformer.data_loader import RescaleT, ToTensorLab, SalObjDataset
from application.image_transformer.u2net_model import U2NET


coco_labels = {
    'car': 2,
    'airplane': 4,
    'boat': 8,
    'cake': 60,
    'road': 148
}


def make_thumbnail(image_arr, size):
    image = Image.fromarray(image_arr)
    image.thumbnail(size, Image.ANTIALIAS)
    return np.array(image)


def generate_trimap(image_arr, models_path=None):
    saliency_map = get_saliency_map(image_arr, models_path)
    saliency_map[saliency_map > 0] = 255
    trimap = transform_saliency_to_trimap(saliency_map, 7, 3)
    return trimap


def trimap_to_segmentation(trimap, label):
    segmentation = np.where(trimap > 0, coco_labels[label], coco_labels['road'])
    return segmentation


def make_segmentation_square(segmentation, size):
    height, width = segmentation.shape
    width_diff = (size[0] - width)
    left = int(width_diff / 2)
    right = width_diff - left
    height_diff = (size[1] - height)
    top = int(height_diff / 2)
    bottom = height_diff - top
    image = Image.fromarray(segmentation.astype(np.uint8)).convert('L')
    img_with_border = ImageOps.expand(image, border=(left, top, right, bottom), fill=coco_labels['road'])
    return np.array(img_with_border)


def transform_saliency_to_trimap(mask, dilation_size, erosion_size):
    dilation_pixels = 2 * dilation_size + 1
    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_pixels, dilation_pixels))
    dilated = cv2.dilate(mask, dilation_kernel, iterations=1)

    erosion_pixels = 2 * erosion_size + 1
    erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_pixels, erosion_pixels))
    eroded = cv2.erode(mask, erosion_kernel, iterations=1)

    remake = np.zeros_like(mask)
    remake[dilated == 255] = 127
    remake[eroded == 255] = 255

    return remake


def norm_prediction(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d-mi)/(ma-mi)
    return dn


def get_saliency_map(image_arr, models_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path = 'models' if models_path is None else models_path

    model = U2NET(3, 1).to(device)
    checkpoint = torch.load(f'{path}/u2net.pth')
    model.load_state_dict(checkpoint)

    test_salobj_dataset = SalObjDataset(img_name_list=[image_arr],
                                        lbl_name_list=[],
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)]))
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)

    model.eval()
    with torch.no_grad():
        for data_test in test_salobj_dataloader:
            inputs_test = data_test['image'].type(torch.FloatTensor)
            inputs_test = Variable(inputs_test.to(device))

            d1, d2, d3, d4, d5, d6, d7 = model(inputs_test)

            pred = d1[:, 0, :, :]
            pred = norm_prediction(pred)

            del d1, d2, d3, d4, d5, d6, d7

            predict_np = pred.squeeze().cpu().data.numpy()
            saliency_map = Image.fromarray(predict_np * 255).convert('L')
            saliency_map = saliency_map.resize((image_arr.shape[1], image_arr.shape[0]), resample=Image.BILINEAR)

            return np.array(saliency_map)
