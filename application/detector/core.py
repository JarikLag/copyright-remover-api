import numpy as np

from detecto import core
from PIL import Image


def get_object_box(image, object_label):
    model = core.Model()
    labels, boxes, _ = model.predict_top(image)
    if object_label in labels:
        idx = labels.index(object_label)
        return boxes[idx]
    else:
        return None


def detect_and_crop(image, label):
    width = image.width
    height = image.height
    image_arr = np.array(image)

    box = get_object_box(image_arr, label)
    if box is not None:
        box = list(map(lambda x: float(x), box))
        x_min, y_min, x_max, y_max = box

        width_diff = (x_max - x_min) * 0.1
        height_diff = (y_max - y_min) * 0.1

        x_min = max(round(x_min - width_diff), 0)
        x_max = min(round(x_max + width_diff), width)
        y_min = max(round(y_min - height_diff), 0)
        y_max = min(round(y_max + height_diff), height)

        image_arr = image_arr[y_min:y_max, x_min:x_max]
        return image_arr
    else:
        return None
