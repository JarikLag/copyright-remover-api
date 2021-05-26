import numpy as np

from detecto import core


def detect_and_crop_biggest(image):
    image_arr = np.array(image)
    box = get_object_box_biggest(image_arr)
    if box is not None:
        return crop(image_arr, box)
    else:
        return None


def detect_and_crop_specified_labels(image, label_list):
    model = core.Model()
    labels, boxes, _ = model.predict_top(image)
    max_sq = -1
    max_box = None
    max_label = None
    for label in label_list:
        if label in labels:
            idx = labels.index(label)
            x_min, y_min, x_max, y_max = boxes[idx]
            sq = (x_max - x_min) * (y_max - y_min)
            if sq > max_sq:
                max_sq = sq
                max_box = boxes[idx]
                max_label = label
    if max_box is not None:
        return crop(np.array(image), max_box), max_label
    else:
        return None


def get_object_box_biggest(image):
    model = core.Model()
    _, boxes, _ = model.predict_top(image)
    max_sq = -1
    max_box = None
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        sq = (x_max - x_min) * (y_max - y_min)
        if sq > max_sq:
            max_sq = sq
            max_box = box
    return max_box


def crop(image_arr, box):
    height, width = image_arr.shape[:2]

    box = list(map(lambda x: float(x), box))
    x_min, y_min, x_max, y_max = box

    width_diff = (x_max - x_min) * 0.05
    height_diff = (y_max - y_min) * 0.05

    x_min = max(round(x_min - width_diff), 0)
    x_max = min(round(x_max + width_diff), width)
    y_min = max(round(y_min - height_diff), 0)
    y_max = min(round(y_max + height_diff), height)

    image_arr = image_arr[y_min:y_max, x_min:x_max]
    return image_arr
