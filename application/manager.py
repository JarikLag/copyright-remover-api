import base64

from PIL import Image
from io import BytesIO

import application.classifier.core as classifier
import application.detector.core as detector
import application.image_transformer.core as transformer
import application.image_generator.inference as generator
import application.image_enhancer.inference as enhancer

labels = {
    0: 'author',
    1: 'author_unmodifiable',
    2: 'non_author'
}


def classify_image(image_raw, models_path=None):
    image = read_image(image_raw)
    label = classifier.classify(image, models_path)
    return labels[label]


def modify_image(image_raw, label, models_path=None):
    size = 256, 256

    image = read_image(image_raw)
    image = detector.detect_and_crop(image, label)
    if image is None:
        return None

    image = transformer.make_thumbnail(image, size)
    trimap = transformer.generate_trimap(image, models_path)
    segmentation = transformer.trimap_to_segmentation(trimap, label)
    segmentation = transformer.make_segmentation_square(segmentation, size)
    generated = generator.generate_image(image, segmentation)
    enhanced = enhancer.enhance(generated, models_path)
    result = Image.fromarray(enhanced).convert('RGB')
    return result


def read_image(image_raw):
    return Image\
        .open(BytesIO(base64.b64decode(image_raw)))\
        .convert('RGB')


if __name__ == '__main__':
    with open('../2.png', 'rb') as f:
        byte_im = f.read()
        encoded = base64.b64encode(byte_im).decode('utf-8')
        #print(classify_image(encoded, '../models'))
        img = modify_image(encoded, 'car', '../models')
        Image.fromarray(img).convert('RGB').show()
