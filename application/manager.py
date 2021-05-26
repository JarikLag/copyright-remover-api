from PIL import Image

import application.classifier.core as classifier
import application.detector.core as detector
import application.image_transformer.core as transformer
import application.image_generator.inference as generator
import application.image_enhancer.inference as enhancer
from application.matting import matting


def classify_image(image, models_path=None):
    label = classifier.classify(image, models_path)
    return label


def cut_object(image, models_path=None):
    image = detector.detect_and_crop_biggest(image)
    if image is None:
        raise ValueError('Object not found on image')
    trimap = transformer.generate_trimap(image, models_path)
    fg, _, alpha = matting.perform_matting(image, trimap, models_path)
    fixed_image = transformer.swap_bg(fg[:, :, ::-1] * 255, alpha)

    return Image.fromarray(fixed_image).convert('RGB')


def modify_image(image, models_path=None):
    size = 256, 256

    image, label = detector.detect_and_crop_specified_labels(image, transformer.allowed_labels)
    if image is None:
        raise ValueError(f'Object not found on image. Allowed classes: {transformer.allowed_labels}')

    image = transformer.make_thumbnail(image, size)
    image = transformer.make_image_square(image, size, 'RGB')

    trimap = transformer.generate_trimap(image, models_path)

    segmentation = transformer.trimap_to_segmentation(trimap, label)
    segmentation = transformer.make_image_square(segmentation, size)

    fg, _, alpha = matting.perform_matting(image, trimap, models_path)
    fixed_image = transformer.swap_bg(fg[:, :, ::-1] * 255, alpha)

    generated = generator.generate_image(fixed_image, segmentation)

    enhanced = enhancer.enhance(generated, models_path)

    return Image.fromarray(enhanced).convert('RGB')


if __name__ == '__main__':
    pass
