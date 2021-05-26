import numpy as np
import torchvision.transforms as transforms

from application.image_generator.spade.options.test_options import TestOptions
from application.image_generator.spade.models.pix2pix_model import Pix2PixModel
from application.image_generator.spade.util import util


def generate_image(image, segmentation):
    opt = TestOptions().parse()

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize(mean, std)])
    label_transform = transforms.ToTensor()

    model = Pix2PixModel(opt)
    model.eval()

    label_tensor = (label_transform(segmentation) * 255.0).unsqueeze(0)
    image_tensor = image_transform(image).unsqueeze(0)

    data_i = {
        'label': label_tensor,
        'instance': label_tensor,
        'image': image_tensor
    }

    generated = model(data_i, mode='inference')[0]
    image_np = util.tensor2im(generated)

    return image_np
