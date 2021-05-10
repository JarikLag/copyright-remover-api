import torch
import timm
import numpy as np
import torchvision.transforms as transforms

from PIL import Image


def classify(image, models_path=None):
    size = 224, 224

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize(mean, std)])

    path = 'models' if models_path is None else models_path

    model = timm.create_model('resnest14d', pretrained=False).to(device)
    checkpoint = torch.load(f'{path}/classifier.pth')
    model.load_state_dict(checkpoint)

    model.eval()
    with torch.no_grad():
        image = image.resize(size, Image.BICUBIC)
        image = image_transform(image).unsqueeze(0).to(device)
        outputs = model(image).cpu()
        _, prediction = torch.max(outputs.data, 1)
        prediction = prediction.squeeze().item()
        return prediction
