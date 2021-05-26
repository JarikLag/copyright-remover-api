import numpy as np
import torch
import application.image_enhancer.RRDBNet_arch as arch


def enhance(image, models_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models_path = '../models' if models_path is None else models_path

    model = arch.RRDBNet(3, 3, 64, 23, gc=32).to(device)
    checkpoint = torch.load(f'{models_path}/esrgan.pth')
    model.load_state_dict(checkpoint, strict=True)

    model.eval()

    img = image * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_lr = img.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_lr).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round().astype(np.uint8)
    return output
