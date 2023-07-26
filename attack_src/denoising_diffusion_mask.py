"""
PyTorch Faster-RCNN Resnet50-FPN object detection model
"""
from collections import defaultdict
from copy import deepcopy
from typing import Dict, List, Optional

from art.estimators.object_detection import PyTorchFasterRCNN
import torch
from torchvision.models.detection.faster_rcnn import FasterRCNN
from armory.baseline_models.pytorch.carla_multimodality_object_detection_frcnn_robust_fusion import MultimodalRobust
# from armory.baseline_models.pytorch.carla_multimodality_object_detection_frcnn_robust_fusion import MultimodalRobust
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from loguru import logger

import numpy as np
import cv2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DiffusionFusion(torch.nn.Module):
    def __init__(self, detector: FasterRCNN, diffusion: GaussianDiffusion, threshold, size=256):
        super().__init__()
        self.detector = detector
        self.diffusion = diffusion
        self.threshold = threshold
        self.size = size
        # TODO: parametrize the noise variance
    
    @staticmethod
    def reconstruct_image(img, height, width, size):
        reshaped_tensor = img.view(width//size, height//size, 3, size, size)

        # Concatenate along the width dimension
        concatenated_tensor = torch.cat(tuple(reshaped_tensor[i] for i in range(width//size)), dim=3)

        # Concatenate along the height dimension
        concatenated_tensor = torch.cat(tuple(concatenated_tensor[i] for i in range(height//size)), dim=1)
        
        return concatenated_tensor.view(3, height, width)

    def forward(self, images: List[torch.Tensor], targets: Optional[List[Dict[str, torch.Tensor]]] = None):
        SIZE = self.size
        THRESHOLD = self.threshold

        for i, x in enumerate(images):
            x_rgb, x_depth = x[0:3], x[3:6]
            x_height, x_width = x_rgb.shape[1], x_rgb.shape[2]

            # pad image to be divisible by SIZE
            x_width_pad = SIZE - x_width % SIZE if x_width % SIZE != 0 else 0
            x_height_pad = SIZE - x_height % SIZE if x_height % SIZE != 0 else 0

            x_pad = torch.nn.functional.pad(x_rgb, (0, x_width_pad, 0, x_height_pad), mode='constant', value=0)
            padded_height, padded_width = x_pad.shape[1], x_pad.shape[2]

            x_spliced = []
            for i in range(0, padded_width, SIZE):
                for j in range(0, padded_height, SIZE):
                    x_spliced.append(x_pad[0:3, j:j+SIZE, i:i+SIZE])
            
            x_spliced = torch.stack(x_spliced)
            # TODO: may not need this?
            # img_spliced = img_spliced.unsqueeze(0).float().div(255).to(DEVICE)
            x_spliced = x_spliced.unsqueeze(0).float().to(DEVICE)
            x_spliced = x_spliced[0]

            # image_noisy_spliced = ((img_spliced[0]*2) - 1) + torch.randn_like(img_spliced[0]) * 0.31622776601
            x_noisy_spliced = ((x_spliced*2) - 1) + torch.randn_like(x_spliced) * np.sqrt(0.01)

            num_slices = padded_height * padded_width // (SIZE * SIZE)
            x_reconstructed = self.diffusion.ddim_sample_img(x_noisy_spliced, (num_slices, 3, SIZE, SIZE), return_all_timesteps=True, tqdm_disabled=True)

            x_reconstructed_neg1 = x_reconstructed[:, x_reconstructed.shape[1]-2,:,:,:]

            x_final = self.reconstruct_image(x_reconstructed_neg1, padded_height, padded_width, SIZE)[:, 0:x_height, 0:x_width]

            # normalize x_final
            x_final = x_final - x_final.min()
            x_final = x_final / x_final.max()

            # not sure if we need the summation
            image_diff = abs(x_final - x_rgb).sum(dim=0)

            # dynamic threshold for mask
            mask = image_diff > image_diff.max() * THRESHOLD

            # TODO: adjust size of kernel
            kernel = np.ones((7, 7), dtype='uint8')
            mask = cv2.dilate(mask.cpu().numpy().astype(np.uint8), kernel)

            # duplicate mask for each channel
            mask = torch.stack([torch.from_numpy(mask) for _ in range(6)], dim=0)

            # img_c = img[0:3, :, :]
            # img_d = img[3:6, :, :]
            # mean_value_rgb = img_c[mask[0:3] == False].mean()
            # mean_value_depth = img_d[mask[3:6] == False].mean()

            # # set to mean for respective channels
            # img[mask][0:3] = mean_value_rgb
            # img[mask][3:6] = mean_value_depth

            x.data[mask] = 0

        device = next(self.detector.parameters()).device
        images = [x.to(device) for x in images]
        if targets:
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = self.detector(images, targets)

        return outputs


# NOTE: PyTorchFasterRCNN expects numpy input, not torch.Tensor input
def get_art_model(
    backbone_class, model_kwargs: dict, wrapper_kwargs: dict, weights_path: Optional[str] = None
) -> PyTorchFasterRCNN:
    model_kwargs = deepcopy(model_kwargs)
    diffusion_kwargs = model_kwargs.pop('diffusion_kwargs', {})
    wrapper_kwargs = deepcopy(wrapper_kwargs)

    unet_model = Unet(
        dim = 64,
        dim_mults = (1, 2, 4, 8),
        flash_attn = True
    )

    sampling_timesteps = diffusion_kwargs.pop('diffusion_timesteps', 20)

    diffusion_model = GaussianDiffusion(
        unet_model,
        image_size = 128,
        timesteps = 1000,           # number of steps
        sampling_timesteps = sampling_timesteps    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    )

    devices = model_kwargs.pop('gpus', defaultdict(int))
    logger.info(f'Using GPUs: {devices}')
    device_diffusion = torch.device(f'cuda:{devices["diffusion"]}')
    device_detector = torch.device(f'cuda:{devices["detector"]}')

    num_classes = model_kwargs.pop("num_classes", 3)
    backbone = backbone_class(**model_kwargs)
    detector = FasterRCNN(
        backbone,
        num_classes=num_classes,
        image_mean=[0.485, 0.456, 0.406, 0.0, 0.0, 0.0],
        image_std=[0.229, 0.224, 0.225, 1.0, 1.0, 1.0],
    )

    model = DiffusionFusion(detector, diffusion_model, diffusion_kwargs['mask_threshold'])

    if weights_path:
        checkpoint = torch.load(weights_path, map_location='cpu')
        weight_key = {
            'detector': model.detector,
            'diffusion': model.diffusion
        }
        for k, m in weight_key.items():
            if m is not None:
                m.load_state_dict(checkpoint[k])

    wrapped_model = PyTorchFasterRCNN(
        model,
        clip_values=(0.0, 1.0),
        channels_first=False,
        **wrapper_kwargs,
    )

    m = wrapped_model._model
    m.diffusion.to(device_diffusion)
    m.detector.to(device_detector)

    return wrapped_model

def get_art_model_robust(*args, **kwargs):
    return get_art_model(MultimodalRobust, *args, **kwargs)