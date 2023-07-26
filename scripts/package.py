import argparse

import torch
import torch.nn as nn
from loguru import logger


def load(model: nn.Module, path: str):
    ckpt = torch.load(path)
    ckpt = ckpt.get('state_dict') or ckpt
    model.load_state_dict(ckpt)


if __name__ == '__main__':
    default_detector = '/nobackup-raid5/gozum/armory/saved_models/carla_multimodal_robust_weights_eval7and8.pt'
    default_diffusion = './results/07_07_birdseye/model-70.pt'
    default_output = './results/packaged_weights_robust.pt'

    parser = argparse.ArgumentParser()
    parser.add_argument('--detector', type=str, default=default_detector, help='path to detector weight')
    parser.add_argument('--diffusion', type=str, default=default_diffusion, help='path to diffusion weight')
    parser.add_argument('-o', '--output', type=str, default=default_output, help='path to output weight')
    args = parser.parse_args()

    logger.info(f'Using detector "{args.detector}".')
    logger.info(f'Using checker "{args.diffusion}".')
    logger.info(f'Saving to "{args.output}".')

    ae = None

    checkpoint = dict(
        detector=torch.load(args.detector),
        diffusion=torch.load(args.diffusion)['model']
    )
    torch.save(checkpoint, args.output)