"""
Author: Tianyu Li
"""

import enum
import os
import argparse
from collections import defaultdict

import numpy as np
import cv2
from tqdm import tqdm
import torch

from models.model_settings import MODEL_POOL
from models.stylegan_generator import StyleGANGenerator
from utils.logger import setup_logger

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', dest='save_path', type=str, required=True)
    parser.add_argument('--boundary', dest='boundary_path', type=str, required=True)
    parser.add_argument('--latent_space_type', dest='latent_space_type', type=str, default='z')
    parser.add_argument('--max_delta', dest='max_delta', type=int, default=2)
    parser.add_argument('--num_steps', dest='num_steps', type=int, default=10)
    parser.add_argument('--generate_style', action='store_true',
                            help='If specified, will generate layer-wise style codes '
                           'in Style GAN. (default: do not generate styles)')
    parser.add_argument('-I', '--generate_image', action='store_false',
                            help='If specified, will skip generating images in '
                           'Style GAN. (default: generate images)')
    parser.add_argument('--intercept', dest='intercept_path', type=str, default=None)

    args = parser.parse_args()

    boundary = np.load(args.boundary_path)
    if args.intercept_path is not None:
        intercept = np.load(args.intercept_path)

    logger = setup_logger(args.save_path, logger_name='generate_data')

    # init model
    model = StyleGANGenerator('stylegan_celebahq', logger)
    assert args.latent_space_type in ['w', 'W', 'z', 'Z'], 'wrong latent space type'
    kwargs = {'latent_space_type': args.latent_space_type}

    # generate latent codes
    logger.info('Preparing latent codes...')
    initial_code = model.easy_sample(1, **kwargs)
    initial_code = initial_code # - (initial_code @ boundary.T - intercept[0, 0]) * boundary 
    step = np.linspace(0, args.max_delta, args.num_steps)[1:]
    step = np.concatenate([step, -step, [0]])
    step.sort()
    latent_codes = initial_code + step.reshape(-1, 1) * boundary

    # generate images
    logger.info('Generating images...')
    for i, latent_code in enumerate(tqdm(latent_codes)):
        outputs = model.easy_synthesize(
            latent_code[None],
            **kwargs,
            generate_style=args.generate_style,
            generate_image=args.generate_image
        )

        for key, val in outputs.items():
            if key == 'image':
                for image in val:
                    cv2.imwrite(os.path.join(args.save_path, f'{i}_{step[i]}.jpg'), image[:, :, ::-1])


            


