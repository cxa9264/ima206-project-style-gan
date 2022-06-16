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
    parser.add_argument('-mb', '--manipulate_boundary', nargs='+', type=str, default=None)
    parser.add_argument('--save_code', type=str, default=None)
    parser.add_argument('--load_code', type=str, default=None)

    args = parser.parse_args()

    boundary_ori = np.load(args.boundary_path)
    if args.intercept_path is not None:
        intercept = np.load(args.intercept_path)
    
    # conditinal manipulation
    boundary = boundary_ori.copy()
    if args.manipulate_boundary is not None:
        if len(args.manipulate_boundary) == 1:
            second_boundary = np.load(args.manipulate_boundary[0])
            boundary = boundary - (boundary @ second_boundary.T) * second_boundary
            boundary /= np.linalg.norm(boundary)
        else:
            conditional_boundary = []
            for path in args.manipulate_boundary:
                conditional_boundary.append(np.load(path))
            conditional_boundary = np.concatenate(conditional_boundary)
            A = conditional_boundary @ conditional_boundary.T
            B = conditional_boundary @ boundary.T
            x = np.linalg.solve(A, B)
            boundary = boundary - x.T @ conditional_boundary
            boundary /= np.linalg.norm(boundary)
        
            print("After manipulation:")
            for b in conditional_boundary:
                print(boundary @ b.T)

    logger = setup_logger(args.save_path, logger_name='generate_data')

    # init model
    model = StyleGANGenerator('stylegan_celebahq', logger)
    assert args.latent_space_type in ['w', 'W', 'z', 'Z'], 'wrong latent space type'
    kwargs = {'latent_space_type': 'z'}

    # generate latent codes
    logger.info('Preparing latent codes...')
    initial_code = model.easy_sample(1, **kwargs)

    if args.latent_space_type in ['w', 'W']:
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            raise AssertionError('No GPU available')

        with torch.no_grad():
            initial_code = model.model.mapping(torch.tensor(initial_code).to(device)).cpu().numpy()
    
    if args.save_code is not None:
        np.save(args.save_code, initial_code)
    
    if args.load_code is not None:
        initial_code = np.load(args.load_code)

    initial_code = initial_code - (initial_code @ boundary_ori.T + intercept[0, 0]) * boundary_ori
    step = np.linspace(0, args.max_delta, args.num_steps)[1:]
    step = np.concatenate([step, -step, [0]])

    if args.manipulate_boundary is not None:
        step = step / (boundary_ori @ boundary.T)
        step = step.flatten()

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


            


