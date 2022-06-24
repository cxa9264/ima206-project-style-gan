#!usr/bin/env python
#-*- coding:utf-8 _*-
"""
Author: Tianyu Li
"""

import argparse
from ast import Store
import os
import platform

import torch
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import config as cfg
from FaceAttr_baseline_model import FaceAttrModel
from featuremap_visulize import preprocess_image


def inference(model, device, target, img_root) -> dict:
    """Predict label of images in given path
    
    Args:
        model: predictor
        device: device to use, 'cpu' or 'cuda' or 'mps'
        target: index of attributs
        img_root: path of images
    
    Returns:
        dict: {image_name: label}, prediction scores of images
    """

    model.eval()

    predict_scores = {}
    for img_name in tqdm(os.listdir(img_root)):
        img = cv2.imread(os.path.join(img_root, img_name))

        if img is None:
            continue

        img = np.float32(cv2.resize(img, (224, 224))) / 255
        input_img = preprocess_image(img)

        input_img = input_img.to(device)

        with torch.no_grad():
            predict_scores[img_name] = model(input_img)[0, target]
    
    return predict_scores



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', dest='img_path', type=str, required=True)
    parser.add_argument('--model_path', dest='model_path', type=str, default='./result/Resnet18.pth')
    parser.add_argument('--target', dest='target', nargs='+', type=int, default=20)
    parser.add_argument('--save_path', dest='save_path', type=str, default='./result')
    parser.add_argument('--save_plot', action='store_true')
    parser.add_argument('--move_target', type=int)
    
    args = parser.parse_args()

    if args.target == [-1]:
        args.target = [i for i in range(40)]
    
    model = FaceAttrModel('Resnet18', pretrained=False, selected_attrs=cfg.selected_attrs)

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    elif platform == 'darwin':
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    model = model.to(device)

    # load params
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    print(args.target)
    scores = inference(model, device, np.array(args.target), args.img_path)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    with open(os.path.join(args.save_path, 'res.csv'), 'w') as f:
        for img_name, score in scores.items():
            line = img_name
            for s in score:
                line += ',' + str(s.item())
            f.write(line + '\n')

    if args.save_plot:
        plt.figure(dpi=300, figsize=(20, 10))

        deltas = []

        scores_plt = {s: [] for s in args.target}
        for img_name, score in scores.items():
            deltas.append(float(img_name.split('_')[-1][:-4]))
            for t, s in zip(args.target, score):
                scores_plt[t].append(s.item())
        
        legends = []
        idx = np.argsort(deltas)
        deltas = np.array(deltas)[idx]
        for i, (key, val)in enumerate(scores_plt.items()): 
            legends.append(cfg.selected_attrs[key])
            style = 'x-' if key != args.move_target else 'o-'
            plt.plot(deltas, np.array(val)[idx], style, 
                color=colors.hsv_to_rgb((i / len(args.target), 0.5, 0.8)), 
                linewidth=1.5+(key==args.move_target))
        
        plt.legend(legends, loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title(f"Move {cfg.selected_attrs[args.move_target]}")

        plt.savefig(os.path.join(args.save_path, 'scores.png'), bbox_inches='tight')