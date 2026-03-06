#!/usr/bin/env python3

import os
import sys
import argparse
import json

import numpy as np
import pdnl_sana.image
import pdnl_sana.quantify

import pandas as pd
from matplotlib import pyplot as plt

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', help="path to positive mask", required=True)
    parser.add_argument('-m', '--mask_file', help="path to mask file to apply to the image", default=None)
    parser.add_argument('-a', '--activations_file', help="path to output model activations", default=None)
    parser.add_argument('--debug', action='store_true', help="plots an output image to analyze")
    args = parser.parse_args()

    name = os.path.splitext(os.path.basename(args.input_file))[0]

    pos = pdnl_sana.image.Frame(args.input_file)
    pos.img = pos.img // np.max(pos.img)
    if args.mask_file:
        mask = pdnl_sana.image.Frame(args.mask_file)
    else:
        mask = None
    if args.activations_file:
        acts = np.load(args.activations_file)
        class_names = sorted([key for key in acts.keys()])
        acts = np.array([acts[key] for key in class_names])
        def softmax(x, axis):
            e_x = np.exp(x - np.max(x))
            return e_x / np.sum(e_x, axis=axis)
        proba = softmax(acts, axis=0)
        preds = pdnl_sana.image.Frame(np.argmax(proba, axis=0)[:,:,None])
    else:
        preds = None

    num, den = pdnl_sana.quantify.calculate_ao(pos=pos, mask=mask)
    rows = []
    rows.append([name, "Main", num, den])
    if not preds is None:
        for i, class_name in enumerate(class_names):
            class_pos = pdnl_sana.image.frame_like(pos, pos.img & (preds.img == i))
            class_num, den = pdnl_sana.quantify.calculate_ao(pos=class_pos, mask=mask)
            rows.append([name, class_name, class_num, den])

    df = pd.DataFrame(rows, columns=['Name','Measurement','PositivePixels','AreaPixels'])

    if args.debug:
        fig, axs = plt.subplots(1,2)
        axs[0].imshow(pos.img)
        axs[1].imshow(preds.img)
        plt.show()

    print(df)
    
if __name__ == "__main__":
    main()
