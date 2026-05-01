#!/usr/bin/env python3

import os
import sys
import argparse
import json

import geojson
import numpy as np
from tqdm import tqdm
import pdnl_sana.image
import pdnl_sana.quantify
import pdnl_sana.logging

import pandas as pd
from matplotlib import pyplot as plt

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', help="path to pdnl_extract outputs", required=True)
    parser.add_argument('-o', '--output_path', help="path to write outputs to", required=True)    
    parser.add_argument('-p', '--process_path', help="path to pdnl_process outputs", required=True)
    parser.add_argument('-w', '--wildcat_path', help="path to pdnl_wildcat outputs", default=None)
    parser.add_argument('--debug', action='store_true', help="plots an output image to analyze")
    args = parser.parse_args()

    ao_df, wildcat_df = [], []
    roi_occu, roi_area = {}, {}
    annos = []    
    for d in tqdm(os.listdir(args.input_path), desc="Aggregating Chunks"):
        if not os.path.isdir(os.path.join(args.input_path, d)):
            continue
        
        i, j = list(map(int, d.split('_')))

        in_d = os.path.join(args.input_path, f'{i}_{j}')

        logger = pdnl_sana.logging.Logger('normal', os.path.join(in_d, 'log.pkl'))
        level = logger.data['level']
        mpp = logger.data['mpp']
        ds = logger.data['ds']
        converter = pdnl_sana.geo.Converter(mpp, ds)
        frame = pdnl_sana.image.Frame(os.path.join(in_d, 'frame.png'), level=level, converter=converter)
        mask = pdnl_sana.image.Frame(os.path.join(in_d, 'mask.png'), level=level, converter=converter)
        mask.to_binary()

        roi_classes = []
        roi_masks = []
        for f in os.listdir(os.path.join(in_d)):
            if 'mask' in f and not f == 'mask.png':
                roi_classes.append(os.path.splitext(f)[0].split('_', 1)[1])
                roi_mask = pdnl_sana.image.Frame(os.path.join(in_d, f), level=level, converter=converter)
                roi_mask.to_binary()
                roi_masks.append(roi_mask)
                
        process_d = os.path.join(args.process_path, f'{i}_{j}')
        stain = pdnl_sana.image.Frame(os.path.join(process_d, 'stain.npy'))
        pos = pdnl_sana.image.Frame(os.path.join(process_d, 'pos.png'))
        pos.to_binary()

        for roi_name, roi_mask in zip(roi_classes, roi_masks):
            if not roi_name in roi_occu:
                roi_occu[roi_name] = []
                roi_area[roi_name] = []
                num, den = pdnl_sana.quantify.calculate_ao(pos=pos, mask=roi_mask)
            roi_occu[roi_name].append(num)
            roi_area[roi_name].append(den)
            
        if not args.wildcat_path is None:
            wildcat_d = os.path.join(args.wildcat_path, f'{i}_{j}')
            activations = np.load(os.path.join(wildcat_d, 'wildcat_proba.npz'))
            class_names = sorted([key for key in activations.keys()])
            activations = np.array([activations[key] for key in class_names])
            
            def softmax(x, axis):
                e_x = np.exp(x - np.max(x))
                return e_x / np.sum(e_x, axis=axis)
            proba = softmax(activations, axis=0).transpose(1,2,0)
            preds = pdnl_sana.image.Frame(np.argmax(proba, axis=2)[:,:,None])

            x = int(i * frame.size()[0])
            y = int(j * frame.size()[1])
            loc = pdnl_sana.geo.Point(x, y, level=level, is_micron=False)

            for cls, class_name in enumerate(class_names):
                pred_cls = pdnl_sana.image.frame_like(preds, preds.img == cls)
                pos_cls = pos.copy()
                pos_cls.mask(pred_cls)
                polys = pos_cls.to_polygons()[0]

                for x in polys:

                    # store each confidence value associated with each annotation
                    for roi_name, roi_mask in zip(roi_classes, roi_masks):
                        poly_mask = pdnl_sana.image.create_mask_like(roi_mask, [x])
                        poly_mask.mask(roi_mask)
                        poly_proba = proba[poly_mask.img[:,:,0] != 0]
                        if len(poly_proba) != 0:
                            poly_proba = np.mean(poly_proba, axis=0)
                            record = {
                                'id': str(len(annos)),
                                'ROI': roi_name,
                                'area': x.get_area(),
                                'pred': class_name,
                            }
                            for i in range(len(poly_proba)):
                                record[class_names[i]] = poly_proba[i]
                            wildcat_df.append(record)

                    x.translate(-loc)
                    annos.append(x.to_annotation(class_name=class_name, annotation_name=str(len(annos))))                

        polys = pos.to_polygons()[0]
        [x.translate(-loc) for x in polys]        
        
    with open(os.path.join(args.output_path, 'detections.geojson'), 'w') as fp:
        geojson.dump([x.to_geojson() for x in annos], fp)

    name = os.path.split(os.path.split(args.input_path.strip('/\\'))[0])[1]
    measurement = os.path.split(args.process_path.strip('/\\'))[1]
    for roi_name in roi_occu:
        ao_df.append({
            'Name': name,
            'ROI': roi_name,
            'Measurement': measurement,
            'Area': sum(roi_area[roi_name]),
            'Positive': sum(roi_occu[roi_name]),
        })
    ao_df.append({
        'Name': name,
        'ROI': 'TOTAL',
        'Measurement': measurement,
        'Area': sum([sum(roi_area[roi_name]) for roi_name in roi_area]),
        'Positive': sum([sum(roi_occu[roi_name]) for roi_name in roi_area]),
    })

    wildcat_df = pd.DataFrame(wildcat_df)
    if len(wildcat_df) != 0:
        model_name = os.path.split(args.wildcat_path.strip('/\\'))[1]
        for column in class_names:
            for roi_name in roi_classes:
                df = wildcat_df[(wildcat_df['pred'] == column) & (wildcat_df['ROI'] == roi_name)]
                ao_df.append({
                    'Name': name,
                    'ROI': roi_name,
                    'Measurement': f'{model_name}_{column}',
                    'Area': sum(roi_area[roi_name]),
                    'Positive': sum(df['area']),
                })

            df = wildcat_df[wildcat_df['pred'] == column].drop_duplicates('id')
            ao_df.append({
                'Name': name,
                'ROI': 'TOTAL',
                'Measurement': f'{model_name}_{column}',
                'Area': sum([sum(roi_area[roi_name]) for roi_name in roi_area]),
                'Positive': sum(df['area']),
            })
        
    ao_df = pd.DataFrame(ao_df)
    ao_df['AO'] = ao_df['Positive'] / ao_df['Area']

    ao_df.to_csv(os.path.join(args.output_path, 'ao.csv'), index=False)
    wildcat_df.to_csv(os.path.join(args.output_path, 'detections.csv'), index=False)
    
if __name__ == "__main__":
    main()
