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
import pdnl_sana.interpolate

import pandas as pd
from matplotlib import pyplot as plt

def read_geojson(f):
    annotations = []
    data = geojson.load(open(f, 'r'))
    if 'features' in data:
        data = data['features']
    for ann in data:
        
        xy = np.squeeze(np.array(ann['geometry']['coordinates'])).astype(float)
        #cls = ann['properties']['classification']['name']
        try:
            cls = ann['properties']['classification']['name']
            name = ann['properties']['name']
        except:
            try:
                cls = ann['properties']['name']
                name = None
            except:
                cls = None
                name = None

        annotation = pdnl_sana.geo.Annotation(*xy.T, class_name=cls, annotation_name=name, level=0)

        # shape checking
        if (len(annotation.shape) != 2) or \
           (annotation.shape[0] < 2) or \
           (annotation.shape[1] != 2):
            print(f"ERROR: Improper polygon -- {annotation.shape} | {annotation.class_name}")
            exit()
        
        annotations.append(annotation)
        
    return annotations

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', help="path to pdnl_extract outputs", required=True)
    parser.add_argument('-o', '--output_path', help="path to write outputs to", required=True)
    parser.add_argument('--output_name', required=True)
    parser.add_argument('-p', '--process_path', help="path to pdnl_process outputs", required=True)
    parser.add_argument('-f', '--feature_path', required=False, default="")
    parser.add_argument('-w', '--wildcat_path', help="path to pdnl_wildcat outputs", default=None)
    parser.add_argument('--debug', action='store_true', help="plots an output image to analyze")
    args = parser.parse_args()

    microglia_classes = ['rammified', 'dystrophic', 'rod', 'ameboid', 'hypertrophic']

    # generate the sample grids if able
    seg_classes = ['CSF_GM', 'R', 'GM_WM', 'L']
    sample_grids = {}
    deforms = {}
    for f in tqdm(os.listdir(os.path.join(args.input_path, '..')), desc="generating sample grids"):
        if f.endswith('.geojson'):
            roi_name = f.replace('.geojson', '')
            annotations = read_geojson(os.path.join(os.path.join(args.input_path, '..'), f))
            segments = []
            for cls in seg_classes:
                for a in annotations:
                    if a.class_name == cls:
                        segments.append(a.to_curve())
                        break
                else:
                    print(cls, roi_name)
            sample_grids[roi_name] = pdnl_sana.interpolate.fan_sample(*segments)[0][::8, ::8, :]

    
    ao_df, wildcat_df, microglia_df = [], [], []
    roi_occu, roi_area = {}, {}
    annos = []    
    for d in tqdm(os.listdir(args.input_path), desc="Aggregating Chunks"):
        if not os.path.isdir(os.path.join(args.input_path, d)):
            continue
        
        i, j = list(map(int, d.split('_')))

        in_d = os.path.join(args.input_path, f'{i}_{j}')

        # load the WSI extraction data
        logger = pdnl_sana.logging.Logger('normal', os.path.join(in_d, 'log.pkl'))
        level = logger.data['level']
        mpp = logger.data['mpp']
        ds = logger.data['ds']
        converter = pdnl_sana.geo.Converter(mpp, ds)
        frame = pdnl_sana.image.Frame(os.path.join(in_d, 'frame.png'), level=level, converter=converter)
        mask = pdnl_sana.image.Frame(os.path.join(in_d, 'mask.png'), level=level, converter=converter)
        mask.to_binary()

        # find where the chunk is in the WSI
        chunk_x = int(i * frame.size()[0])
        chunk_y = int(j * frame.size()[1])
        loc = pdnl_sana.geo.Point(chunk_x, chunk_y, level=level, is_micron=False)
        
        # load the individual ROI masks
        roi_classes = []
        roi_masks = []
        for f in os.listdir(os.path.join(in_d)):
            if 'mask' in f and not f == 'mask.png':
                roi_classes.append(os.path.splitext(f)[0].split('_', 1)[1])
                roi_mask = pdnl_sana.image.Frame(os.path.join(in_d, f), level=level, converter=converter)
                roi_mask.to_binary()
                roi_masks.append(roi_mask)
                
        # load the processed data
        process_d = os.path.join(args.process_path, f'{i}_{j}')
        stain = pdnl_sana.image.Frame(os.path.join(process_d, 'stain.npy'))
        pos = pdnl_sana.image.Frame(os.path.join(process_d, 'pos.png'))
        pos.to_binary()

        # for each ROI, calculate the positive stain and ROI area
        for roi_name, roi_mask in zip(roi_classes, roi_masks):
            if not roi_name in roi_occu:
                roi_occu[roi_name] = []
                roi_area[roi_name] = []
            num, den = pdnl_sana.quantify.calculate_ao(pos=pos, mask=roi_mask)
            roi_occu[roi_name].append(num)
            roi_area[roi_name].append(den)

            # apply the sample grid if relevant
            if roi_name in sample_grids:
                sample_grid = sample_grids[roi_name] - loc[::-1]
                l = pos.img.shape[0]
                # make sure the patch overlaps with the sample grid
                if np.sum((sample_grid > 0) & (sample_grid < l)) > 0:
                    out = pdnl_sana.interpolate.grid_sample(pos, sample_grid)
                    if not roi_name in deforms:
                        deforms[roi_name] = out
                    else:
                        deforms[roi_name].img += out.img
                
        # load in the WILDCAT results
        if not args.wildcat_path is None:

            # get probability and prediction of each class
            wildcat_d = os.path.join(args.wildcat_path, f'{i}_{j}')
            activations = np.load(os.path.join(wildcat_d, 'wildcat_proba.npz'))
            class_names = sorted([key for key in activations.keys()])
            activations = np.array([activations[key] for key in class_names])
            def softmax(x, axis):
                e_x = np.exp(x - np.max(x))
                return e_x / np.sum(e_x, axis=axis)
            proba = softmax(activations, axis=0).transpose(1,2,0)
            preds = pdnl_sana.image.Frame(np.argmax(proba, axis=2)[:,:,None])

            # find the positive pixels for each WILDCAT class
            for cls, class_name in enumerate(class_names):
                pred_cls = pdnl_sana.image.frame_like(preds, preds.img == cls)
                pos_cls = pos.copy()
                pos_cls.mask(pred_cls)
                polys = pos_cls.to_polygons()[0]

                # store the confidence and prediction of each instance, only if its in a specific ROI mask
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

                    # store the instance as an annotation to be plotted on the WSI
                    x.translate(-loc)
                    annos.append(x.to_annotation(class_name=class_name, annotation_name=str(len(annos))))

        
        features_f = os.path.join(args.process_path, args.feature_path+'.npy')
        if os.path.exists(features_f):
            features = np.load(features_f)
            
            for feat in features:
                instance_loc = pdnl_sana.geo.Point(*feat[:2], is_micron=False, level=level)
                instance_size = pdnl_sana.geo.Point(*feat[2:4], is_micron=False, level=level)
                instance_loc.translate(loc)
                
                instance_x, instance_y = np.rint(instance_loc).astype(int)
                if instance_x >= frame.size()[0] or instance_y >= frame.size()[1] or instance_x < 0 or instance_y < 0:
                    continue

                for roi_name, roi_mask in zip(roi_classes, roi_masks):
                    if roi_mask.img[instance_y, instance_x] != 0:
                        record = {
                            'chunk': d,
                            'chunk_x': chunk_x,
                            'chunk_y': chunk_y,
                            'id': str(len(annos)),
                            'x': instance_x+loc[0],
                            'y': instance_y+loc[1],
                            'w': instance_size[0],
                            'h': instance_size[1],
                            'features': feat[4:-5],
                            'ROI': roi_name,
                        }
                        for class_idx, class_name in enumerate(microglia_classes):
                            record[class_name] = feat[-5:][class_idx]
                        microglia_df.append(record)

                microglia_cls = microglia_classes[np.argmax(feat[-5:])] 
                anno_loc = pdnl_sana.geo.Point(instance_x+loc[0], instance_y+loc[1])
                rect = pdnl_sana.geo.rectangle_like(anno_loc, anno_loc, instance_size)
                annos.append(rect.to_annotation(class_name=microglia_cls, annotation_name=str(len(annos))))
                        
    if len(annos) != 0:
        with open(os.path.join(args.output_path, args.output_name+'.geojson'), 'w') as fp:
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
        # save the deformed data
        if roi_name in deforms:
            deform = deforms[roi_name]
            num, den = pdnl_sana.quantify.bin_cortex(deform, nbins=12)
            ao_df[-1]['BIN_NUM'] = num
            ao_df[-1]['BIN_DEN'] = den

    ao_df.append({
        'Name': name,
        'ROI': 'TOTAL',
        'Measurement': measurement,
        'Area': sum([sum(roi_area[roi_name]) for roi_name in roi_area]),
        'Positive': sum([sum(roi_occu[roi_name]) for roi_name in roi_area]),
    })

    microglia_df = pd.DataFrame(microglia_df)


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
    microglia_df.to_csv(os.path.join(args.output_path, args.output_name+'.csv'), index=False)
        
if __name__ == "__main__":
    main()
