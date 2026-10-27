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
import pdnl_sana.utils

import pandas as pd
from matplotlib import pyplot as plt

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', help="path to pdnl_extract outputs", required=True)
    parser.add_argument('-o', '--output_path', help="path to write outputs to", required=True)
    parser.add_argument('--output_name', required=True)
    parser.add_argument('-p', '--process_path', help="path to pdnl_process outputs", required=True)
    parser.add_argument('-f', '--feature_path', required=False, default="")
    parser.add_argument('-w', '--wildcat_path', help="path to pdnl_wildcat outputs", default=None)
    parser.add_argument('--seg_classes', default=['CSF', 'R', 'GM', 'L'])
    parser.add_argument('--debug', action='store_true', help="plots an output image to analyze")
    args = parser.parse_args()

    microglia_classes = ['rammified', 'dystrophic', 'rod', 'ameboid', 'hypertrophic']

    # generate the sample grids if able
    seg_classes = args.seg_classes
    sample_grids = {}
    deforms = {}
    all_annotations = []
    for f in tqdm(os.listdir(os.path.join(args.input_path, '..')), desc="generating sample grids"):
        if f.endswith('.geojson') and not f.startswith('._'):
            roi_name = f.replace('.geojson', '')
            annotations = pdnl_sana.utils.read_geojson(os.path.join(os.path.join(args.input_path, '..'), f))
            all_annotations += annotations
            segments = []
            for cls in seg_classes:
                for a in annotations:
                    if a.class_name == cls:
                        segments.append(a.to_curve())
                        break
                else:
                    print(cls, roi_name)

            sample_grids[roi_name] = pdnl_sana.interpolate.fan_sample(*segments)[0]

    roi_occu, roi_area, deforms = {}, {}, {}
    for roi_name in os.listdir(args.process_path):
        roi_directory = os.path.join(args.process_path, roi_name)
        if not os.path.isdir(roi_directory):
            continue

        # load the processed data
        logger = pdnl_sana.logging.Logger('normal', os.path.join(roi_directory, 'log.pkl'))
        stain = pdnl_sana.image.Frame(os.path.join(roi_directory, 'stain.npy'))
        pos = pdnl_sana.image.Frame(os.path.join(roi_directory, 'pos.png'))
        mask = pdnl_sana.image.Frame(os.path.join(roi_directory, 'mask.png'))

        pos.to_binary()
        mask.to_binary()

        # fig, ax = plt.subplots(1,1)
        # ax.imshow(pos.img)
        # for a in all_annotations:
        #     x = a.copy()
        #     x.translate(logger.data['frame_loc'])
        #     ax.plot(*x.T)
        # plt.show()


        num, den = pdnl_sana.quantify.calculate_ao(pos=pos, mask=mask)

        roi_occu[roi_name] = num
        roi_area[roi_name] = den

        # apply the sample grid if relevant
        # TODO: this is not working, check that frame_loc is correct
        #.       if it is, then idk maybe the frame size isn't, or the cropping
        if roi_name in sample_grids:
            print(logger.data['frame_loc'])
            sample_grid = sample_grids[roi_name] - logger.data['frame_loc'][::-1]
            l = pos.img.shape[0]
            # make sure the patch overlaps with the sample grid
            if np.sum((sample_grid > 0) & (sample_grid < l)) > 0:
                out = pdnl_sana.interpolate.grid_sample(pos, sample_grid)
                # fig, ax = plt.subplots(1,2)
                # ax[0].imshow(pos.img)
                # ax[1].imshow(out.img)
                # plt.show()
                deforms[roi_name] = out
            
        # # load in the WILDCAT results
        # if not args.wildcat_path is None:

        #     # get probability and prediction of each class
        #     wildcat_d = os.path.join(args.wildcat_path, f'{i}_{j}')
        #     activations = np.load(os.path.join(wildcat_d, 'wildcat_proba.npz'))
        #     class_names = sorted([key for key in activations.keys()])
        #     activations = np.array([activations[key] for key in class_names])
        #     def softmax(x, axis):
        #         e_x = np.exp(x - np.max(x))
        #         return e_x / np.sum(e_x, axis=axis)
        #     proba = softmax(activations, axis=0).transpose(1,2,0)
        #     preds = pdnl_sana.image.Frame(np.argmax(proba, axis=2)[:,:,None])

        #     # find the positive pixels for each WILDCAT class
        #     for cls, class_name in enumerate(class_names):
        #         pred_cls = pdnl_sana.image.frame_like(preds, preds.img == cls)
        #         pos_cls = pos.copy()
        #         pos_cls.mask(pred_cls)
        #         polys = pos_cls.to_polygons()[0]

        #         # store the confidence and prediction of each instance, only if its in a specific ROI mask
        #         for x in polys:

        #             # store each confidence value associated with each annotation
        #             for roi_name, roi_mask in zip(roi_classes, roi_masks):
        #                 poly_mask = pdnl_sana.image.create_mask_like(roi_mask, [x])
        #                 poly_mask.mask(roi_mask)
        #                 poly_proba = proba[poly_mask.img[:,:,0] != 0]
        #                 if len(poly_proba) != 0:
        #                     poly_proba = np.mean(poly_proba, axis=0)
        #                     record = {
        #                         'id': str(len(annos)),
        #                         'ROI': roi_name,
        #                         'area': x.get_area(),
        #                         'pred': class_name,
        #                     }
        #                     for i in range(len(poly_proba)):
        #                         record[class_names[i]] = poly_proba[i]
        #                     wildcat_df.append(record)

        #             # store the instance as an annotation to be plotted on the WSI
        #             x.translate(-loc)
        #             annos.append(x.to_annotation(class_name=class_name, annotation_name=str(len(annos))))

        
    #     features_f = os.path.join(args.process_path, args.feature_path+'.npy')
    #     if os.path.exists(features_f):
    #         features = np.load(features_f)
            
    #         for feat in features:
    #             instance_loc = pdnl_sana.geo.Point(*feat[:2], is_micron=False, level=level)
    #             instance_size = pdnl_sana.geo.Point(*feat[2:4], is_micron=False, level=level)
    #             instance_loc.translate(loc)
                
    #             instance_x, instance_y = np.rint(instance_loc).astype(int)
    #             if instance_x >= frame.size()[0] or instance_y >= frame.size()[1] or instance_x < 0 or instance_y < 0:
    #                 continue

    #             for roi_name, roi_mask in zip(roi_classes, roi_masks):
    #                 if roi_mask.img[instance_y, instance_x] != 0:
    #                     record = {
    #                         'chunk': d,
    #                         'chunk_x': chunk_x,
    #                         'chunk_y': chunk_y,
    #                         'id': str(len(annos)),
    #                         'x': instance_x+loc[0],
    #                         'y': instance_y+loc[1],
    #                         'w': instance_size[0],
    #                         'h': instance_size[1],
    #                         'features': feat[4:-5],
    #                         'ROI': roi_name,
    #                     }
    #                     for class_idx, class_name in enumerate(microglia_classes):
    #                         record[class_name] = feat[-5:][class_idx]
    #                     microglia_df.append(record)

    #             microglia_cls = microglia_classes[np.argmax(feat[-5:])] 
    #             anno_loc = pdnl_sana.geo.Point(instance_x+loc[0], instance_y+loc[1])
    #             rect = pdnl_sana.geo.rectangle_like(anno_loc, anno_loc, instance_size)
    #             annos.append(rect.to_annotation(class_name=microglia_cls, annotation_name=str(len(annos))))
                        
    # if len(annos) != 0:
    #     with open(os.path.join(args.output_path, args.output_name+'.geojson'), 'w') as fp:
    #         geojson.dump([x.to_geojson() for x in annos], fp)

    name = os.path.split(os.path.split(args.input_path.strip('/\\'))[0])[1]
    measurement = os.path.split(args.process_path.strip('/\\'))[1]
    ao_df = []
    for roi_name in roi_occu:
        ao_df.append({
            'Name': name,
            'ROI': roi_name,
            'Measurement': measurement,
            'Area': roi_area[roi_name],
            'Positive': roi_occu[roi_name],
        })
        # save the deformed data
        if roi_name in deforms:
            deform = deforms[roi_name]
            num, den = pdnl_sana.quantify.bin_cortex(deform, nbins=12)
            ao_df[-1]['BIN_NUM'] = num
            ao_df[-1]['BIN_DEN'] = den

    # ao_df.append({
    #     'Name': name,
    #     'ROI': 'TOTAL',
    #     'Measurement': measurement,
    #     'Area': sum([roi_area[roi_name]) for roi_name in roi_area]),
    #     'Positive': sum([sum(roi_occu[roi_name]) for roi_name in roi_area]),
    # })

    # microglia_df = pd.DataFrame(microglia_df)


    # wildcat_df = pd.DataFrame(wildcat_df)
    # if len(wildcat_df) != 0:
    #     model_name = os.path.split(args.wildcat_path.strip('/\\'))[1]
    #     for column in class_names:
    #         for roi_name in roi_classes:
    #             df = wildcat_df[(wildcat_df['pred'] == column) & (wildcat_df['ROI'] == roi_name)]
    #             ao_df.append({
    #                 'Name': name,
    #                 'ROI': roi_name,
    #                 'Measurement': f'{model_name}_{column}',
    #                 'Area': sum(roi_area[roi_name]),
    #                 'Positive': sum(df['area']),
    #             })

    #         df = wildcat_df[wildcat_df['pred'] == column].drop_duplicates('id')
    #         ao_df.append({
    #             'Name': name,
    #             'ROI': 'TOTAL',
    #             'Measurement': f'{model_name}_{column}',
    #             'Area': sum([sum(roi_area[roi_name]) for roi_name in roi_area]),
    #             'Positive': sum(df['area']),
    #         })
        
    ao_df = pd.DataFrame(ao_df)
    ao_df['AO'] = ao_df['Positive'] / ao_df['Area']

    ao_df.to_csv(os.path.join(args.output_path, 'ao.csv'), index=False)
    # wildcat_df.to_csv(os.path.join(args.output_path, 'detections.csv'), index=False)
    # microglia_df.to_csv(os.path.join(args.output_path, args.output_name+'.csv'), index=False)
        
if __name__ == "__main__":
    main()
