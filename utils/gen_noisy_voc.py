import os
import argparse
import numpy as np

import mmcv
from voc07_dataset import VOCDataset

def gen_noisy_gt(data_infos, box_noise_level=0.0):
    print('\ngenerating noisy ground-truth ...')
    data_len = len(data_infos)
    for idx in range(data_len):

        img_w, img_h = data_infos[idx]['width'], data_infos[idx]['height']
        anno = data_infos[idx]['ann']
        bboxes = anno['bboxes']
        
        # perturb bbox
        if box_noise_level > 0:
            noisy_bboxes = []
            for i, bbox in enumerate(bboxes):
                x1, y1, x2, y2 = bbox
                x, y, w, h = x1, y1, x2-x1, y2-y1
                cx, cy = (x1+x2)/2, (y1+y2)/2

                # shift bbox x-y
                xy_rand_range, wh_rand_range = box_noise_level, box_noise_level
                x_offset = w * np.random.uniform(-xy_rand_range, xy_rand_range)
                y_offset = h * np.random.uniform(-xy_rand_range, xy_rand_range)
                noisy_cx, noisy_cy = cx + x_offset, cy + y_offset

                # scale bbox w-h
                noisy_w = w * (np.random.uniform(-wh_rand_range, wh_rand_range) + 1.0)
                noisy_h = h * (np.random.uniform(-wh_rand_range, wh_rand_range) + 1.0)

                # noisy coordinates
                noisy_x1, noisy_y1, noisy_x2, noisy_y2 = max(0, noisy_cx - noisy_w/2), max(0, noisy_cy - noisy_h/2), min(noisy_cx + noisy_w/2, img_w - 1), min(noisy_cy + noisy_h/2, img_h-1)
                
                # eliminate invalid noisy bbox
                if noisy_x2 <= noisy_x1 or noisy_y2 <= noisy_y1:
                    noisy_bboxes.append(bbox)
                    continue

                noisy_bboxes.append([noisy_x1, noisy_y1, noisy_x2, noisy_y2])
            
            # save noisy gt
            data_infos[idx]['ann']['bboxes'] = np.array(noisy_bboxes).astype(np.float32)
        
    print('done')
    return data_infos
       
def load_annotations(ann_file):
    return mmcv.load(ann_file)

def save_annotations(outputs, ann_file):
    return mmcv.dump(outputs, ann_file)

def preprocess_dataset(prefix='./data/VOCdevkit/VOC2007/'):
    print('\npreprocess VOC dataset ...')
    voc = VOCDataset(prefix)

    # load trainval data & save as pkl
    ann_file_trainval = prefix + 'ImageSets/Main/trainval.txt'
    anno_trainval = voc.load_annotations(ann_file_trainval)
    save_path_trainval = prefix + 'voc07_trainval.pkl'
    save_annotations(anno_trainval, save_path_trainval)

    # load test data & save as pkl
    ann_file_test = prefix + 'ImageSets/Main/test.txt'
    anno_test = voc.load_annotations(ann_file_test)
    save_path_test = prefix + 'voc07_test.pkl'
    save_annotations(anno_test, save_path_test)
    print('preprocess done')
    return save_path_trainval

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate noisy VOC2007')
    parser.add_argument(
        '--box_noise_level',
        type=float,
        default='0.4')
    args = parser.parse_args()

    # fix random seed
    seed = 45
    np.random.seed(seed)

    # preprocess voc dataset
    data_prefix = './data/VOCdevkit/VOC2007/'
    save_path_trainval = preprocess_dataset(data_prefix)    

    # generate noisy annotations
    data_infos = load_annotations(save_path_trainval)
    box_noise_level = args.box_noise_level
    noisy_data_infos = gen_noisy_gt(data_infos, box_noise_level)

    # save noisy anno 
    anno_prefix = './data/VOCdevkit/VOC2007/noisy_pkl/'
    os.makedirs(anno_prefix, exist_ok=True)
    ann_name = save_path_trainval.split('/')[-1].split('.')[0]
    out_ann_file = '{}{}_noise-r{:.1f}.pkl'.format(anno_prefix, ann_name, box_noise_level)
    print('\nsave to {}'.format(out_ann_file))
    save_annotations(noisy_data_infos, out_ann_file)
