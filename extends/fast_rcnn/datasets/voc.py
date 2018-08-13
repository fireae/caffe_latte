from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
p = os.path.abspath(__file__)
base, _ = os.path.split(p)
print(base)
sys.path.insert(0, base+'/..')

from six.moves import xrange

import os
from datasets.imdb import imdb
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
from  utils.bbox import *
import subprocess
from utils.config import cfg
import json_tricks
import uuid
from voc_eval import voc_eval
from utils.config import *
print(cfg)

class VOC(imdb):
    def __init__(self, devkit_path = None):
        self._devkit_path = self._get_default_path() if devkit_path is None \
            else devkit_path
        self._data_path = self._devkit_path
        self._name = 'voc'
        self._classes = 'label_map.json'
        self._class_to_ind = dict(process_config('label_map.json'))
        self._image_ext = '.jpg'
        self._image_set = 'train'
        self._image_index = self._load_image_set_index()
        
        #Default to roidb handler
        self._roidb_handler = self.rpn_roidb
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        # PASCAL specific config options
        self.config = {
            'cleanup': True,
            'use_difficult': False,
            'use_salt': True,
            'rpn_file': None,
            'min_size': 2
        }


    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
            """
            Construct an image path from the image's "index" identifier.
            """
            image_path = os.path.join(self._data_path, 'JPEGImages',
                                    index + self._image_ext)
            assert os.path.exists(image_path), \
                    'Path does not exist: {}'.format(image_path)
            return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /ImageSets/Main/val.txt
        image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, 'VOCdevkit')

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.json')
        print(cache_file)
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as fid:
                roidb = json_tricks.load(fid)
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        gt_roidb = [self._load_pascal_annotation(index)
                    for index in self.image_index]
        print(gt_roidb)
        with open(cache_file, 'w') as fid:
            json_tricks.dump(gt_roidb, fid)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb

    def rpn_roidb(self):
        if self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = imdb.merge_roidb(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)
        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print('loading {}'.format(filename))
        assert os.path.exists(filename), \
            'rpn data not found at: {}'.format(filename)

        with open(filename, 'rb') as f:
            box_list = json.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)


    def _load_pascal_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        filename = os.path.join(self._data_path, 'Annotations', 
            index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')
        if not self.config['use_difficult']:
            # Exclude the samples labled as difficult
            non_diff_objs = [
                obj for obj in objs if int(obj.find('difficult').text) == 0]
            objs  = non_diff_objs
        num_objs = len(objs)

        # Text Quad
        boxes = np.zeros((num_objs, 8), dtype=np.uint32)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        
        # 'Seg' area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        im_size = tree.find('size')
        im_width = float(im_size.find('width').text)
        im_height = float(im_size.find('height').text)

        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            
            # x1 = np.minimum(np.maximum(float(bbox.find('x1').text), 0), im_width - 1)
            # y1 = np.minimum(np.maximum(float(bbox.find('y1').text), 0), im_height - 1)
            # x2 = np.minimum(np.maximum(float(bbox.find('x2').text), 0), im_width - 1)
            # y2 = np.minimum(np.maximum(float(bbox.find('y2').text), 0), im_height - 1)
            # x3 = np.minimum(np.maximum(float(bbox.find('x3').text), 0), im_width - 1)
            # y3 = np.minimum(np.maximum(float(bbox.find('y3').text), 0), im_height - 1)
            # x4 = np.minimum(np.maximum(float(bbox.find('x4').text), 0), im_width - 1)
            # y4 = np.minimum(np.maximum(float(bbox.find('y4').text), 0), im_height - 1)
            # boxes[ix, :] = [x1, y1, x2, y2, x3, y3, x4, y4]

            x1 = np.minimum(np.maximum(float(bbox.find('xmin').text), 0), im_width - 1)
            y1 = np.minimum(np.maximum(float(bbox.find('ymin').text), 0), im_height - 1)
            x2 = np.minimum(np.maximum(float(bbox.find('xmax').text), 0), im_width - 1)
            y2 = np.minimum(np.maximum(float(bbox.find('ymax').text), 0), im_height - 1)
            x3 = 0
            y3 = 0
            x4 = 0
            y4 = 0
            boxes[ix, :] = [x1, y1, x2, y2, x3, y3, x4, y4]

            cls = self._class_to_ind[obj.find('name').text.lower().strip()]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2- y1 + 1)
        
        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {
            'boxes': boxes,
            'gt_classes': gt_classes,
            'gt_overlap': overlaps,
            'flipped': False,
            'seg_areas': seg_areas}

    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt'] else 
            self._comp_id)
        return comp_id

    def _get_voc_results_file_template(self):
        # VOCdevkit/results/Main/<comp_id>_det_test_aeronplane.txt
        filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
        path = os.path.join(
            self._devkit_path,
            'results',
            'Main', filename)
        return path
    
    def _write_voc_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print('Writing {} VOC results file'.format(cls))
            filename = self._get_voc_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    for k in range(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f}\n'.
                            format(index, dets[k, -1], dets[k, 0]+1,
                                dets[k, 1]+1, dets[k, 2]+1, dets[k, 3]+1))

    def _do_python_eval(self, output_dir = 'output'):
        anno_path = os.path.join(self._devkit_path,
            'Annotations', '{:s}.xml')
        imageset_file = os.path.join(self._devkit_path,
            'ImageSets', 'Main', self._image_set + '.txt')
        cachedir = os.path.join(self._devkit_path, 'annotations_cache')
        aps = []
        recs = []
        precs = []
        # The Pascal VOC metric changed in 2010
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_voc_results_file_template().format(cls)
            rec, prec, ap = voc_eval(
                filename, anno_path, imageset_file, cls, cachedir, ovthresh=0.5)
            aps += [ap]
            recs += [rec[-1]]
            precs += [prec[-1]]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'w') as f:
                json.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('Mean REC = {:.4f}'.format(np.mean(recs)))
        print('Mean PREC = {:.4f}'.format(np.mean(precs)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')

    def evaluate_detections(self, all_boxes, output_dir):
        self._write_voc_results_file(all_boxes)
        self._do_python_eval(output_dir)
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_voc_results_file_template().format(cls)
                os.remove(filename)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

if __name__ == "__main__":
    voc = VOC('D:\\datasets\\VOCdevkit')
    print(voc.gt_roidb())