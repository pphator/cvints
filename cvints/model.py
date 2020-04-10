#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import json


class Model:
    """ Base class for models which results we want to explore with this lib

    """
    def __init__(self, name='default', train_dataset=None, val_dataset=None, test_dataset=None,
                 path_to_detection_results_file=None):
        self.name = name
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        self.path_to_detection_results_file = path_to_detection_results_file
        self.model_evaluation_results = None

    def evaluate(self):
        pass


class ObjectDetectionModel(Model):
    def __init__(self, name='default', train_dataset=None, val_dataset=None, test_dataset=None,
                 path_to_detection_results_file=None):
        super(ObjectDetectionModel, self).__init__(name, train_dataset, val_dataset, test_dataset,
                                                   path_to_detection_results_file)

    def load_test_dataset(self, test_dataset):
        pass

    def load_detection_results(self):
        # load whole file
        if self.path_to_detection_results_file:
            with open(self.path_to_detection_results_file, 'r') as f:
                detection_results = json.load(f)
        # select images from loaded part of test dataset

        loaded_test_images_ids = self.test_dataset.get_images_ids()
        self.model_evaluation_results = [x for x in loaded_test_images_ids if x['image_id'] in loaded_test_images_ids]
        
    @staticmethod
    def get_iou(pred_box, gt_box):
        ixmin = max(pred_box[0], gt_box[0])
        ixmax = min(pred_box[2], gt_box[2])
        iymin = max(pred_box[1], gt_box[1])
        iymax = min(pred_box[3], gt_box[3])

        iw = np.maximum(ixmax - ixmin + 1., 0.)
        ih = np.maximum(iymax - iymin + 1., 0.)

        inters = iw * ih

        uni = ((pred_box[2] - pred_box[0] + 1.) * (pred_box[3] - pred_box[1] + 1.) +
               (gt_box[2] - gt_box[0] + 1.) * (gt_box[3] - gt_box[1] + 1.) -
               inters)

        iou = inters / uni

        return iou

    def evaluate(self):



        pass

