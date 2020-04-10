#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import json


IOU_THRESHOLD_DEFAULT = 0.5


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
        self.detection_results = None

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
        self.detection_results = [x for x in detection_results if x['image_id'] in loaded_test_images_ids]

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

    def evaluate(self, iou_threshold=IOU_THRESHOLD_DEFAULT):
        """ Compute metrics for detection results

            Parameters
            ----------
            iou_threshold : float
                The value to say that prediction is correct or incorrect

            Returns
            ----------
            precision : float
            recall : float
            miss_rate : float
            results : dict
        """
        true_positives = 0
        number_of_gt_bboxes = 0
        number_of_predicted_bboxes = 0
        results = {}
        for each_prediction in self.detection_results:
            pred_image_id = each_prediction['image_id']
            gt_bboxes_of_this_image = [item['bbox'] for item in self.test_dataset.annotations['annotations_info']
                                       if item['image_id'] == pred_image_id]
            detection_results = each_prediction['detection_results']
            predicted_bboxes_of_this_image = [x[0] for x in detection_results]
            number_of_gt_bboxes_of_this_image = len(gt_bboxes_of_this_image)
            number_of_predicted_bboxes_of_this_image = len(predicted_bboxes_of_this_image)

            # calculate IoU for all combinations of predicted and ground-truth bboxes
            this_prediction_iou_list = []
            for each_predicted_bbox in predicted_bboxes_of_this_image:
                this_predicted_bbox_iou_list = []
                for each_gt_bbox in gt_bboxes_of_this_image:
                    this_predicted_bbox_iou_list.append(ObjectDetectionModel.get_iou(each_predicted_bbox, each_gt_bbox))
                max_iou_for_prediction = np.max(this_predicted_bbox_iou_list)
                this_prediction_iou_list.append(max_iou_for_prediction)

            # select IoU >= iou_threshold
            true_positives_this_prediction = len([x for x in this_prediction_iou_list if x >= iou_threshold])
            false_positives_this_prediction = len([x for x in this_prediction_iou_list if x < iou_threshold])
            number_of_gt_bboxes += number_of_gt_bboxes_of_this_image
            number_of_predicted_bboxes += number_of_predicted_bboxes_of_this_image
            true_positives += true_positives_this_prediction
            results[pred_image_id] = (true_positives_this_prediction, false_positives_this_prediction, np.mean(this_prediction_iou_list))

        recall = true_positives / number_of_gt_bboxes
        precision = true_positives / number_of_predicted_bboxes
        miss_rate = 1 - recall

        return precision, recall, miss_rate, results

