#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataset import Dataset
import numpy as np


class Estimator:
    def __init__(self, dataset, task):
        self.dataset = dataset
        self.task = task

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
        total_iou = []
        fp_detection_counter = 0
        fn_detection_counter = 0
        for each_prediction in self.dataset.predictions:
            pred_id = each_prediction['image_id']
            this_image_gt = [item for item in self.dataset.annotations
                             if item['image_id'] == pred_id]
            detection_results = each_prediction['detection_results']
            human_detections = [x[0] for x in detection_results]

            tmp_results = []
            gt_size = len(this_image_gt)
            pred_size = len(human_detections)
            for each_gt in this_image_gt:
                for each_pred in human_detections:
                    tmp_results.append(Estimator.get_iou(each_pred, each_gt['bbox']))
            tmp_sorted = np.sort(tmp_results)[::-1]

            if gt_size != 0:
                if pred_size != 0:

                    if gt_size == pred_size:
                        result = np.mean(tmp_sorted[:gt_size])
                    elif gt_size > pred_size:
                        result = np.mean(tmp_sorted[:gt_size])
                        fn_detection_counter += (gt_size - pred_size)
                    else:
                        result = np.mean(tmp_sorted)
                        fp_detection_counter += (pred_size - gt_size)

                    total_iou.append(result)
        return np.mean(total_iou), fp_detection_counter, fn_detection_counter