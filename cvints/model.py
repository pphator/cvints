#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import json
from cvints.utils import COCO_INSTANCE_CATEGORY_NAMES
from cvints import utils as cvints_utils
# from cvints import ProcessingResults

from collections import defaultdict


IOU_THRESHOLD_DEFAULT = 0.5


class BaseModel:
    """ Base class for models which results we want to explore with this lib

    """

    def __init__(self, config=None):
        self.name = None
        self.config = config
        self.dataset = None
        self.path_to_processing_results = None
        self.processing_results = None
        self.evaluation_results = None
        self.input_size = None
        if self.config is not None:
            self._load_config()

    def create_config(self, name, *args, **kwargs):
        raise NotImplementedError('load_processing_results should be implemented in child classes')

    def _load_config(self):
        raise NotImplementedError('load_processing_results should be implemented in child classes')

    def evaluate(self):
        pass

    def load_processing_results(self):
        raise NotImplementedError('load_processing_results should be implemented in child classes')

    def _check_processing_results(self, processing_results_info):
        # TODO: compare results file with dataset
        pass

    def set_processing_results(self, processing_results):
        self.processing_results = processing_results


class ObjectDetectionModel(BaseModel):
    def __init__(self, config=None):
        super(ObjectDetectionModel, self).__init__(config)

    def create_config(self, name, input_size=(224, 224), nms=False):
        config = {'name': name,
                  'input_size': input_size,
                  'NMS': nms}
        self.config = config

    def _load_config(self):
        self.name = self.config['name']
        self.input_size = self.config['input_size']
        self.NMS = self.config['NMS']

    def _check_processing_results(self, processing_results_info):
        images_number_in_processing_results = processing_results_info['info']['images_number']
        assert images_number_in_processing_results == len(self.dataset)

    def load_processing_results(self):
        # load whole file
        if self.path_to_processing_results:
            with open(self.path_to_processing_results, "r") as f:
                processing_results_info = json.load(f)
        self._check_processing_results(processing_results_info)
        processing_results = processing_results_info['results']
        self.processing_results = processing_results

    @staticmethod
    def get_iou(pred_box, gt_box):
        ixmin = max(pred_box[0], gt_box[0])
        ixmax = min(pred_box[2], gt_box[2])
        iymin = max(pred_box[1], gt_box[1])
        iymax = min(pred_box[3], gt_box[3])

        iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
        ih = np.maximum(iymax - iymin + 1.0, 0.0)

        inters = iw * ih

        uni = (
            (pred_box[2] - pred_box[0] + 1.0) * (pred_box[3] - pred_box[1] + 1.0)
            + (gt_box[2] - gt_box[0] + 1.0) * (gt_box[3] - gt_box[1] + 1.0)
            - inters
        )

        iou = inters / uni

        return iou

    def set_processing_results(self, processing_results):
        """
            Adaptation raw processing result to this model

        Parameters
        ----------
        processing_results : ProcessingResults

        Returns
        -------
        """
        # each bbox we should multiply by  raw_image_hight/model_image_size
        results = []
        for each_result in processing_results.results:
            image_id = each_result['image_id']
            width, height = each_result['image_size']
            width_coef = width / self.config['input_size'][0]
            heigh_coef = height / self.config['input_size'][1]
            post_proc_results = defaultdict(list)
            for each_label in each_result['detections'].keys():
                post_proc_detections = []
                for each_detection in each_result['detections'][str(each_label)]:
                    raw_bbox = each_detection[0]
                    score = each_detection[1]
                    post_proc_bbox = [raw_bbox[0]*width_coef,
                                      raw_bbox[1]*heigh_coef,
                                      raw_bbox[2]*width_coef - raw_bbox[0]*width_coef,
                                      raw_bbox[3]*heigh_coef - raw_bbox[1]*heigh_coef]
                    post_proc_detections.append((post_proc_bbox, score))
                post_proc_results[each_label] = post_proc_detections
            results.append({'image_id': image_id,
                            'image_size': (width, height),
                            'detections': post_proc_results})

        processing_results.results = results
        self.processing_results = processing_results
        return self.processing_results

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
            pred_image_id = each_prediction["image_id"]
            gt_bboxes_of_this_image = [
                item["bbox"]
                for item in self.dataset.annotations["annotations_info"]
                if item["image_id"] == pred_image_id
            ]
            if len(gt_bboxes_of_this_image) == 0:
                continue
            detection_results = each_prediction["detection_results"]
            predicted_bboxes_of_this_image = [x[0] for x in detection_results]
            number_of_gt_bboxes_of_this_image = len(gt_bboxes_of_this_image)
            number_of_predicted_bboxes_of_this_image = len(
                predicted_bboxes_of_this_image
            )

            # calculate IoU for all combinations of predicted and ground-truth bboxes
            this_prediction_iou_list = []
            for each_predicted_bbox in predicted_bboxes_of_this_image:
                this_predicted_bbox_iou_list = []
                for each_gt_bbox in gt_bboxes_of_this_image:
                    this_predicted_bbox_iou_list.append(
                        ObjectDetectionModel.get_iou(each_predicted_bbox, each_gt_bbox)
                    )

                max_iou_for_prediction = np.max(this_predicted_bbox_iou_list)
                this_prediction_iou_list.append(max_iou_for_prediction)

            # select IoU >= iou_threshold
            true_positives_this_prediction = len(
                [x for x in this_prediction_iou_list if x >= iou_threshold]
            )
            false_positives_this_prediction = len(
                [x for x in this_prediction_iou_list if x < iou_threshold]
            )
            number_of_gt_bboxes += number_of_gt_bboxes_of_this_image
            number_of_predicted_bboxes += number_of_predicted_bboxes_of_this_image
            true_positives += true_positives_this_prediction
            results[pred_image_id] = (
                true_positives_this_prediction,
                false_positives_this_prediction,
                np.mean(this_prediction_iou_list),
            )

        recall = true_positives / number_of_gt_bboxes
        precision = true_positives / number_of_predicted_bboxes
        miss_rate = 1 - recall
        self.evaluation_results = results
        return precision, recall, miss_rate, results
