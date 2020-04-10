#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from cvints import Dataset
import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt


class Evaluator:
    def __init__(self, dataset):
        self.dataset = dataset


class ObjectDetectionEvaluator(Evaluator):
    def __init__(self, dataset):
        super(ObjectDetectionEvaluator, self).__init__(dataset)


class PersonDetectionEvaluator(Evaluator):
    def __init__(self, dataset):
        super(PersonDetectionEvaluator, self).__init__(dataset)

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

    def compute_mean_iou(self):
        total_iou = []
        fp_detection_counter = 0
        missed_counter = 0

        fp_per_number_of_person_on_image = defaultdict()

        for each_prediction in self.dataset.model_evaluation_results:
            mean_iou_of_this_image = 0
            pred_id = each_prediction['image_id']
            gt_bboxes_of_this_image = [item['bbox'] for item in self.dataset.annotations['annotations_info']
                             if item['image_id'] == pred_id]
            detection_results = each_prediction['detection_results']
            predicted_bboxes_of_this_image = [x[0] for x in detection_results]

            tmp_results = []
            number_of_gt_bboxes_of_this_image = len(gt_bboxes_of_this_image)
            number_of_predicted_bboxes_of_this_image = len(predicted_bboxes_of_this_image)
            for each_predicted_bbox in predicted_bboxes_of_this_image:
                for each_gt_bbox in gt_bboxes_of_this_image:
                    tmp_results.append(PersonDetectionEvaluator.get_iou(each_predicted_bbox, each_gt_bbox))
            tmp_sorted = np.sort(tmp_results)[::-1]  # reverse the ascending sorted list
            if number_of_gt_bboxes_of_this_image != 0:
                if number_of_predicted_bboxes_of_this_image != 0:

                    if number_of_gt_bboxes_of_this_image == number_of_predicted_bboxes_of_this_image:
                        mean_iou_of_this_image = np.mean(tmp_sorted[:number_of_gt_bboxes_of_this_image])
                    elif number_of_gt_bboxes_of_this_image > number_of_predicted_bboxes_of_this_image:
                        iou_to_estimate = tmp_sorted[:number_of_predicted_bboxes_of_this_image]
                        miss_size = number_of_gt_bboxes_of_this_image - number_of_predicted_bboxes_of_this_image
                        for _ in range(miss_size):
                            np.append(iou_to_estimate, [0])
                        mean_iou_of_this_image = np.mean(iou_to_estimate)
                        missed_counter += miss_size
                    else:
                        mean_iou_of_this_image = np.mean(tmp_sorted[:number_of_gt_bboxes_of_this_image])
                        fp_size = number_of_predicted_bboxes_of_this_image - number_of_gt_bboxes_of_this_image
                        fp_detection_counter += fp_size
                        if number_of_gt_bboxes_of_this_image not in fp_per_number_of_person_on_image.keys():
                            fp_per_number_of_person_on_image[number_of_gt_bboxes_of_this_image] = fp_size
                        else:
                            fp_per_number_of_person_on_image[number_of_gt_bboxes_of_this_image] += fp_size

            total_iou.append(mean_iou_of_this_image)

        person_per_image_values = fp_per_number_of_person_on_image.keys()
        fp_values = fp_per_number_of_person_on_image.values()

        plt.title('False positive errors No per No of persons on images')
        plt.xlabel('No of persons on image')
        plt.ylabel('False positive No')
        plt.bar(person_per_image_values, fp_values)
        plt.show()
        print('Mean Iou = {:.2f}'.format(np.mean(total_iou)))

        return None
