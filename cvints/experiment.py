#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from cvints.utils import MS_COCO_CATEGORIES_DICT, COCO_INSTANCE_CATEGORY_NAMES
import cvints.utils as cvints_utils


class Tasks:
    CLASSIFICATION = 'Classification'
    SEMANTIC_SEGMENTATION = 'Semantic Segmentation'
    OBJECT_DETECTION = 'Object Detection'
    PERSON_KEYPOINT_DETECTION = 'Person Keypoint Detection'


class Experiment:
    def __init__(self, task, model, dataset, results=None):
        self.task = task
        self.model = model
        self.dataset = dataset
        self.results = []

    def describe(self):
        print('Number of images: {}'.format(len(self.dataset)))
        print('Model config:', end=' ')
        print(self.model.config)

    def filer_results_by_scores(self, scores_threshold=0.5):

        self.results = cvints_utils.low_scores_filter()

    def apply_nms(self, nms_threshold=0.75):
        for each_result in self.results:
            categories = each_result['labels'].numpy()
            bboxes = each_result['boxes'].numpy()
            scores = each_result['scores'].numpy()
            unique_categories = np.unique(each_result['labels'].numpy())
            for each_category in unique_categories:
                print('Processing of {}'.format(COCO_INSTANCE_CATEGORY_NAMES[each_category]))
                indices = np.where(categories == each_category)
                this_cat_boxes = bboxes[indices]
                this_cat_scores = scores[indices]

                non_max_suppression(this_cat_boxes, this_cat_scores)



    def run(self):
        """
        Take a model, feed the dataset to the model and get the results

        :return:
        """
        self.filer_results_by_scores()

        pass