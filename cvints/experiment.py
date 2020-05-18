#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from cvints.utils.utils import COCO_INSTANCE_CATEGORY_NAMES
import cvints.utils.utils as cvints_utils
import cvints.visialization as cvints_vis
from cvints.metrics import Metrics


class Tasks:
    CLASSIFICATION = 'Classification'
    SEMANTIC_SEGMENTATION = 'Semantic Segmentation'
    OBJECT_DETECTION = 'Object Detection'
    PERSON_KEYPOINT_DETECTION = 'Person Keypoint Detection'


class Experiment:
    """
        Description of intercommunication data and models with some results in the end
    """
    def __init__(self, task, model, dataset, results=None, metrics=None):
        self.task = task
        self.model = model
        self.dataset = dataset
        self.results = results
        self._metrics = metrics

    @property
    def metrics(self):
        return self._metrics

    @metrics.setter
    def metrics(self, value):
        if isinstance(value, Metrics):
            self._metrics = value
        elif isinstance(value, list):
            self._metrics = Metrics(self.dataset, self.results)
            self._metrics.set_metrics(*value)

    def describe(self):
        print('Number of images: {}'.format(len(self.dataset)))
        print('Model config:', end=' ')
        print(self.model.config)

    def filer_results_by_scores(self, scores_threshold=0.5):
        self.results = cvints_utils.low_scores_filter(self.results, scores_threshold)

    def apply_nms(self, nms_threshold=0.75):

        for i in range(len(self.results)):
            image_id = self.results.results[i]['image_id']
            image_filename = self.dataset.get_filename_by_id(image_id)
            print(image_filename)
            detections = self.results.results[i]['detections']
            # categories = each_result['labels']
            # bboxes = each_result['boxes']
            # scores = each_result['scores']
            # unique_categories = np.unique(each_result['labels'])
            unique_categories = detections.keys()
            for each_category in unique_categories:
                print('Processing of {}'.format(COCO_INSTANCE_CATEGORY_NAMES[int(each_category)]))

                # indices = np.where(categories == each_category)
                this_cat_boxes = [x[0] for x in detections[each_category]]
                this_cat_scores = [x[1] for x in detections[each_category]]
                print('{} items in the image'.format(len(this_cat_boxes)))
                if len(this_cat_boxes) > 1:
                    new_boxes, new_scores = cvints_utils.non_max_suppression(this_cat_boxes, this_cat_scores, nms_threshold)
                    self.results.results[i]['detections'][each_category] = [(b, s) for b, s in zip(new_boxes, new_scores)]

    def show_images(self, with_annotations=False):
        """
        Show images with predictions


        """
        for each_image_info in self.dataset.annotations['images_info']:
            file_name = each_image_info['file_name']
            image_id = each_image_info['id']
            this_image_prediction = self.results.get_results_by_image_id(image_id)
            image_path = self.dataset.path_to_images + file_name
            image = cvints_vis.open_image(image_path)
            image = cvints_vis.put_annotations_to_image(image, this_image_prediction)
            cvints_vis.show_image(image)

    def run(self):
        """
        Take a model, feed the dataset to the model and get the results

        :return:
        """
        self.filer_results_by_scores()
        self.apply_nms()
