from .dataset import Dataset
from cvints.processing_results import ProcessingResults
from cvints.utils.exceptions import CvintsException
import cvints.utils.utils as cvints_utils
import numpy as np
from collections import defaultdict


class Metrics:
    POSSIBLE_METRICS = ['Jaccard index', 'Precision', 'Recall', 'Miss rate']

    def __init__(self, ground_truth, predictions, iou_threshold=0.5):
        """
        Parameters
        ----------
        ground_truth : Dataset
            Ground truth data wrapped in Dataset or Dataset-inherited class

        predictions : ProcessingResults
            Model predictions wrapped in ProcessingResults or ProcessingResults-inherited class

        """
        self.items = []
        self.ground_truth = ground_truth
        self.predictions = predictions
        self.iou_threshold = iou_threshold

    def set_metrics(self, *args):
        for each_metrics_candidate in args:
            if each_metrics_candidate in self.POSSIBLE_METRICS:
                self.items.append(each_metrics_candidate)
            else:
                raise CvintsException('Wrong metrics name in setter')

    def set_parameters(self, *args, **kwargs):
        pass

    def calculate(self):
        results = defaultdict(dict)
        if len(self.items) > 0:
            for each_metric in self.items:
                if each_metric == 'Jaccard index':
                    jaccard_index_by_categories = defaultdict(list)
                    mean_jaccard_index = 0
                    for each_filename in self.ground_truth.filenames:
                        image_predictions = self.predictions.get_results_by_filename(each_filename)
                        image_gt = self.ground_truth.get_image_annotations_by_filename(each_filename)
                        # get all categories from image_gt
                        object_categories = image_gt.keys()
                        for each_category in object_categories:
                            this_cat_iou = None

                            this_cat_gt = image_gt[each_category]  # list of bboxes
                            this_cat_detections = image_predictions[each_category]  # list of tuples (bbox, score)
                            this_cat_detections_bboxes = [x[0] for x in this_cat_detections]

                            this_cat_gt_number = len(this_cat_gt)
                            this_cat_detections_number = len(this_cat_detections_bboxes)

                            this_cat_iou_candidate = []
                            for each_gt_bbox in this_cat_gt:
                                for each_detection_bbox in this_cat_detections_bboxes:
                                    this_cat_iou_candidate.append(cvints_utils.get_iou(each_gt_bbox, each_detection_bbox))

                            this_cat_iou_candidate_sorted = np.sort(this_cat_iou_candidate)[::-1]

                            if this_cat_gt_number == this_cat_detections_number:
                                this_cat_iou = this_cat_iou_candidate_sorted[:this_cat_gt_number]
                            elif this_cat_gt_number > this_cat_detections_number:
                                this_cat_iou = this_cat_iou_candidate_sorted[:this_cat_detections_number]
                                this_cat_iou = np.pad(this_cat_iou,
                                                      (0, this_cat_gt_number-this_cat_detections_number),
                                                      'constant',
                                                      constant_values=(0, 0))

                            elif this_cat_gt_number < this_cat_detections_number:
                                this_cat_iou = this_cat_iou_candidate_sorted[:this_cat_gt_number]
                                this_cat_iou = np.pad(this_cat_iou,
                                                      (0, this_cat_detections_number-this_cat_gt_number),
                                                      'constant',
                                                      constant_values=(0, 0))

                            jaccard_index_by_categories[each_category] += list(this_cat_iou)
                    jaccard_index_values = []
                    for each in jaccard_index_by_categories.values():
                        jaccard_index_values += each

                    mean_jaccard_index = np.mean(jaccard_index_values)
                    results['Jaccard_index'] = {'Jaccard_index_by_categories': jaccard_index_by_categories,
                                                'Mean_jaccard_index': mean_jaccard_index}

                elif each_metric == 'Precision':
                    precision_calculation_by_categories = defaultdict(list)
                    precision_by_categories = defaultdict(float)
                    mean_precision = 0
                    jaccard_index_by_categories = defaultdict(list)
                    mean_jaccard_index = 0
                    for each_filename in self.ground_truth.filenames:
                        image_predictions = self.predictions.get_results_by_filename(each_filename)
                        image_gt = self.ground_truth.get_image_annotations_by_filename(each_filename)
                        # get all categories from image_gt
                        object_categories = image_gt.keys()
                        for each_category in object_categories:
                            this_cat_iou = None

                            this_cat_gt = image_gt[each_category]  # list of bboxes
                            this_cat_detections = image_predictions[each_category]  # list of tuples (bbox, score)
                            this_cat_detections_bboxes = [x[0] for x in this_cat_detections]

                            this_cat_gt_number = len(this_cat_gt)
                            this_cat_detections_number = len(this_cat_detections_bboxes)

                            this_cat_iou_candidate = []
                            for each_gt_bbox in this_cat_gt:
                                for each_detection_bbox in this_cat_detections_bboxes:
                                    this_cat_iou_candidate.append(cvints_utils.get_iou(each_gt_bbox, each_detection_bbox))

                            this_cat_iou_candidate_sorted = np.sort(this_cat_iou_candidate)[::-1]

                            if this_cat_gt_number == this_cat_detections_number:
                                this_cat_iou = this_cat_iou_candidate_sorted[:this_cat_gt_number]
                            elif this_cat_gt_number > this_cat_detections_number:
                                this_cat_iou = this_cat_iou_candidate_sorted[:this_cat_detections_number]
                                this_cat_iou = np.pad(this_cat_iou,
                                                      (0, this_cat_gt_number - this_cat_detections_number),
                                                      'constant',
                                                      constant_values=(0, 0))

                            elif this_cat_gt_number < this_cat_detections_number:
                                this_cat_iou = this_cat_iou_candidate_sorted[:this_cat_gt_number]
                                this_cat_iou = np.pad(this_cat_iou,
                                                      (0, this_cat_detections_number - this_cat_gt_number),
                                                      'constant',
                                                      constant_values=(0, 0))

                            this_cat_tp_number = len([x for x in this_cat_iou if x >= self.iou_threshold])
                            if this_cat_detections_number != 0:
                                this_cat_precision = this_cat_tp_number / this_cat_detections_number
                            else:
                                this_cat_precision = 0
                            precision_calculation_by_categories[each_category].append(this_cat_precision)
                    for each_category in precision_calculation_by_categories.keys():
                        precision_by_categories[each_category] = np.mean(precision_calculation_by_categories[each_category])
                    print(precision_by_categories)

                elif each_metric == 'Recall':
                    recall_calculation_by_categories = defaultdict(list)
                    recall_by_categories = defaultdict(float)
                    mean_recall = 0
                    jaccard_index_by_categories = defaultdict(list)
                    mean_jaccard_index = 0
                    for each_filename in self.ground_truth.filenames:
                        image_predictions = self.predictions.get_results_by_filename(each_filename)
                        image_gt = self.ground_truth.get_image_annotations_by_filename(each_filename)
                        # get all categories from image_gt
                        object_categories = image_gt.keys()
                        for each_category in object_categories:
                            this_cat_iou = None

                            this_cat_gt = image_gt[each_category]  # list of bboxes
                            this_cat_detections = image_predictions[each_category]  # list of tuples (bbox, score)
                            this_cat_detections_bboxes = [x[0] for x in this_cat_detections]

                            this_cat_gt_number = len(this_cat_gt)
                            this_cat_detections_number = len(this_cat_detections_bboxes)

                            this_cat_iou_candidate = []
                            for each_gt_bbox in this_cat_gt:
                                for each_detection_bbox in this_cat_detections_bboxes:
                                    this_cat_iou_candidate.append(cvints_utils.get_iou(each_gt_bbox, each_detection_bbox))

                            this_cat_iou_candidate_sorted = np.sort(this_cat_iou_candidate)[::-1]

                            if this_cat_gt_number == this_cat_detections_number:
                                this_cat_iou = this_cat_iou_candidate_sorted[:this_cat_gt_number]
                            elif this_cat_gt_number > this_cat_detections_number:
                                this_cat_iou = this_cat_iou_candidate_sorted[:this_cat_detections_number]
                                this_cat_iou = np.pad(this_cat_iou,
                                                      (0, this_cat_gt_number - this_cat_detections_number),
                                                      'constant',
                                                      constant_values=(0, 0))

                            elif this_cat_gt_number < this_cat_detections_number:
                                this_cat_iou = this_cat_iou_candidate_sorted[:this_cat_gt_number]
                                this_cat_iou = np.pad(this_cat_iou,
                                                      (0, this_cat_detections_number - this_cat_gt_number),
                                                      'constant',
                                                      constant_values=(0, 0))

                            this_cat_tp_number = len([x for x in this_cat_iou if x >= self.iou_threshold])
                            if this_cat_detections_number != 0:
                                this_cat_recall = this_cat_tp_number / this_cat_gt_number
                            else:
                                this_cat_recall = 0
                            recall_calculation_by_categories[each_category].append(this_cat_recall)
                    for each_category in recall_calculation_by_categories.keys():
                        recall_by_categories[each_category] = np.mean(recall_calculation_by_categories[each_category])
                    print(recall_by_categories)

        else:
            raise CvintsException('No metrics to calculate')

        return results
