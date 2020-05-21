from .dataset import Dataset
from cvints.processing_results import ProcessingResults
from cvints.utils.exceptions import CvintsException
import cvints.utils.utils as cvints_utils
import numpy as np
from collections import defaultdict
from pprint import PrettyPrinter


class Metrics:
    """
    Attributes
    ----------
    items : list
        List of metrics to calculate


    """
    POSSIBLE_METRICS = ['Jaccard index', 'Precision', 'Recall', 'Miss rate', 'TP', 'FP', 'FN']

    def __init__(self, ground_truth, predictions, task, iou_threshold=0.5):
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
        self.task = task
        self.iou_threshold = iou_threshold
        self.result = defaultdict(defaultdict)

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
        overall_categories = []
        dataset_categories = self.ground_truth.get_dataset_objects_categories()
        if len(self.items) > 0:
            if self.task == 'Object Detection':
                jaccard_index_by_categories = defaultdict(float)
                jaccard_index_by_categories_calculation = defaultdict(list)
                fn_by_categories_calculations = defaultdict(list)
                fp_by_categories_calculations = defaultdict(list)
                tp_by_categories_calculations = defaultdict(list)

                fn_by_categories = defaultdict(int)
                fp_by_categories = defaultdict(int)
                tp_by_categories = defaultdict(int)

                precision_by_categories = defaultdict(float)
                recall_by_categories = defaultdict(float)
                for each_filename in self.ground_truth.filenames:
                    image_predictions = self.predictions.get_results_by_filename(each_filename)
                    image_gt = self.ground_truth.get_image_annotations_by_filename(each_filename)
                    # get all categories from image_gt
                    object_categories = list(set(list(image_gt.keys()) + list(image_predictions.keys())))
                    overall_categories += object_categories
                    for each_category in object_categories:
                        category_label = cvints_utils.MS_COCO_CATEGORIES_DICT[int(each_category)]
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
                            # this_cat_iou = np.pad(this_cat_iou,
                            #                       (0, this_cat_gt_number-this_cat_detections_number),
                            #                       'constant',
                            #                       constant_values=(0, 0))
                            fn_by_categories_calculations[each_category].append(this_cat_gt_number-this_cat_detections_number)

                        elif this_cat_gt_number < this_cat_detections_number:
                            this_cat_iou = this_cat_iou_candidate_sorted[:this_cat_gt_number]
                            # this_cat_iou = np.pad(this_cat_iou,
                            #                       (0, this_cat_detections_number-this_cat_gt_number),
                            #                       'constant',
                            #                       constant_values=(0, 0))
                            fp_by_categories_calculations[each_category].append(this_cat_detections_number-this_cat_gt_number)

                        jaccard_index_by_categories_calculation[each_category] += list(this_cat_iou)

                for each_category in dataset_categories:
                    category_label = cvints_utils.MS_COCO_CATEGORIES_DICT[int(each_category)]
                    this_cat_fp_number = len([x for x in jaccard_index_by_categories_calculation[each_category] if x < self.iou_threshold])
                    this_cat_tp_number = len([x for x in jaccard_index_by_categories_calculation[each_category] if x >= self.iou_threshold])
                    fp_by_categories_calculations[each_category].append(this_cat_fp_number)
                    tp_by_categories_calculations[each_category].append(this_cat_tp_number)

                    jaccard_index_by_categories[category_label] = round(np.mean(jaccard_index_by_categories_calculation[each_category]), 2)
                    tp_by_categories[category_label] = np.sum(tp_by_categories_calculations[each_category])
                    fp_by_categories[category_label] = np.sum(fp_by_categories_calculations[each_category])
                    fn_by_categories[category_label] = np.sum(fn_by_categories_calculations[each_category])
                    if (tp_by_categories[category_label] + fp_by_categories[category_label]) != 0:
                        precision_by_categories[category_label] = tp_by_categories[category_label] / (tp_by_categories[category_label] + fp_by_categories[category_label])
                    else:
                        precision_by_categories[category_label] = 0
                    if (tp_by_categories[category_label] + fn_by_categories[category_label]) != 0:
                        recall_by_categories[category_label] = tp_by_categories[category_label] / (tp_by_categories[category_label] + fn_by_categories[category_label])
                    else:
                        recall_by_categories[category_label] = 0

                if 'Jaccard index' in self.items:
                    self.result['Jaccard_index'] = jaccard_index_by_categories
                if 'TP' in self.items:
                    self.result['TP'] = tp_by_categories
                if 'FP' in self.items:
                    self.result['FP'] = fp_by_categories
                if 'FN' in self.items:
                    self.result['FN'] = fn_by_categories
                if 'Precision' in self.items:
                    self.result['Precision'] = precision_by_categories
                if 'Recall' in self.items:
                    self.result['Recall'] = recall_by_categories

        else:
            raise CvintsException('No metrics to calculate')

        return self.result

    def print_values(self):
        pprinter = PrettyPrinter()
        if self.result:
            for each_metric in self.result.keys():
                print('Metric is {}'.format(each_metric))
                pprinter.pprint(dict(self.result[each_metric]))
