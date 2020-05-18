from cvints import Dataset
from cvints.processing_results import ProcessingResults
from cvints.utils.exceptions import CvintsException
from cvints.utils.utils import MS_COCO_CATEGORIES_DICT
import numpy as np
from collections import defaultdict


def get_iou(bbox1, bbox2):
    """

    Parameters
    ----------
    bbox1 : list
    bbox2 : list

    Notes
    -----
    bbox[0] = x1
    bbox[1] = y1
    bbox[2] = width
    bbox[3] = height
    """

    bbox1_x1 = bbox1[0]
    bbox1_y1 = bbox1[1]
    bbox1_x2 = bbox1[0] + bbox1[2]
    bbox1_y2 = bbox1[1] + bbox1[3]

    bbox2_x1 = bbox2[0]
    bbox2_y1 = bbox2[1]
    bbox2_x2 = bbox2[0] + bbox2[2]
    bbox2_y2 = bbox2[1] + bbox2[3]

    ixmin = max(bbox1_x1, bbox2_x1)
    ixmax = min(bbox1_x2, bbox2_x2)
    iymin = max(bbox1_y1, bbox2_y1)
    iymax = min(bbox1_y2, bbox2_y2)

    iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
    ih = np.maximum(iymax - iymin + 1.0, 0.0)

    inters = iw * ih

    uni = (
            (bbox1_x2 - bbox1_x1 + 1.0) * (bbox1_y2 - bbox1_y1 + 1.0)
            + (bbox2_x2 - bbox2_x1 + 1.0) * (bbox2_y2 - bbox2_y1 + 1.0)
            - inters
    )

    iou = inters / uni
    return iou


class Metrics:
    POSSIBLE_METRICS = ['Jaccard index', 'Precision', 'Recall', 'Miss rate']

    def __init__(self, ground_truth, predictions):
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

    def set_metrics(self, *args):
        for each_metrics_candidate in args:
            if each_metrics_candidate in self.POSSIBLE_METRICS:
                self.items.append(each_metrics_candidate)
            else:
                raise CvintsException('Wrong metrics name in setter')

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
                                    this_cat_iou_candidate.append(get_iou(each_gt_bbox, each_detection_bbox))

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

                # elif each_metric == 'Precision':

        else:
            raise CvintsException('No metrics to calculate')

        return results
