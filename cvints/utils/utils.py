import json
import numpy as np
from collections import defaultdict


DEFAULT_SCORES_THRESHOLD = 0.5


MS_COCO_CATEGORIES_DICT = {0: u'__background__',
                           1: u'person',
                           2: u'bicycle',
                           3: u'car',
                           4: u'motorcycle',
                           5: u'airplane',
                           6: u'bus',
                           7: u'train',
                           8: u'truck',
                           9: u'boat',
                           10: u'traffic light',
                           11: u'fire hydrant',
                           13: u'stop sign',
                           14: u'parking meter',
                           15: u'bench',
                           16: u'bird',
                           17: u'cat',
                           18: u'dog',
                           19: u'horse',
                           20: u'sheep',
                           21: u'cow',
                           22: u'elephant',
                           23: u'bear',
                           24: u'zebra',
                           25: u'giraffe',
                           27: u'backpack',
                           28: u'umbrella',
                           31: u'handbag',
                           32: u'tie',
                           33: u'suitcase',
                           34: u'frisbee',
                           35: u'skis',
                           36: u'snowboard',
                           37: u'sports ball',
                           38: u'kite',
                           39: u'baseball bat',
                           40: u'baseball glove',
                           41: u'skateboard',
                           42: u'surfboard',
                           43: u'tennis racket',
                           44: u'bottle',
                           46: u'wine glass',
                           47: u'cup',
                           48: u'fork',
                           49: u'knife',
                           50: u'spoon',
                           51: u'bowl',
                           52: u'banana',
                           53: u'apple',
                           54: u'sandwich',
                           55: u'orange',
                           56: u'broccoli',
                           57: u'carrot',
                           58: u'hot dog',
                           59: u'pizza',
                           60: u'donut',
                           61: u'cake',
                           62: u'chair',
                           63: u'couch',
                           64: u'potted plant',
                           65: u'bed',
                           67: u'dining table',
                           70: u'toilet',
                           72: u'tv',
                           73: u'laptop',
                           74: u'mouse',
                           75: u'remote',
                           76: u'keyboard',
                           77: u'cell phone',
                           78: u'microwave',
                           79: u'oven',
                           80: u'toaster',
                           81: u'sink',
                           82: u'refrigerator',
                           84: u'book',
                           85: u'clock',
                           86: u'vase',
                           87: u'scissors',
                           88: u'teddy bear',
                           89: u'hair drier',
                           90: u'toothbrush'}


COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

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


def low_scores_filter(processing_results, threshold=DEFAULT_SCORES_THRESHOLD):
    """
    Return detection results (bboxes, scores, labels) with scores higher than threshold

    Parameters
    ----------
    processing_results : ProcessingResults
    threshold : float

    Returns
    -------
    new_results : ProcessingResults
    """

    new_results = []

    for each_result in processing_results.results:
        image_id = each_result['image_id']
        image_filename = each_result['image_filename']
        width, height = each_result['image_size']

        post_proc_results = defaultdict(list)
        for each_label in each_result['detections'].keys():
            post_proc_detections = []
            for each_detection in each_result['detections'][str(each_label)]:
                bbox = each_detection[0]
                score = each_detection[1]
                if score >= threshold:
                    post_proc_detections.append((bbox, score))
            post_proc_results[each_label] = post_proc_detections
        new_results.append({'image_id': image_id,
                            'image_filename': image_filename,
                            'image_size': (width, height),
                            'detections': post_proc_results})

    processing_results.set_results(new_results)
    return processing_results


def non_max_suppression(bboxes, scores, iou_threshold=0.5):
    """
    This function could be applied to the bboxes and scores of common label

    Parameters
    ----------
    bboxes : array-like
    scores : array-like
    iou_threshold : float

    Returns
    -------
    """
    # sort bboxes by scores (from highest to lowest value)
    sort_index = np.argsort(scores)[::-1]

    # create empty lists to return
    bboxes_to_return = []
    scores_to_return = []

    # put there elements with the first index from sort_index
    bboxes_to_return.append(bboxes[sort_index[0]])
    scores_to_return.append(scores[sort_index[0]])

    while len(bboxes) > 0:
        # create the list of indices which will be deleted from sort_index by IoU
        indices_to_remove = []
        current_bbox = bboxes[sort_index[0]]

        # select bboxes which have IoU value with current bbox higher than threshold
        for each_index in sort_index[1:]:
            iou_val = get_iou(current_bbox, bboxes[each_index])
            if iou_val >= iou_threshold:
                indices_to_remove.append(each_index)

        # remove elements with indices_to_remove from bboxes and scores and from sort_index
        bboxes = np.delete(bboxes, indices_to_remove, axis=0)
        scores = np.delete(scores, indices_to_remove)

        # and remove current bbox from bboxes
        bboxes = np.delete(bboxes, sort_index[0], axis=0)
        scores = np.delete(scores, sort_index[0])

        # if there are some items in the bboxes and scores - prepare to next loot iteration
        if len(scores) > 1:
            sort_index = np.argsort(scores)[::-1]

            bboxes_to_return.append(bboxes[sort_index[0]])
            scores_to_return.append(scores[sort_index[0]])
        # if there is only one item - put in into bboxes to return
        elif len(scores) == 1:
            bboxes_to_return.append(bboxes[-1])
            scores_to_return.append(scores[-1])

            bboxes = np.delete(bboxes, -1, axis=0)
            scores = np.delete(scores, -1)

    return bboxes_to_return, scores_to_return


def non_max_suppression_slow(boxes, threshold=0.5):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list, add the index
        # value to the list of picked indexes, then initialize
        # the suppression list (i.e. indexes that will be deleted)
        # using the last index
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        suppress = [last]
        # loop over all indexes in the indexes list
        for pos in range(0, last):
            # grab the current index
            j = idxs[pos]
            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])
            # compute the width and height of the bounding box
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)
            # compute the ratio of overlap between the computed
            # bounding box and the bounding box in the area list
            overlap = float(w * h) / area[j]
            # if there is sufficient overlap, suppress the
            # current bounding box
            if overlap > threshold:
                suppress.append(pos)
        # delete all indexes from the index list that are in the
        # suppression list
        idxs = np.delete(idxs, suppress)
    # return only the bounding boxes that were picked
    return boxes[pick]


def bring_categories_to_COCO_values_for_instances_default(input_file_path, output_file_path):
    with open(input_file_path, "r") as f:
        annotations_data = json.load(f)

    # create map = old_id: new_id for each category by label
    cat_map = {}

    categories = [x['name'] for x in annotations_data['categories']]
    for each_category in categories:
        res = 0
        for k, v in MS_COCO_CATEGORIES_DICT.items():
            if v == each_category:
                res = k
        cat_map[next(x for x in annotations_data['categories'] if x['name'] == each_category)['id']] = res

    for i in range(len(annotations_data['categories'])):
        old_id = annotations_data['categories'][i]['id']
        annotations_data['categories'][i]['id'] = cat_map[old_id]

    for i in range(len(annotations_data['annotations'])):
        old_id = annotations_data['annotations'][i]['category_id']
        annotations_data['annotations'][i]['category_id'] = cat_map[old_id]

    with open(output_file_path, "w") as f:
        json.dump(annotations_data, f)
    print('Done.')


def bring_categories_to_COCO_values_for_labels_default(input_file_path, output_file_path):
    with open(input_file_path, "r") as f:
        labels_data = json.load(f)

    cat_map = {}
    categories = [x['name'] for x in labels_data['categories']]

    for each_category in categories:
        res = 0
        for k, v in MS_COCO_CATEGORIES_DICT.items():
            if v == each_category:
                res = k
        cat_map[next(x for x in labels_data['categories'] if x['name'] == each_category)['id']] = res

    for i in range(len(labels_data['categories'])):
        old_id = labels_data['categories'][i]['id']
        labels_data['categories'][i]['id'] = cat_map[old_id]

    with open(output_file_path, "w") as f:
        json.dump(labels_data, f)
    print('Done.')
