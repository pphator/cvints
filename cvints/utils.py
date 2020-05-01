import numpy as np


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
                           12: u'stop sign',
                           13: u'parking meter',
                           14: u'bench',
                           15: u'bird',
                           16: u'cat',
                           17: u'dog',
                           18: u'horse',
                           19: u'sheep',
                           20: u'cow',
                           21: u'elephant',
                           22: u'bear',
                           23: u'zebra',
                           24: u'giraffe',
                           25: u'backpack',
                           26: u'umbrella',
                           27: u'handbag',
                           28: u'tie',
                           29: u'suitcase',
                           30: u'frisbee',
                           31: u'skis',
                           32: u'snowboard',
                           33: u'sports ball',
                           34: u'kite',
                           35: u'baseball bat',
                           36: u'baseball glove',
                           37: u'skateboard',
                           38: u'surfboard',
                           39: u'tennis racket',
                           40: u'bottle',
                           41: u'wine glass',
                           42: u'cup',
                           43: u'fork',
                           44: u'knife',
                           45: u'spoon',
                           46: u'bowl',
                           47: u'banana',
                           48: u'apple',
                           49: u'sandwich',
                           50: u'orange',
                           51: u'broccoli',
                           52: u'carrot',
                           53: u'hot dog',
                           54: u'pizza',
                           55: u'donut',
                           56: u'cake',
                           57: u'chair',
                           58: u'couch',
                           59: u'potted plant',
                           60: u'bed',
                           61: u'dining table',
                           62: u'toilet',
                           63: u'tv',
                           64: u'laptop',
                           65: u'mouse',
                           66: u'remote',
                           67: u'keyboard',
                           68: u'cell phone',
                           69: u'microwave',
                           70: u'oven',
                           71: u'toaster',
                           72: u'sink',
                           73: u'refrigerator',
                           74: u'book',
                           75: u'clock',
                           76: u'vase',
                           77: u'scissors',
                           78: u'teddy bear',
                           79: u'hair drier',
                           80: u'toothbrush'}


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


def low_scores_filter(processing_results, threshold=DEFAULT_SCORES_THRESHOLD):
    """
    Return detection results (bboxes, scores, labels) with scores higher than threshold

    Parameters
    ----------
    processing_results : ProcessingResults

    Returns
    -------
    new_results : ProcessingResults
    """

    new_results = [x for x in processing_results.results if x['score'] > threshold]
    processing_results.set_results(new_results)
    return processing_results


def non_max_suppression(bboxes, scores, threshold=0.5):
    # remove all boxes with scores lower than 0.5
    sort_index = np.argsort(scores)[::-1]

    # create empty lists to return
    bboxes_to_return = []
    scores_to_return = []

    # put there elements with the first index from sort_index
    bboxes_to_return.append(bboxes[sort_index[0]])
    scores_to_return.append(scores[sort_index[0]])

    # remove from bboxes and scores items with index sort_index[0]
    np.delete(bboxes, sort_index[0])
    np.delete(scores, sort_index[0])

    # remove from sort_index the first element
    if sort_index.size > 1:
        sort_index = sort_index[1:]




    # for ind in range(len(sort_index)):
    #     current_index = sort_index[ind]
    #     bbox = bboxes[current_index]


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
