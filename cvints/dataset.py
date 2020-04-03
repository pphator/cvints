#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import numpy as np
from random import sample
import cv2
from matplotlib import pyplot as plt


GREEN = (0, 255, 0)


class Dataset:
    def __init__(self, path_to_data, path_to_annotations_file, is_sampled=False, samples_number=5):
        self.path_to_data = path_to_data
        self.path_to_annotations_file = path_to_annotations_file
        self.is_sampled = is_sampled
        self.predictions = None
        if self.path_to_annotations_file:
            with open(self.path_to_annotations_file, 'r') as f:
                annotations_data = json.load(f)

        if is_sampled:
            sampled_images_info = sample(annotations_data['images'], samples_number)
            sampled_images_ids = [x['id'] for x in sampled_images_info]
            sampled_annotations_data = [x for x in annotations_data['annotations'] if
                                        x['image_id'] in sampled_images_ids]
            self.annotations = {'images_info': sampled_images_info, 'annotation_info': sampled_annotations_data}

        else:
            images_info = annotations_data['images']
            annotations_data = annotations_data['annotations']
            self.annotations = {'images_info': images_info, 'annotation_info': annotations_data}

    def draw_gt_bboxes(self, separately=False, bbox_line_width=3):
        shown = False
        images_info = self.annotations['images_info']
        annotation_info = self.annotations['annotation_info']
        for each_image in images_info:
            image_path = self.path_to_data + '/' + each_image['file_name']
            image_id = each_image['id']
            annotations = [x for x in annotation_info if x['image_id'] == image_id]
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            for each_gt in annotations:
                cv2.rectangle(img, (each_gt['bbox'][0], each_gt['bbox'][1]),
                              (each_gt['bbox'][0] + each_gt['bbox'][2],
                               each_gt['bbox'][1] + each_gt['bbox'][3]),
                              GREEN, bbox_line_width)
                if separately:
                    plt.imshow(img)
                    plt.show()
                    shown = True
            if not shown:
                plt.imshow(img)
                plt.show()

    def set_predictions(self, predictions):
        self.predictions = predictions


class PersonDetectionDataset(Dataset):
    def __init__(self, path_to_data, path_to_annotations_file, is_sampled=False, samples_number=5):
        super(PersonDetectionDataset, self).__init__(path_to_data, path_to_annotations_file, is_sampled=False,
                                                     samples_number=5)

    def describe_gt(self, with_plots=False):
        images_number = len(self.annotations['images_info'])


        print('Images number in dataset = {}'.format(images_number))
