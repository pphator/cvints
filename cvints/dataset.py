#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import numpy as np
from random import sample
import cv2
from matplotlib import pyplot as plt
import seaborn as sns


GREEN = (0, 255, 0)


class Dataset:
    def __init__(self, path_to_data, path_to_annotations_file, is_sampled=False, samples_number=5):
        self.path_to_data = path_to_data
        self.path_to_annotations_file = path_to_annotations_file
        self.is_sampled = is_sampled
        self.path_to_evaluations_results_file = None
        self.model_evaluation_results = None
        if self.path_to_annotations_file:
            with open(self.path_to_annotations_file, 'r') as f:
                annotations_data = json.load(f)

        if is_sampled:
            sampled_images_info = sample(annotations_data['images'], samples_number)
            sampled_images_ids = [x['id'] for x in sampled_images_info]
            sampled_annotations_data = [x for x in annotations_data['annotations'] if
                                        x['image_id'] in sampled_images_ids]
            self.annotations = {'images_info': sampled_images_info, 'annotations_info': sampled_annotations_data}

        else:
            images_info = annotations_data['images']
            annotations_data = annotations_data['annotations']
            self.annotations = {'images_info': images_info, 'annotations_info': annotations_data}

    def draw_gt_bboxes(self, separately=False, bbox_line_width=3):
        shown = False
        images_info = self.annotations['images_info']
        annotation_info = self.annotations['annotations_info']
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

    def set_model_evaluation_results(self, path_to_file_with_evaluation_results):
        self.path_to_evaluations_results_file = path_to_file_with_evaluation_results
        if self.path_to_evaluations_results_file:
            with open(self.path_to_evaluations_results_file, 'r') as f:
                self.model_evaluation_results = json.load(f)


class PersonDetectionDataset(Dataset):
    def __init__(self, path_to_data, path_to_annotations_file, is_sampled=False, samples_number=5):
        super(PersonDetectionDataset, self).__init__(path_to_data, path_to_annotations_file, is_sampled,
                                                     samples_number)

    def describe_gt(self, with_plots=False):
        images_number = len(self.annotations['images_info'])
        total_person_number = len(self.annotations['annotations_info'])
        persons_per_image_dict = {}
        for each_image_info in self.annotations['annotations_info']:
            img_id = each_image_info['image_id']
            if img_id not in persons_per_image_dict.keys():
                persons_per_image_dict[img_id] = 1
            else:
                persons_per_image_dict[img_id] += 1

        persons_per_image_list = list(persons_per_image_dict.values())
        mean_persons_per_image = np.mean(persons_per_image_list)
        median_persons_per_image = np.median(persons_per_image_list)
        std_persons_per_image = np.std(persons_per_image_list)

        print('Images number in dataset = {}'.format(images_number))
        print('Persons number in dataset = {}'.format(total_person_number))
        print('Mean value of persons per image = {:.2f}'.format(mean_persons_per_image))
        print('Median value of persons per image = {:.2f}'.format(median_persons_per_image))
        print('Std value of persons per image = {:.2f}'.format(std_persons_per_image))

        if with_plots:
            plt.title('Persons per image distribution')
            sns.distplot(persons_per_image_list, kde=False)
            plt.show()

    # ToDo: complete method to check person detection model evaluation file format. Should be list of dicts (but may
    #  be discussed)
    def check_model_evaluation_results_format(self):

        return None
