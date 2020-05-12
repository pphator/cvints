#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import numpy as np
from random import sample
from matplotlib import pyplot as plt
import seaborn as sns
from os import listdir
from PIL import Image
from cvints import visialization as cvints_vis
from collections import defaultdict

GREEN = (0, 255, 0)


class Dataset:
    """
        Base wrapper of data for CV tasks

        Attributes
        ----------
        path_to_data : str
            Absolute path to the folder with images and masks
        path_to_annotations_file : str
            Absolute path to the annotations file (file should be in COCO format)
        is_sampled: bool
            If the value of the parameter is True, to the Dataset will be loaded only
            `samples_number` images from the `path_to_data` with associated annotations from `path_to_annotations_file`
            If the value of the parameter is False, all images from `path_to_data` will be loaded to the Dataset
        samples_number: int
            The number of images to load to the Dataset if `is_sampled` is True
    """

    def __init__(self, path_to_data, path_to_annotations_file, is_sampled=False, sample_size=5):
        self.path_to_data = path_to_data
        # if self.path_to_data is not None:
        #     self.filenames = listdir(self.path_to_data)
        self.path_to_annotations_file = path_to_annotations_file
        self.is_sampled = is_sampled
        self.path_to_evaluations_results_file = None

        if self.path_to_annotations_file:
            with open(self.path_to_annotations_file, "r") as f:
                annotations_data = json.load(f)

        if is_sampled:
            sampled_images_info = sample(annotations_data["images"], sample_size)
            sampled_images_ids = [x["id"] for x in sampled_images_info]
            sampled_annotations_data = [x for x in annotations_data["annotations"]
                                        if x["image_id"] in sampled_images_ids]
            self.annotations = {
                "images_info": sampled_images_info,
                "annotations_info": sampled_annotations_data,
            }

        else:
            images_info = annotations_data["images"]
            annotations_data = annotations_data["annotations"]
            self.annotations = {
                "images_info": images_info,
                "annotations_info": annotations_data,
            }

        self.filenames = [x['file_name'] for x in self.annotations['images_info']]


    def __len__(self):
        result = 0
        if self.filenames is not None:
            result = len(self.filenames)
        return result

    @classmethod
    def get_subset(cls, dataset, size=5):
        return cls(
            path_to_data=dataset.path_to_data,
            path_to_annotations_file=dataset.path_to_annotations_file,
            is_sampled=True,
            sample_size=size,
        )

    def get_images_info_by_filenames(self, filenames):
        """
        Method to get id and size of all images by their filenames

        Parameters
        ----------
        filenames : list

        :return:
        """
        images_info = []
        for each_filename in filenames:
            item = next(x for x in self.annotations['images_info'] if x['file_name'] == each_filename)
            image_path = self.path_to_data + each_filename
            image = Image.open(image_path)
            images_info.append({'id': item['id'],
                                'size': image.size})

        return images_info

    def get_filename_by_id(self, image_id):
        item = next(x for x in self.annotations['images_info'] if x['id'] == image_id)['file_name']
        return item

    def get_ids_by_filenames(self, filenames):
        """
        Returns the list of images ids in filenames array order

        Parameters
        ----------
        filenames : array-like

        Returns
        -------
        images_ids : array-like
        """
        images_ids = []

        for each_filename in filenames:
            item = next(x for x in self.annotations['images_info'] if x['file_name'] == each_filename)
            images_ids.append(item['id'])

        return images_ids

    def get_annotations(self, filename):
        raise NotImplementedError('load_processing_results should be implemented in child classes')


class ObjectDetectionDataset(Dataset):
    """
        Wrapper of data for object detection tasks

        Attributes
        ----------
        path_to_data : str
            Absolute path to the folder with images and masks
        path_to_annotations_file : str
            Absolute path to the annotations file (file should be in COCO format)
        is_sampled: bool
            If the value of the parameter is True, to the Dataset will be loaded only
            `samples_number` images from the `path_to_data` with associated annotations from `path_to_annotations_file`
            If the value of the parameter is False, all images from `path_to_data` will be loaded to the Dataset
        samples_number: int
            The number of images to load to the Dataset if `is_sampled` is True
    """

    def __init__(self, path_to_data, path_to_annotations_file, is_sampled=False, sample_size=5):
        super(ObjectDetectionDataset, self).__init__(path_to_data, path_to_annotations_file, is_sampled, sample_size)

    def get_images_ids(self):
        return [x["id"] for x in self.annotations["images_info"]]

    def get_annotations(self, filename):
        image_id = self.get_ids_by_filenames([filename])[0]
        annotations = defaultdict(list)
        for each in self.annotations['annotations_info']:
            if each['image_id'] == image_id:
                annotations[str(each['category_id'])].append((each['bbox']))
        return annotations

    def show_images(self, with_bboxes=False, annotations=None):
        for each_image in self.filenames:
            img = Image.open(self.path_to_data + '\\' + each_image)
            if with_bboxes:
                if annotations is None:
                    annotations = self.get_annotations(each_image)
                img = cvints_vis.put_bboxes_to_image(img, annotations)
            img.show()


class HumanDetectionDataset(ObjectDetectionDataset):
    """
        Wrapper of data for the human detection task

        Attributes
        ----------
        path_to_data : str
            Absolute path to the folder with images and masks
        path_to_annotations_file : str
            Absolute path to the annotations file (file should be in COCO format)
        is_sampled: bool
            If the value of the parameter is True, to the Dataset will be loaded only
            `samples_number` images from the `path_to_data` with associated annotations from `path_to_annotations_file`
            If the value of the parameter is False, all images from `path_to_data` will be loaded to the Dataset
        samples_number: int
            The number of images to load to the Dataset if `is_sampled` is True
    """

    def __init__(self, path_to_data, path_to_annotations_file, is_sampled=False, sample_size=5):
        super(HumanDetectionDataset, self).__init__(path_to_data, path_to_annotations_file, is_sampled, sample_size)

    def describe_gt(self, with_plots=False):
        """ Describe data in dataset

            Parameters
            ----------
            with_plots : bool
                if True, the method will print some statistics as text and draw some plots
                if False, the method will only print some statistics of the data as text

            Returns
            ----------
            None
        """
        images_number = len(self.annotations["images_info"])
        total_person_number = len(self.annotations["annotations_info"])
        persons_per_image_dict = {}
        for each_image_info in self.annotations["annotations_info"]:
            img_id = each_image_info["image_id"]
            if img_id not in persons_per_image_dict.keys():
                persons_per_image_dict[img_id] = 1
            else:
                persons_per_image_dict[img_id] += 1

        persons_per_image_list = list(persons_per_image_dict.values())
        mean_persons_per_image = np.mean(persons_per_image_list)
        median_persons_per_image = np.median(persons_per_image_list)
        std_persons_per_image = np.std(persons_per_image_list)

        print("Images number in dataset = {}".format(images_number))
        print("Persons number in dataset = {}".format(total_person_number))
        print("Mean value of persons per image = {:.2f}".format(mean_persons_per_image))
        print(
            "Median value of persons per image = {:.2f}".format(
                median_persons_per_image
            )
        )
        print("Std value of persons per image = {:.2f}".format(std_persons_per_image))

        if with_plots:
            plt.title("Persons per image distribution")
            sns.distplot(persons_per_image_list, kde=False)
            plt.show()

    # ToDo: complete method to check person detection model evaluation file format. Should be list of dicts (but may
    #  be discussed)
    def check_model_evaluation_results_format(self):

        return None
