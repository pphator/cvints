#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 14:03:18 2020

@author: innolab001
"""

class Dataset():
    def __init__(self, path_to_images, path_to_annotations, is_sampled=False,
                 samples_number=5):
        self.images_path = path_to_images
        self.annotations_path = path_to_annotations
        self.is_sampled = is_sampled
        if self.annotations_path:
            with open(self.annotations_path, 'r') as f:
                annotations_data = json.load(f)
        
        if is_sampled:
            samples_from_images = sample(annotations_data['images'], 
                                          samples_number)
            sample_images_ids = [x['id'] for x in samples_from_images]
            
            images = samples_from_images
            annotations = [x for x in annotations_data['annotations'] if 
                           x['image_id'] in sample_images_ids]
            self.annotations = {'images': images,
                      'annotations': annotations}
            
        else:
            self.annotations = annotations_data