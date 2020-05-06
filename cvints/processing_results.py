from collections import defaultdict


class ProcessingResults:
    def __init__(self):
        self.results = []


class ObjectDetectionResults(ProcessingResults):
    def __init__(self):
        super(ObjectDetectionResults, self).__init__()
        self.bboxes = []
        self.scores = []
        self.labels = []

    def __str__(self):
        return "member of ObjectDetectionResults"

    def __len__(self):
        return len(self.results)

    def load_results(self, results, images_ids):
        """
        Load raw_results and perform it in association with images ids

        Parameters
        ----------
        results : array-like
        images_ids : array-like

        Returns
        -------

        """
        if isinstance(results, list):
            if len(results) > 0:
                # every item in the list should have bboxes, scores and labels fields
                for each_image_results, image_id in zip(results, images_ids):
                    img_results = {'image_id': image_id,
                                   'detections': defaultdict(list)}

                    if 'boxes' in each_image_results and \
                            'scores' in each_image_results and \
                            'labels' in each_image_results:
                        for b, s, l in zip(each_image_results['boxes'].numpy(),
                                           each_image_results['scores'].numpy(),
                                           each_image_results['labels'].numpy()):

                            img_results['detections'][str(l)].append((b, s))

                            self.results.append(img_results)
                            # self.results.append({'image_id': image_id,
                            #                      'bbox': b,
                            #                      'score': s,
                            #                      'label': l})
                    else:
                        raise Exception
            else:
                raise Exception
        else:
            raise TypeError

    def set_bboxes(self, bboxes):
        self.bboxes = bboxes

    def set_scores(self, scores):
        self.scores = scores

    def set_labels(self, labels):
        self.labels = labels

    def set_results(self, results):
        self.results = results
