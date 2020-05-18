from cvints.dataset import DesktopCODataset
from cvints.processing_results import ObjectDetectionResults
from cvints.model import ObjectDetectionModel
from cvints.experiment import Experiment, Tasks
from cvints.metrics import Metrics

import pickle
from pprint import PrettyPrinter


if __name__ == '__main__':
    pprinter = PrettyPrinter()

    task = Tasks.OBJECT_DETECTION

    path_to_processing_results = 'rus\\fasterrcnn_resnet50_desktopco.txt'
    path_to_processed_files_filenames = 'rus\\desktopco_processed_files_filenames.txt'

    dataset = DesktopCODataset()

    with open(path_to_processed_files_filenames, 'rb') as in_file:
        processed_images_filenames = pickle.load(in_file)

    # prepare processing result to load
    with open(path_to_processing_results, "rb") as fp:  # Unpickling
        raw_processing_results_data = pickle.load(fp)
    raw_processing_results = ObjectDetectionResults()
    processed_images_info = dataset.get_images_info_by_filenames(processed_images_filenames)
    raw_processing_results.load_results(raw_processing_results_data, processed_images_info)

    # processed_images_ids = dataset.get_ids_by_filenames(processed_images_filenames)

    # describe model
    model_config = {'name': 'fasterrcnn_resnet50',
                    'input_size': (300, 300),
                    'NMS': False}
    model = ObjectDetectionModel(config=model_config)

    # push raw processing result through the model to update them
    # for example transform bboxes to original images sizes
    processing_results = model.set_processing_results(raw_processing_results)

    # create experiment
    experiment = Experiment(task, model, dataset, processing_results)

    experiment.filer_results_by_scores(scores_threshold=0.6)
    experiment.apply_nms()

    experiment.metrics = ['Jaccard index', 'Precision']

    metrics_result = experiment.metrics.calculate()

    pprinter.pprint(metrics_result)

