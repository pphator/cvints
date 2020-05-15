from cvints import ObjectDetectionDataset, DesktopCODataset
from cvints import ObjectDetectionModel
from cvints.processing_results import ObjectDetectionResults
from cvints import Experiment
from cvints.experiment import Tasks
import pickle
from pprint import PrettyPrinter


if __name__ == '__main__':
    pprinter = PrettyPrinter()

    task = Tasks.OBJECT_DETECTION

    path_to_images = '..\\Datasets\\detection\\desktopco\\images\\'
    path_to_annotation_file = '..\\Datasets\\detection\\desktopco\\annotations\\instances_default.json'
    path_to_processing_results = 'rus\\fasterrcnn_resnet50_desktopco.txt'
    path_to_processed_files_filenames = 'rus\\desktopco_processed_files_filenames.txt'

    dataset = DesktopCODataset()

    # sample_ds = ObjectDetectionDataset.get_subset(dataset, size=1)

    # dataset.show_images(with_bboxes=True)

    with open(path_to_processed_files_filenames, 'rb') as in_file:
        processed_images_filenames = pickle.load(in_file)

    processed_images_ids = dataset.get_ids_by_filenames(processed_images_filenames)
    processed_images_info = dataset.get_images_info_by_filenames(processed_images_filenames)

    processing_results = ObjectDetectionResults()

    with open(path_to_processing_results, "rb") as fp:  # Unpickling
        raw_processing_results = pickle.load(fp)

    processing_results.load_results(raw_processing_results, processed_images_info)

    model_config = {'name': 'fasterrcnn_resnet50',
                    'input_size': (300, 300),
                    'NMS': False}

    model = ObjectDetectionModel(config=model_config)
    processing_results = model.set_processing_results(processing_results)

    experiment = Experiment(task, model, dataset, processing_results)

    experiment.metrics = ['Jaccard index']
    metrics_result = experiment.metrics.calculate()

    pprinter.pprint(metrics_result)

    # experiment.filer_results_by_scores(scores_threshold=0.5)
    #
    # experiment.show_images()
    #
    # experiment.apply_nms(nms_threshold=0.01)
    #
    # experiment.show_images()
