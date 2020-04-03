from cvints import PersonDetectionDataset
from cvints import PersonDetectionEstimator

def describe_gt_data_with_plots():
    path_to_data = 'C:\\Users\\vboychuk\\Work\\Projects\\Conference room\\Dataset\\PIC_2.0\\image\\val'
    path_to_annotations_file = 'C:\\Users\\vboychuk\\Work\\Projects\\Conference room\\Dataset\\PIC_2.0\\val_.json'

    dataset = PersonDetectionDataset(path_to_data, path_to_annotations_file)

    dataset.describe_gt(with_plots=True)


if __name__ == '__main__':
    path_to_data = 'C:\\Users\\vboychuk\\Work\\Projects\\Conference room\\Dataset\\PIC_2.0\\image\\val'
    path_to_annotations_file = 'C:\\Users\\vboychuk\\Work\\Projects\\Conference room\\Dataset\\PIC_2.0\\val_.json'
    path_to_model_evaluation_results_file = 'C:\\Users\\vboychuk\\Work\\Projects\\cvints\\examples\\ssd_mobilenet_v2_coco_33_22_1_4_2020.json'

    dataset = PersonDetectionDataset(path_to_data, path_to_annotations_file)
    dataset.set_model_evaluation_results(path_to_model_evaluation_results_file)

    estimator = PersonDetectionEstimator(dataset)

    res = estimator.compute_mean_iou()

