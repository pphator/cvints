from cvints import HumanDetectionDataset
from cvints import PersonDetectionEvaluator
from cvints import ObjectDetectionModel

if __name__ == '__main__':
    path_to_data = 'C:\\Users\\vboychuk\\Work\\Projects\\Conference room\\Dataset\\PIC_2.0\\image\\val'
    path_to_annotations_file = 'C:\\Users\\vboychuk\\Work\\Projects\\Conference room\\Dataset\\PIC_2.0\\val_.json'
    path_to_model_evaluation_results_file = 'C:\\Users\\vboychuk\\Work\\Projects\\cvints\\examples\\ssd_mobilenet_v2_coco_33_22_1_4_2020.json'

    test_dataset = HumanDetectionDataset(path_to_data, path_to_annotations_file, is_sampled=True)

    model = ObjectDetectionModel(name='Jetson Nano human detector (ssd_mobilenet_v2_coco)', test_dataset=test_dataset,
                                 path_to_detection_results_file=path_to_model_evaluation_results_file)

    model.load_detection_results()

