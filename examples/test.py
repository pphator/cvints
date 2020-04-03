from cvints import PersonDetectionDataset


if __name__ == '__main__':
    path_to_data = 'C:\\Users\\vboychuk\\Work\\Projects\\Conference room\\Dataset\\PIC_2.0\\image\\val'
    path_to_annotations_file = 'C:\\Users\\vboychuk\\Work\\Projects\\Conference room\\Dataset\\PIC_2.0\\val_.json'

    dataset = PersonDetectionDataset(path_to_data, path_to_annotations_file)

    dataset.describe_gt()
