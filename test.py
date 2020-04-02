from dataset import Dataset


if __name__ == '__main__':
    path_to_data = 'C:\\Users\\vboychuk\\Work\\Projects\\Conference room\\Dataset\\PIC_2.0\\image\\val'
    path_to_annotations_file = 'C:\\Users\\vboychuk\\Work\\Projects\\Conference room\\Dataset\\PIC_2.0\\val_.json'

    dataset = Dataset(path_to_data, path_to_annotations_file, is_sampled=True)

    dataset.draw_gt_bboxes(bbox_line_width=5, separately=True)
