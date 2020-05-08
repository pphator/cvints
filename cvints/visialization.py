import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt
import seaborn as sns
from PIL import Image
from PIL import ImageDraw


COLORS = ['#00ff88', '#ff0000', '#07aca7', '#ff9800', '#ff023c', '#a2aca7', '#070aa7']


def draw_gt_with_predictions(model):
    pass


def recall_scatterplot(model):
    evaluated_images_ids = model.evaluation_results.keys()
    true_positives_per_image = {}
    annotated_per_image = {}
    for each_image_id in evaluated_images_ids:
        annotated_per_image[each_image_id] = len(
            [
                x
                for x in model.test_dataset.annotations["annotations_info"]
                if x["image_id"] == each_image_id
            ]
        )
        true_positives_per_image[each_image_id] = model.evaluation_results[
            each_image_id
        ][0]

    sns.scatterplot(
        list(true_positives_per_image.values()), list(annotated_per_image.values())
    )
    plt.show()


def miss_rate_per_annotated_number(model):
    evaluated_images_ids = model.evaluation_results.keys()
    annotated_per_image = {}
    miss_rate_per_image = {}
    for each_image_id in evaluated_images_ids:
        annotated_per_image[each_image_id] = len(
            [
                x
                for x in model.test_dataset.annotations["annotations_info"]
                if x["image_id"] == each_image_id
            ]
        )
        miss_rate_per_image[each_image_id] = (
            1
            - model.evaluation_results[each_image_id][0]
            / annotated_per_image[each_image_id]
        )

    grouped_by_annotated_number_dict = defaultdict(list)

    for key, value in sorted(annotated_per_image.items()):
        grouped_by_annotated_number_dict[value].append(key)

    result = {}
    for each_annotated_humans_number in grouped_by_annotated_number_dict.keys():
        images_with_same_annotated_number = grouped_by_annotated_number_dict[
            each_annotated_humans_number
        ]
        tmp_miss_rate_list = [
            miss_rate_per_image[x] for x in images_with_same_annotated_number
        ]
        result[each_annotated_humans_number] = round(np.mean(tmp_miss_rate_list), 2)

    sorted_results = sorted(result.items())
    number_of_annotated_humans_in_dataset = result.keys()
    mean_miss_rate_per_annotated_humans_number = result.values()

    plt.bar(
        number_of_annotated_humans_in_dataset,
        mean_miss_rate_per_annotated_humans_number,
    )
    plt.title("Mean miss rate per annotated humans number")
    plt.xlabel("Annotated humans No")
    plt.ylabel("Mean miss rate")
    plt.xticks(
        range(max(number_of_annotated_humans_in_dataset) + 1),
        range(max(number_of_annotated_humans_in_dataset) + 1),
    )
    plt.show()


def show_image(image):
    """
    Function to show image

    Parameters
    ----------
    image : PIL Image

    """
    image.show()


def open_image(path_to_image):
    """
        Return PIL Image by path to the image

    Parameters
    ----------
    path_to_image : str

    Returns
    -------
    image : PIL Image
    """
    image = Image.open(path_to_image)
    return image


def put_bboxes_to_image(image, annotations):
    """
    Parameters
    ----------
    image : PIL Image
    annotations : dict
        {cat_id: (bbox, score)}

    Returns
    -------
    draw : PIL Image
    """

    # check how many categories there are in the annotations
    categories = list(annotations.keys())
    colors = COLORS[:len(categories)]

    draw = ImageDraw.Draw(image)
    for cat_index in range(len(categories)):
        category = categories[cat_index]
        each_category_annotations = annotations[category]
        for each_annotation in each_category_annotations:
            draw.rectangle(((each_annotation[0][0], each_annotation[0][1]),
                            (each_annotation[0][0] + each_annotation[0][2],
                             each_annotation[0][1] + each_annotation[0][3])),
                           outline=colors[cat_index], width=5)

    return image
