"""Read and write helpers."""
from __future__ import annotations
import enum
import os
import xml.etree.ElementTree as ET
from numpy import imag
import torch
from torchvision.io import read_image
from torchvision import transforms
from torchvision.utils import draw_bounding_boxes

# read images and annotations
# logic: different folders/same file name/different file type
def get_files(
    image_folder, annotation_folder, image_type=".jpg", annotation_type=".xml"
):
    files = [
        (
            image_folder + image_file,
            annotation_folder + image_file.replace(image_type, annotation_type),
        )
        for image_file in os.listdir(image_folder)
    ]
    return files


def get_label_map(annotation_files):
    for annotation_file in annotation_files:
        tree = ET.parse(annotation_file)
        root = tree.getroot()
        class_names = []
        for member in root.findall("object"):
            class_names.append(member[0].text)
    return {
        class_name: label
        for label, class_name in enumerate(sorted(list(set(class_names))))
    }


def get_image_size(image_files):
    image_sizes = []
    for image_file in image_files:
        image_sizes.append(read_image(image_file).shape[-2:])
    all_image_sizes = list(set(image_sizes))
    if len(all_image_sizes) == 1:
        adjustment = False
        all_image_sizes = tuple(all_image_sizes[0])
    else:
        adjustment = True
        all_image_sizes = (
            max([image_size[0] for image_size in all_image_sizes]),
            max([image_size[0] for image_size in all_image_sizes]),
        )
    return all_image_sizes, adjustment


# parse xml file
def get_annotation(file_path, mapping=None):
    tree = ET.parse(file_path)
    root = tree.getroot()
    bbox_coordinates = []
    class_names = []
    for member in root.findall("object"):
        class_name = member[0].text  # class name
        if mapping is not None:
            class_name = torch.tensor(mapping[class_name])
        class_names.append(class_name)

        # bbox coordinates
        xmin = int(member[4][0].text)
        ymin = int(member[4][1].text)
        xmax = int(member[4][2].text)
        ymax = int(member[4][3].text)
        # store data in list
        bbox_coordinates.append([xmin, ymin, xmax, ymax])

    return {
        "labels": torch.Tensor(class_names).long(),
        "boxes": torch.Tensor(bbox_coordinates),
    }


# visualize annotation
def get_visualization(image_tensor, annotations, out_path=None):
    transformer = transforms.ToPILImage()
    annotated_tensor = draw_bounding_boxes(image_tensor, annotations)
    annotated_image = transformer(annotated_tensor)
    if out_path is None:
        annotated_image.show()
    else:
        annotated_image.save(out_path)


def visualize_prediction(image_tensors, predictions, out_path, annotations, denormalize, top_n=3):
    transformer = transforms.ToPILImage()
    denormalizer = transforms.Compose(
        [
            transforms.Normalize(
                mean=[0.0, 0.0, 0.0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
            ),
            transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0]),
        ]
    )
    for i, image_tensor in enumerate(image_tensors):
        if denormalize:
            image_tensor = denormalizer(image_tensor)
        image_tensor = image_tensor*255
        image_tensor = image_tensor.to(torch.uint8)
        if predictions:
            boxes = predictions[i]["boxes"].to(torch.int64)[:top_n]
            labels = [str(l) for l in predictions[i]["labels"].tolist()][top_n]

            scores = predictions[i]["scores"]
            image_tensor = draw_bounding_boxes(
                image_tensor, boxes=boxes, labels=labels, colors="red"
            )
        if annotations:
            boxes = annotations[i]["boxes"].to(torch.int64)
            labels = [str(l) for l in annotations[i]["labels"].tolist()]

            image_tensor = draw_bounding_boxes(
                image_tensor, boxes=boxes, labels=labels, colors="green"
            )

        transformer(image_tensor).save(out_path + f"{i}_img.png")
