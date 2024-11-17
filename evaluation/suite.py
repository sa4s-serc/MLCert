import os
import torch
import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision.transforms as T
import cv2
from ultralytics import YOLO
from pycocotools.coco import COCO

# model = ""


def predict_on_image(image, model):
    results = model.predict(image, save=True)
    prediction = []

    for r in results:
        preds = r.boxes
        if len(preds.cls) == 0:
            continue

        # Collecting the bounding box predictions
        for i in range(len(preds.cls)):
            x_min = preds.xywh[i][0] - preds.xywh[i][2] / 2
            y_min = preds.xywh[i][1] - preds.xywh[i][3] / 2
            w = preds.xywh[i][2]
            h = preds.xywh[i][3]
            confidence = preds.conf[i]
            class_id = preds.cls[i]
            prediction.append((x_min, y_min, w, h, confidence, class_id))

    return prediction


class MilitaryVehiclesDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms

        # Load COCO annotations
        ann_path = os.path.join(root, "_annotations.coco.json")
        with open(ann_path, "r") as f:
            self.coco = json.load(f)

        self.images = self.coco["images"]
        self.annotations = self.coco["annotations"]
        self.categories = {cat["id"]: cat["name"] for cat in self.coco["categories"]}

    def __getitem__(self, idx):
        # Get image and its metadata

        img_info = self.images[idx]
        img_path = os.path.join(self.root, img_info["file_name"])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        # Get annotations for this image
        ann_ids = [ann for ann in self.annotations if ann["image_id"] == img_info["id"]]
        bboxes = [ann["bbox"] for ann in ann_ids]
        labels = [ann["category_id"] for ann in ann_ids]

        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # if self.transforms is not None:
        #     res = self.transforms(
        #         image=np.array(img), bboxes=bboxes, class_labels=labels
        #     )
        #     img = res["image"]
        #     bboxes = res["bboxes"]
        #     labels = res["class_labels"]
        target = {
            "boxes": [torch.Tensor(x) for x in bboxes],
            "labels": [torch.Tensor(x) for x in labels],
        }
        # target = {"boxes": torch.Tensor(bboxes), "labels": torch.tensor(labels)}

        return img, target

    def __len__(self):
        return len(self.images)


# import albumentations as A
# from albumentations.pytorch import ToTensorV2

# Update transformations to use COCO format
# data_transforms = A.Compose(
#     [
#         A.Resize(height=256, width=256),
#         A.CenterCrop(height=224, width=224),
#         A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#         ToTensorV2(),
#     ],
#     bbox_params=A.BboxParams(format="coco", label_fields=["class_labels"]),
# )


def get_datasets(dataset_path):
    # Assuming you have renamed the class to MilitaryVehiclesDataset
    train_dataset = MilitaryVehiclesDataset(
        root=dataset_path + "/train",
        transforms=None,
    )
    test_dataset = MilitaryVehiclesDataset(
        root=dataset_path + "/valid_noisy",
        transforms=None,
    )
    val_dataset = MilitaryVehiclesDataset(
        root=dataset_path + "/valid",
        transforms=None,
    )

    print(f"Number of training images: {len(train_dataset)}")
    print(f"Number of test images: {len(test_dataset)}")
    print(f"Number of val images: {len(val_dataset)}")

    print(f"Example output of an image shape: {train_dataset[0][0].shape}")
    print(f"Example output of a label: {train_dataset[0][1]}")
    # test_dataset.transforms = A.Compose([ToTensorV2()])

    return train_dataset, test_dataset


from deepchecks.vision.vision_data import BatchOutputFormat

# model = YOLO("best.pt")


def transform_labels_to_cxywh(original_labels):
    """
    Convert a batch of data to labels in the expected format. The expected format is an iterator of arrays, each array
    corresponding to a sample. Each array element is in a shape of [B, 5], where B is the number of bboxes
    in the image, and each bounding box is in the structure of [class_id, x, y, w, h].
    """
    label = []
    for annotation in original_labels:
        if len(annotation["boxes"]):
            bbox = torch.stack(annotation["boxes"])

            label.append(
                torch.concat(
                    [torch.stack(annotation["labels"]).reshape((-1, 1)), bbox], dim=1
                )
            )
        else:
            label.append(torch.tensor([]))
    return label


def infer_on_images(images, model):
    predictions = []
    for img in images:
        predictions.append(predict_on_image(img, model))

    return predictions


def get_untransformed_images(original_images):
    """
    Convert a batch of data to images in the expected format. The expected format is an iterable of images,
    where each image is a numpy array of shape (height, width, channels). The numbers in the array should be in the
    range [0, 255] in a uint8 format.
    """
    inp = (
        torch.stack(list(original_images))
        .cpu()
        .detach()
        .numpy()
        .transpose((0, 2, 3, 1))
    )
    return inp


def deepchecks_collate_fn(batch) -> BatchOutputFormat:
    """Return a batch of images, labels and predictions in the deepchecks format."""
    # batch received as iterable of tuples of (image, label) and transformed to tuple of iterables of images and labels:
    batch = tuple(zip(*batch))
    images = batch[0]
    labels = transform_labels_to_cxywh(batch[1])
    predictions = infer_on_images(batch[0], model)

    return BatchOutputFormat(images=images, labels=labels, predictions=predictions)


def get_label_map(dataset_path):
    # Load the COCO dataset
    coco = COCO(dataset_path + "/train" + "/_annotations.coco.json")
    categories = coco.cats
    classes = [i[1]["name"] for i in categories.items()]

    # Generate the LABEL_MAP dictionary
    label_map = {idx: class_name for idx, class_name in enumerate(classes)}

    return label_map


from deepchecks.vision.vision_data import VisionData


def get_data(dataset_path):

    LABEL_MAP = get_label_map(dataset_path)
    train_dataset, test_dataset = get_datasets(dataset_path)

    train_loader = DataLoader(
        train_dataset, batch_size=64, collate_fn=deepchecks_collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=64, collate_fn=deepchecks_collate_fn
    )

    training_data = VisionData(
        batch_loader=train_loader, task_type="object_detection", label_map=LABEL_MAP
    )
    test_data = VisionData(
        batch_loader=test_loader, task_type="object_detection", label_map=LABEL_MAP
    )
    return training_data, test_data


# from deepchecks.vision.suites import model_evaluation
from deepchecks.vision.suites import full_suite


def run_suite(training_data, test_data):
    suite = full_suite()
    result = suite.run(training_data, test_data)
    result.save_as_html("output_noisy.html")

    # Assume suite_result is an instance of SuiteResult
    # json_str = result.to_json()
    # result_dict = json.loads(json_str)

    # sorted_results = sorted(result_dict["results"], key=lambda x: x["check"]["name"])

    # # Save the sorted result JSON into a new file (outputs_sorted.json) for easier reference
    # sorted_result_json = {"name": result_dict["name"], "results": sorted_results}

    # # Save the sorted JSON to a file
    # with open("outputs_sorted.json", "w") as f:
    #     json.dump(sorted_result_json, f, indent=4)

    # # with open("full_suite.json", "w") as f:
    # #     f.write(json_str)

    # return sorted_result_json


from ultralytics import YOLO


def main(model_name, dataset_path):

    global model
    model = YOLO(model_name, task="detect")

    training_data, test_data = get_data(dataset_path)

    return run_suite(training_data, test_data)


if __name__ == "__main__":
    import sys

    main("best.onnx", "../Military vehicles object detection.v16i.coco")
