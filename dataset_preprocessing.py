from clearml import Task
import os
import json
import numpy as np
from PIL import Image
import yaml

# Initialize ClearML Task
# task = Task.init(project_name="CryptoSeek", task_name="Step 2 - Dataset Preprocessing")

# Arguments
args = {
    'dataset_task_id': '',  # Fill Step1 task ID
    'resize_width': 512,
    'resize_height': 256,
}
# task.connect(args)

# Download dataset artifact
# dataset_task = Task.get_task(task_id=args['dataset_task_id'])
# dataset_dir = dataset_task.artifacts['raw_sampled_split_dataset'].get_local_copy()

# ==== 这里是本地测试参数 ====
dataset_dir = 'dataset/cityscapes_raw_sampled'  # Step1采样+split好的数据
resize_width = 512
resize_height = 256
# ===========================

# Output directory
preprocessed_base_dir = "dataset/cityscapes_preprocessed"
os.makedirs(preprocessed_base_dir, exist_ok=True)

categories = {
    "person": 0, "rider": 1, "car": 2, "truck": 3, "bus": 4,
    "train": 5, "motorcycle": 6, "bicycle": 7, "traffic light": 8, "traffic sign": 9
}

# Function
def polygon_to_bbox(polygon):
    polygon = np.array(polygon)
    xmin = np.min(polygon[:, 0])
    ymin = np.min(polygon[:, 1])
    xmax = np.max(polygon[:, 0])
    ymax = np.max(polygon[:, 1])
    return xmin, ymin, xmax, ymax

# Process splits
for split in ['train', 'val', 'test']:
    images_in = os.path.join(dataset_dir, split, "images")
    annotations_in = os.path.join(dataset_dir, split, "annotations")
    images_out = os.path.join(preprocessed_base_dir, "images", split)
    labels_out = os.path.join(preprocessed_base_dir, "labels", split)

    os.makedirs(images_out, exist_ok=True)
    os.makedirs(labels_out, exist_ok=True)

    for file in os.listdir(annotations_in):
        if file.endswith("_polygons.json"):
            base_name = file.replace("_gtFine_polygons.json", "")
            json_path = os.path.join(annotations_in, file)
            img_path = os.path.join(images_in, base_name + "_leftImg8bit.png")

            if not os.path.exists(img_path):
                continue

            with open(json_path, 'r') as f:
                data = json.load(f)

            img_w, img_h = data["imgWidth"], data["imgHeight"]
            label_lines = []

            for obj in data["objects"]:
                label = obj["label"]
                if label in categories:
                    xmin, ymin, xmax, ymax = polygon_to_bbox(obj["polygon"])
                    x_center = (xmin + xmax) / 2.0 / img_w
                    y_center = (ymin + ymax) / 2.0 / img_h
                    width = (xmax - xmin) / img_w
                    height = (ymax - ymin) / img_h
                    class_id = categories[label]
                    label_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

            if label_lines:
                with open(os.path.join(labels_out, base_name + ".txt"), "w") as out_f:
                    out_f.write("\n".join(label_lines))

                img = Image.open(img_path)
                img = img.resize((args['resize_width'], args['resize_height']))
                img.save(os.path.join(images_out, base_name + ".png"))

# Create updated data.yaml
data_yaml = {
    'path': os.path.join('..', preprocessed_base_dir),
    'train': "images/train",
    'val': "images/val",
    'test': "images/test",
    'nc': len(categories),
    'names': list(categories.keys())
}

with open(os.path.join(preprocessed_base_dir, "data.yaml"), 'w') as f:
    yaml.dump(data_yaml, f)

# Upload artifact
# task.upload_artifact('preprocessed_dataset', artifact_object=preprocessed_base_dir)

print(f'Step 2 done: Preprocessed dataset (train/val/test) uploaded')
