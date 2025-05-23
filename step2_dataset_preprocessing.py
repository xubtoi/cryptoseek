import os
import json
import shutil
import yaml
from glob import glob
from PIL import Image
from sklearn.model_selection import train_test_split
from clearml import Task, Dataset

# Initialize ClearML Task
task = Task.init(project_name='CryptoSeek', task_name='Step 2 - Dataset Preprocessing')

# Configuration arguments
args = {
    'input_dataset_project': 'CryptoSeek',
    'input_dataset_name': 'Resized_Cityscapes',
    'input_dataset_version': '1.0.1',
    'input_width': 640,
    'input_height': 640,
    'output_dir': 'dataset',
    'output_tag': ['step2', 'v1.0'],
    'output_id': '9fa4d7e8172b4c70867e77849813f2a5',
    'classes': ['person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'traffic light', 'traffic sign'],
    'test_split_ratio': 0.1,
    'random_seed': 42
}
task.connect(args)

# task.execute_remotely()

def FindMatchedDataset(ID):
    try:
        ds = Dataset.get(dataset_id=ID)
        return True, ds
    except ValueError as e:
        print("Dataset Not Exist!")
        return False, None

# Function: Convert a Cityscapes polygon annotation to YOLO format and save as .txt
def convert_annotation(json_path, image_path, output_txt_path, original_size=(args['input_width'], args['input_height'])):
    with open(json_path, 'r') as f:
        data = json.load(f)
    img = Image.open(image_path)
    w, h = img.size # resized size

    # original size
    orig_w, orig_h = original_size
    scale_x = w / orig_w
    scale_y = h / orig_h

    yolo_lines = []
    for obj in data.get('objects', []):
        label = obj.get('label')
        if label not in label2id:
            continue
        polygon = obj.get('polygon', [])
        if not polygon:
            continue

        # scaling coordinates
        scaled_polygon = [(p[0] * scale_x, p[1] * scale_y) for p in polygon]

        x_coords = [p[0] for p in scaled_polygon]
        y_coords = [p[1] for p in scaled_polygon]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        x_center = (x_min + x_max) / 2.0 / w
        y_center = (y_min + y_max) / 2.0 / h
        bbox_width = (x_max - x_min) / w
        bbox_height = (y_max - y_min) / h
        cls_id = label2id[label]
        yolo_lines.append(f'{cls_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}')
    with open(output_txt_path, 'w') as f:
        f.write('\n'.join(yolo_lines))

# Function: Convert image+annotation pairs to YOLO format for a specific split
def process_split(images, annots, split_name):
    for img_path, json_path in zip(images, annots):
        filename = os.path.basename(img_path)
        img_dst = os.path.join(output_base, 'images', split_name, filename)
        lbl_dst = os.path.join(output_base, 'labels', split_name, filename.replace('.png', '.txt'))
        shutil.copy(img_path, img_dst)
        convert_annotation(json_path, img_path, lbl_dst)

check, temp = FindMatchedDataset(args['output_id'])

if check:
    print(f'SKIP Dataset: {temp.id}')
    print(f'Dataset version: {temp.version}')
    task.upload_artifact('dataset_id', temp.id)
    task.upload_artifact('dataset_version', temp.version)
else:
    # Load dataset from ClearML Dataset v1.0.1
    dataset = Dataset.get(
        dataset_project=args['input_dataset_project'],
        dataset_name=args['input_dataset_name'],
        dataset_version=args['input_dataset_version']
    )
    local_path = dataset.get_local_copy()

    # Create YOLO-format output directories
    output_base = args['output_dir']
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_base, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_base, 'labels', split), exist_ok=True)

    # Map class names to YOLO class IDs
    label2id = {label: idx for idx, label in enumerate(args['classes'])}

    # Process train split
    train_images = sorted(glob(os.path.join(local_path, 'train/images/*.png')))
    train_annots = sorted(glob(os.path.join(local_path, 'train/annotations/*.json')))

    # Split a portion of train as test set
    X_train, X_test, y_train, y_test = train_test_split(
        train_images, train_annots,
        test_size=args['test_split_ratio'],
        random_state=args['random_seed']
    )

    # Convert train and test splits
    process_split(X_train, y_train, 'train')
    process_split(X_test, y_test, 'test')

    # Process val split
    val_images = sorted(glob(os.path.join(local_path, 'val/images/*.png')))
    val_annots = sorted(glob(os.path.join(local_path, 'val/annotations/*.json')))
    process_split(val_images, val_annots, 'val')

    # Create and upload YOLO Dataset
    ds = Dataset.create(
        dataset_name=args['input_dataset_name'],
        dataset_project=args['input_dataset_project'],
        parent_datasets=[dataset.id]
    )

    # Write YOLO dataset.yaml
    dataset_yaml_path = os.path.join(output_base, 'dataset.yaml')
    yolo_yaml = {
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(args['classes']),
        'names': args['classes']
    }
    with open(dataset_yaml_path, 'w') as f:
        yaml.dump(yolo_yaml, f)

    ds.remove_files('*')
    ds.add_files(output_base)

    ds.add_tags(args['output_tag'])
    ds.finalize(auto_upload=True)
    task.upload_artifact('dataset_id', ds.id)
    task.upload_artifact('dataset_version', ds.version)
    
    print(f'Dataset created with train/val/test split. ID: {ds.id}')
    print('Uploading artifacts in Step 2')
    print('Step 2 Done.')