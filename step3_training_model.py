import os
os.environ.pop("MPLBACKEND", None)

import matplotlib
matplotlib.use("Agg")

from clearml import Task, Logger, Dataset
from ultralytics import YOLO
import torch
import yaml

def Training():
    # === Initialize ClearML Task ===
    task = Task.init(project_name="CryptoSeek", task_name="Step 3 - Training Model")
    # task.set_packages({'torchvision': '0.21'})
    logger = Logger.current_logger()

    # === Define configurable arguments ===
    args = {
        'input_dataset_project': 'CryptoSeek',
        'input_dataset_name': 'Resized_Cityscapes',
        'input_dataset_version': '1.0.2',     
        'model_arch': 'yolo11s.pt',  # Model architecture (nano by default)
        'img_size': 640,
        'epochs': 200,
        'learning_rate': 0.001,
        'batch': 16
    }
    task.connect(args)

    # === Load YOLO-format Dataset (v1.0.4) ===
    dataset = Dataset.get(
        dataset_project=args['input_dataset_project'],
        dataset_name=args['input_dataset_name'],
        dataset_version=args['input_dataset_version']
    )
    local_dataset_path = dataset.get_local_copy()

    # === Generate dataset.yaml dynamically ===
    dataset_yaml_path = os.path.join(local_dataset_path, "dataset.yaml")
    if not os.path.exists(dataset_yaml_path):
        raise FileNotFoundError(f"dataset.yaml not found in dataset path: {dataset_yaml_path}")

    print(f"Using dataset.yaml from: {dataset_yaml_path}")

    # with open(dataset_yaml_path, 'r') as f:
    #     dataset_yaml_content = yaml.safe_load(f)
    #     print("===== dataset.yaml content =====")
    #     print(yaml.dump(dataset_yaml_content, sort_keys=False))
    #     print("================================")
    
    # print(f"Dataset path: {local_dataset_path}")
    # print(f"Files in dataset path: {os.listdir(os.path.join(local_dataset_path, 'labels'))}")

    # === Load YOLO model and start training ===
    model = YOLO(args['model_arch'])

    results = model.train(
        data=dataset_yaml_path,
        epochs=args['epochs'],
        imgsz=args['img_size'],
        batch=args['batch'],
        lr0=args['learning_rate'],
        device=0,
        augment=True,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=10.0, translate=0.1, scale=0.5, shear=2.0,
        patience=40,
        conf=0.25 
    )

    # === Logging training metrics ===
    # === Logging per-epoch metrics from results.csv ===
    save_dir = getattr(model.trainer, 'save_dir', './runs/train/exp')
    metrics_file = os.path.join(save_dir, 'results.csv')
    
    if os.path.exists(metrics_file):
        import pandas as pd
        df = pd.read_csv(metrics_file)
        for idx, row in df.iterrows():
            if 'train/box_loss' in row:
                logger.report_scalar("Loss", "box_loss", row['train/box_loss'], iteration=idx)
            if 'train/cls_loss' in row:
                logger.report_scalar("Loss", "cls_loss", row['train/cls_loss'], iteration=idx)
            if 'train/dfl_loss' in row:
                logger.report_scalar("Loss", "dfl_loss", row['train/dfl_loss'], iteration=idx)

    # === Upload best.pt as ClearML artifact ===
    best_model_path = os.path.join(save_dir, "weights", "best.pt")
    if os.path.exists(best_model_path):
        task.upload_artifact("trained_model", artifact_object=best_model_path)
        print(f"Model uploaded: {best_model_path}")
    else:
        print(f"[Warning] best.pt not found at: {best_model_path}")

    print("Training complete. Model + metrics logged.")

if __name__ == "__main__":
    Training()