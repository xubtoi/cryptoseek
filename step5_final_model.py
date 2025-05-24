import os
os.environ.pop("MPLBACKEND", None)

import matplotlib
matplotlib.use("Agg")

from clearml import Task, Logger, Dataset
from ultralytics import YOLO
import os
import json
import logging
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import pandas as pd

# Setup logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

def Training():
    # === Init ClearML task ===
    task = Task.init(project_name="CryptoSeek", task_name="Step 5 - Final Model Training", task_type=Task.TaskTypes.training)
    logger = Logger.current_logger()
    # logger.info("Task initialized")

    # === Parameters from pipeline ===
    args = {
        'input_dataset_id': '9fa4d7e8172b4c70867e77849813f2a5',
        'hpo_task_id': '8b9c6a6557f24027b1cb589b80d43dc9',
        'model_arch': 'yolo11s.pt',
        'img_size': 640,
        'epochs': 100,
        'learning_rate': 0.001,
        'batch': 16,
    }
    args = task.connect(args)
    # logger.info(f"Connected parameters: {args}")

    # === Load dataset ===
    dataset_id = args.get('input_dataset_id')
    if not dataset_id:
        raise ValueError("Missing input_dataset_id")
    dataset = Dataset.get(dataset_id=dataset_id)
    dataset_path = dataset.get_local_copy()
    # logger.info(f"Dataset loaded from: {dataset_path}")

    # === Load best hyperparameters from HPO task ===
    hpo_task_id = args.get('hpo_task_id')
    if not hpo_task_id:
        raise ValueError("Missing hpo_task_id")

    hpo_task = Task.get_task(task_id=hpo_task_id)

    # Try to get best parameters from task parameters or artifact
    best_params = hpo_task.get_parameter("best_parameters")
    if not best_params and 'best_parameters' in hpo_task.artifacts:
        artifact_path = hpo_task.artifacts['best_parameters'].get_local_copy()
        with open(artifact_path, 'r') as f:
            best_params = json.load(f).get('parameters', {})

    if not best_params:
        raise ValueError("Best hyperparameters not found in HPO task")

    # === Override args with best parameters ===
    args['epochs'] = int(best_params.get('epochs', args['epochs']))
    args['batch'] = int(best_params.get('batch', args['batch']))
    args['learning_rate'] = float(best_params.get('learning_rate', args['learning_rate']))
    # logger.info(f"Using HPO parameters: {best_params}")

    # === Load YOLO and train ===
    model = YOLO(args['model_arch'])

    dataset_yaml_path = os.path.join(dataset_path, 'dataset.yaml')
    if not os.path.exists(dataset_yaml_path):
        raise FileNotFoundError("dataset.yaml not found in dataset directory")

    results = model.train(
        data=dataset_yaml_path,
        epochs=args['epochs'],
        imgsz=args['img_size'],
        batch=args['batch'],
        lr0=args['learning_rate'],
        device=0 if torch.cuda.is_available() else 'cpu',
        conf=0.25,
        patience=30,
        augment=True,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=10.0, translate=0.1, scale=0.5, shear=2.0
    )

    save_dir = getattr(model.trainer, 'save_dir', './runs/train/exp')
    metrics_file = os.path.join(save_dir, 'results.csv')
    
    if os.path.exists(metrics_file):
        df = pd.read_csv(metrics_file)
        for idx, row in df.iterrows():
            if 'train/box_loss' in row:
                logger.report_scalar("Loss", "box_loss", row['train/box_loss'], iteration=idx)
            if 'train/cls_loss' in row:
                logger.report_scalar("Loss", "cls_loss", row['train/cls_loss'], iteration=idx)
            if 'train/dfl_loss' in row:
                logger.report_scalar("Loss", "dfl_loss", row['train/dfl_loss'], iteration=idx)
            # Detection metrics
            if 'metrics/precision(B)' in row:
                logger.report_scalar("Metrics", "Precision", row['metrics/precision(B)'], iteration=idx)
            if 'metrics/recall(B)' in row:
                logger.report_scalar("Metrics", "Recall", row['metrics/recall(B)'], iteration=idx)
            if 'metrics/mAP50(B)' in row:
                logger.report_scalar("Metrics", "mAP@0.5", row['metrics/mAP50(B)'], iteration=idx)
            if 'metrics/mAP50-95(B)' in row:
                logger.report_scalar("Metrics", "mAP@0.5:0.95", row['metrics/mAP50-95(B)'], iteration=idx)
    
    logger.report_matplotlib_figure('Confusion Matrix', 'confusion_matrix', plt.gcf(), epoch)
    # === Upload final model ===
    best_model_path = os.path.join(save_dir, 'weights', 'best.pt')
    if os.path.exists(best_model_path):
        task.upload_artifact("final_model", artifact_object=best_model_path)
        print(f"Model uploaded: {best_model_path}")
        # logger.info(f"Final model saved and uploaded from: {best_model_path}")
    else:
        print(f"[Warning] best.pt not found at: {best_model_path}")
        # logger.warning(f"Final model not found at: {best_model_path}")

    print("Training complete. Model + metrics logged.")
    # logger.info("Final model training completed successfully.")

if __name__ == "__main__":
    Training()
