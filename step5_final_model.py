from clearml import Task, Logger, Dataset
from ultralytics import YOLO
import os
import json
import logging
import torch

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Init ClearML task ===
task = Task.init(project_name="CryptoSeek", task_name="Step 5 - Final Model Training", task_type=Task.TaskTypes.training)
logger.info("Task initialized")

# === Parameters from pipeline ===
args = {
    'input_dataset_id': None,
    'hpo_task_id': None,
    'model_arch': 'yolo11s.pt',
    'img_size': 640,
    'epochs': 100,
    'learning_rate': 0.001,
    'batch': 16,
}
args = task.connect(args)
logger.info(f"Connected parameters: {args}")

# === Load dataset ===
dataset_id = args.get('input_dataset_id')
if not dataset_id:
    raise ValueError("Missing input_dataset_id")
dataset = Dataset.get(dataset_id=dataset_id)
dataset_path = dataset.get_local_copy()
logger.info(f"Dataset loaded from: {dataset_path}")

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
logger.info(f"Using HPO parameters: {best_params}")

# === Load YOLO and train ===
from ultralytics import YOLO
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

# === Upload final model ===
best_model_path = os.path.join(getattr(model.trainer, 'save_dir', './runs/train/exp'), 'weights', 'best.pt')
if os.path.exists(best_model_path):
    task.upload_artifact("final_model", artifact_object=best_model_path)
    logger.info(f"Final model saved and uploaded from: {best_model_path}")
else:
    logger.warning(f"Final model not found at: {best_model_path}")

logger.info("Final model training completed successfully.")
