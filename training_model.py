# from clearml import Task
from ultralytics import YOLO
import os

# Initialize ClearML Task
# task = Task.init(project_name="CryptoSeek", task_name="Step 3 - Training Model")

# Define configurable arguments
args = {
    'dataset_task_id': '',  # Fill in Step2 Task ID
    'epochs': 30,           # Number of training epochs
    'batch_size': 16,       # Batch size
    'img_size': 512,        # Input image size
    'model_arch': 'yolo11n.pt',  # Model architecture (nano by default)
}
# task.connect(args)

# Download preprocessed dataset artifact
# dataset_task = Task.get_task(task_id=args['dataset_task_id'])
# dataset_path = dataset_task.artifacts['preprocessed_dataset'].get_local_copy()

dataset_path = "dataset/cityscapes_preprocessed"

if __name__ == "__main__":
    # Load YOLO model
    model = YOLO(args['model_arch'])

    # Start training
    model.train(
        data=os.path.join(dataset_path, 'data.yaml'),
        epochs=args['epochs'],
        imgsz=args['img_size'],
        batch=args['batch_size'],
    )

    print(f'Step 3 done: YOLO training complete ({args["epochs"]} epochs, batch {args["batch_size"]}, size {args["img_size"]})')