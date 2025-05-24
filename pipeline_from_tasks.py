from clearml import PipelineController, PipelineDecorator, Task
import os

EXECUTION_QUEUE = 'pipeline'

def run_pipeline():
    pipe = PipelineController(
        name="CryptoSeek Full Pipeline",
        project="CryptoSeek",
        version="0.0.1",
        add_pipeline_tags=False
    )

    pipe.set_default_execution_queue(EXECUTION_QUEUE)  # Replace with your ClearML queue name

    # Step 1: Create dataset artifact
    pipe.add_step(
        name="stage_data",
        base_task_id='24a61ad5475b419cb613a017efbda512',
        # base_task_project="CryptoSeek",
        # base_task_name="Step 1 - Dataset Artifact",
        execution_queue=EXECUTION_QUEUE,
        parameter_override={
            "General/input_dataset_version": "1.0.0",
            "General/output_tag": ['step1', 'v1.0'],
            "General/output_id": '57a2ad12e6a3428fae3317d1cb6f12b5',
        },
    )

    # Step 2: Data preprocessing (depends on Step 1)
    pipe.add_step(
        name="stage_process",
        parents=["stage_data"],
        base_task_id='e4caf42d153d43cebf19a3063d3f431e',
        # base_task_project="CryptoSeek",
        # base_task_name="Step 2 - Dataset Preprocessing",
        execution_queue=EXECUTION_QUEUE,
        parameter_override={
            "General/input_dataset_project": 'CryptoSeek',  # Project name in ClearML
            "General/input_dataset_name": 'Resized_Cityscapes',  # Dataset name (v1.0.1 input)
            "General/input_dataset_version": '1.0.1',  # Dataset version to read from
            "General/input_width": 640,
            "General/input_height": 640,
            "General/output_dir": 'dataset',
            "General/output_tag": ['step2', 'v1.0'],
            "General/output_id": '9fa4d7e8172b4c70867e77849813f2a5',
            "General/classes": ["person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle", "traffic light", "traffic sign"],  # Target object classes
            "General/test_split_ratio": 0.1,  # Proportion of train data to use for test set
            "General/random_seed": 42  # Random seed for reproducibility
        },
    )

    # Step 3: Model training (depends on Step 2)
    pipe.add_step(
        name="stage_train",
        parents=["stage_process"],
        base_task_id='d8e90e791b3044f6b13576edc294ee24',
        # base_task_project="CryptoSeek",
        # base_task_name="Step 3 - Training Model",
        execution_queue=EXECUTION_QUEUE,
        parameter_override={
            "General/input_dataset_project": 'CryptoSeek',
            "General/input_dataset_name": 'Resized_Cityscapes',
            "General/input_dataset_version": '1.0.2',     
            "General/model_arch": 'yolo11s.pt',  # Model architecture (nano by default)
            "General/img_size": 640,
            "General/epochs": 200,
            "General/learning_rate": 0.001,
            "General/batch": 16
        },
    )

    pipe.add_step(
        name="stage_hpo",
        parents=["stage_train", "stage_process", "stage_data"],
        base_task_id='e5d11ac6954a44a48ee4efa6f83f6972',
        # base_task_project="AI_Studio_Demo",
        # base_task_name="HPO: Train Model",
        execution_queue=EXECUTION_QUEUE,
        parameter_override={
            "General/dataset_id": "${stage_process.parameters.General/output_id}",
            "General/test_queue": EXECUTION_QUEUE,
            "General/num_trials": 5,
            "General/time_limit_minutes": 30,
            "General/run_as_service": False,
            "General/base_train_task_id": "${stage_train.id}"
        }
    )

    # Choose one of the launch methods
    pipe.start_locally()  # Local debugging
    # pipe.start(queue="pipeline")  # For remote agent execution

    print("done")