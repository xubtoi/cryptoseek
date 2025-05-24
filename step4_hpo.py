from clearml import Task, Dataset
from clearml.automation import HyperParameterOptimizer
from clearml.automation import UniformIntegerParameterRange, UniformParameterRange
import logging
import time
import json
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the HPO task
task = Task.init(project_name='CryptoSeek', task_name='Step 4 - Hyperparameter Optimisation', task_type=Task.TaskTypes.optimizer, reuse_last_task_id=False)

# Connect parameters
args = {
    'base_train_task_id': 'e66005a3afc54219a5f210a610fd8fac',  # Will be set from pipeline
    'num_trials': 5,  # Reduced from 10 to 3 trials
    'time_limit_minutes': 30,  # Reduced from 60 to 5 minutes
    'run_as_service': False,
    'test_queue': 'pipeline',  # Queue for test tasks
    'dataset_id': '9fa4d7e8172b4c70867e77849813f2a5',  # Will be set from pipeline
    'epochs': 200,  # Reduced from 50 to 20 epochs
    'batch_size': 16,  # Default batch size
    'learning_rate': 1e-3,  # Default learning rate
}
args = task.connect(args)
logger.info(f"Connected parameters: {args}")

# Execute the task remotely
# task.execute_remotely()

# # Get the dataset ID from pipeline parameters
# dataset_id = task.get_parameter('General/processed_dataset_id')  # Get from General namespace
# if not dataset_id:
#     # Try getting from args as fallback
#     dataset_id = args.get('processed_dataset_id')
#     logger.info(f"No dataset_id in General namespace, using from args: {dataset_id}")

# if not dataset_id:
#     # Use fixed dataset ID as last resort
#     dataset_id = "c0b726853a084f4c94e6a2661aa7d7fa"
#     logger.info(f"Using fixed dataset ID: {dataset_id}")

# logger.info(f"Using dataset ID: {dataset_id}")

# # Get the actual training model task
# try:
#     BASE_TRAIN_TASK_ID = args['base_train_task_id']
#     logger.info(f"Using base training task ID: {BASE_TRAIN_TASK_ID}")
# except Exception as e:
#     logger.error(f"Failed to get base training task ID: {e}")
#     raise

# # Verify dataset exists
# try:
#     dataset = Dataset.get(dataset_id=dataset_id)
#     logger.info(f"Successfully verified dataset: {dataset.name}")
# except Exception as e:
#     logger.error(f"Failed to verify dataset: {e}")
#     raise

# Create the HPO task
hpo_task = HyperParameterOptimizer(
    base_task_id=args['base_train_task_id'],
    hyper_parameters=[
        UniformIntegerParameterRange('epochs', min_value=20, max_value=40, step_size=20),
        UniformIntegerParameterRange('batch', min_value=16, max_value=32, step_size=8),  # Reduced range
        UniformParameterRange('learning_rate', min_value=1e-4, max_value=1e-2),  # Reduced range
    ],
    objective_metric_title='Metrics',
    objective_metric_series='mAP@0.5',
    objective_metric_sign='max',
    max_number_of_concurrent_tasks=2,
    optimization_time_limit=args['time_limit_minutes'] * 60,
    compute_time_limit=None,
    total_max_jobs=args['num_trials'],
    min_iteration_per_job=1,
    max_iteration_per_job=args['epochs'],
    pool_period_min=1.0,
    execution_queue=args['test_queue'],
    save_top_k_tasks_only=2,
    parameter_override={
        'input_dataset_id': args['dataset_id'],
        'General/input_dataset_id': args['dataset_id'],
        'epochs': args['epochs'],
        'General/epochs': args['epochs'],
        'batch': args['batch_size'],
        'General/batch': args['batch_size'],
        'learning_rate': args['learning_rate'],
        'General/learning_rate': args['learning_rate'],
    }
    # objective_metric_title='validation',
    # objective_metric_series='accuracy',
    # objective_metric_sign='max',
    # max_number_of_concurrent_tasks=2,
    # optimization_time_limit=args['time_limit_minutes'] * 60,
    # compute_time_limit=None,
    # total_max_jobs=args['num_trials'],
    # min_iteration_per_job=1,
    # max_iteration_per_job=args['num_epochs'],
    # pool_period_min=1.0,  # Reduced from 2.0 to 1.0 to check more frequently
    # execution_queue=args['test_queue'],
    # save_top_k_tasks_only=2,  # Reduced from 5 to 2
    # parameter_override={
    #     'processed_dataset_id': dataset_id,
    #     'General/processed_dataset_id': dataset_id,
    #     'test_queue': args['test_queue'],
    #     'General/test_queue': args['test_queue'],
    #     'num_epochs': args['num_epochs'],
    #     'General/num_epochs': args['num_epochs'],
    #     'batch_size': args['batch_size'],
    #     'General/batch_size': args['batch_size'],
    #     'learning_rate': args['learning_rate'],
    #     'General/learning_rate': args['learning_rate'],
    #     'weight_decay': args['weight_decay'],
    #     'General/weight_decay': args['weight_decay']
    # }
)

hpo_task.set_time_limit(in_minutes=args['time_limit_minutes'])

# Start the HPO task
logger.info("Starting HPO task...")
hpo_task.start()

# Wait for optimization to complete
logger.info(f"Waiting for optimization to complete (time limit: {args['time_limit_minutes']} minutes)...")
hpo_task.wait()

# Get the top performing experiments
try:
    top_exp = hpo_task.get_top_experiments(top_k=1)  # Get only the best experiment
    if top_exp:
        best_exp = top_exp[0]
        logger.info(f"Best experiment: {best_exp.id}")
        
        # Get the best parameters and result
        best_params = best_exp.get_parameters()
        metrics = best_exp.get_last_scalar_metrics()
        # best_accuracy = metrics['validation']['accuracy'] if metrics and 'validation' in metrics and 'accuracy' in metrics['validation'] else None
        best_map = metrics['Metrics']['mAP@0.5']['value'] if 'Metrics' in metrics and 'mAP@0.5' in metrics['Metrics'] else None
        
        # Log detailed information about the best experiment
        logger.info("Best hyperparameters:")
        for key, val in best_params.items():
            logger.info(f"  - {key}: {val}")
        logger.info(f"Best mAP@0.5: {best_map}")
        
        # Save best parameters and result
        best_results = {
            'parameters': best_params,
            'mAP@0.5': best_map
        }
        
        # # Save to a temporary file
        # temp_file = 'best_parameters.json'
        # with open(temp_file, 'w') as f:
        #     json.dump(best_results, f, indent=4)
        
        # # Upload as artifact
        # task.upload_artifact('best_parameters', temp_file)
        # logger.info(f"Saved best parameters with accuracy: {best_accuracy}")
        
        # # Also save as task parameters for easier access
        # task.set_parameter('best_parameters', best_params)
        # task.set_parameter('best_accuracy', best_accuracy)
        with open('best_parameters.json', 'w') as f:
            json.dump(best_results, f, indent=4)
        task.upload_artifact('best_parameters', 'best_parameters.json')

        task.set_parameter('best_parameters', best_params)
        task.set_parameter('best_map@0.5', best_map)
        
        logger.info("Best parameters saved as both artifact and task parameters")
    else:
        logger.warning("No experiments completed yet. This might be normal if the optimization just started.")
except Exception as e:
    logger.error(f"Failed to get top experiments: {e}")
    raise
finally:
    # Make sure background optimization stopped
    hpo_task.stop()
    logger.info("Optimizer stopped")