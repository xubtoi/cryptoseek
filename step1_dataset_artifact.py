from clearml import Task, Dataset

# Initialize ClearML Task
task = Task.init(project_name="CryptoSeek", task_name="Step 1 - Dataset Artifact")

args = {
    'input_dataset_version': '1.0.0',
}
task.connect(args)

original_dataset =  Dataset.get(
    dataset_name='Resized_Cityscapes',
    dataset_project='CryptoSeek',
    dataset_version=args['input_dataset_version']
)

new_dataset = Dataset.create(
    dataset_name='Resized_Cityscapes',
    dataset_project='CryptoSeek',
    parent_datasets=[original_dataset.id]
)

new_dataset.finalize(auto_upload=True)

print('uploading artifacts in the background')

print("Step 1 Done.")