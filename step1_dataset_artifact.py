from clearml import Task, Dataset

# Initialize ClearML Task
task = Task.init(project_name="CryptoSeek", task_name="Step 1 - Dataset Artifact")

args = {
    'input_dataset_version': '1.0.0',
    'output_tag': ['step1', 'v1.0'],
    'output_id': '57a2ad12e6a3428fae3317d1cb6f12b5',
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
    
check, temp = FindMatchedDataset(args['output_id'])

if check:
    print(f'SKIP Dataset: {temp.id}')
    print(f'Dataset version: {temp.version}')
    task.upload_artifact('dataset_id', temp.id)
    task.upload_artifact('dataset_version', temp.version)
else:
    original_dataset = Dataset.get(
        dataset_name='Resized_Cityscapes',
        dataset_project='CryptoSeek',
        dataset_version=args['input_dataset_version']
    )

    new_dataset = Dataset.create(
        dataset_name='Resized_Cityscapes',
        dataset_project='CryptoSeek',
        parent_datasets=[original_dataset.id]
    )

    new_dataset.add_tags(args['output_tag'])
    new_dataset.finalize(auto_upload=True)
    task.upload_artifact('dataset_id', new_dataset.id)
    task.upload_artifact('dataset_version', new_dataset.version)
    print(f'BUILD Dataset: {new_dataset.id}')
    print('Uploading artifacts in Step 1')
    print("Step 1 Done.")