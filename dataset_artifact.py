from clearml import Task
import os
import random
import shutil
import zipfile

# Initialize ClearML Task
task = Task.init(project_name="CryptoSeek", task_name="Step 1 - Dataset Artifact")

# Define configurable arguments
args = {
    'num_train_samples': 200,   # How many train samples (from train/)
    'num_test_samples': 10,     # How many samples to hold out as "test" (from train/)
    'gtFine_dir': 'gtFine',
    'leftImg8bit_dir': 'leftImg8bit',
    'random_seed': 42,
}
task.connect(args)

# Define source and output directories
gtFine_train = os.path.join(args['gtFine_dir'], "train")
gtFine_val = os.path.join(args['gtFine_dir'], "val")
leftImg8bit_train = os.path.join(args['leftImg8bit_dir'], "train")
leftImg8bit_val = os.path.join(args['leftImg8bit_dir'], "val")

output_base_dir = "dataset/cityscapes_raw_sampled"

images_output_dir = os.path.join(output_base_dir, "images")
labels_output_dir = os.path.join(output_base_dir, "annotations")  # Raw polygons JSONs

# Create output folders
os.makedirs(output_base_dir, exist_ok=True)

# Function to collect samples
def collect_samples(gtFine_dir):
    samples = []
    for root, _, files in os.walk(gtFine_dir):
        for file in files:
            if file.endswith("_polygons.json"):
                base_name = file.replace("_gtFine_polygons.json", "")
                samples.append((root, base_name))
    return samples

# Collect train/val samples
train_samples_all = collect_samples(gtFine_train)
val_samples_all = collect_samples(gtFine_val)

# Fix random seed
random.seed(args['random_seed'])

# Sample train set
sampled_train = random.sample(train_samples_all, min(args['num_train_samples'], len(train_samples_all)))

# Sample small test set from train
sampled_test = random.sample(sampled_train, min(args['num_test_samples'], len(sampled_train)))

# After test samples are selected, remove them from train
sampled_train = [sample for sample in sampled_train if sample not in sampled_test]

# Split assignments
splits = {
    'train': sampled_train,
    'val': val_samples_all,
    'test': sampled_test,
}

# Copy files
for split_name, split_samples in splits.items():
    img_out_dir = os.path.join(output_base_dir, split_name, "images")
    ann_out_dir = os.path.join(output_base_dir, split_name, "annotations")
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(ann_out_dir, exist_ok=True)

    for root, base_name in split_samples:
        city = os.path.basename(root)
        split_type = "train" if "train" in root else "val"
        
        json_path = os.path.join(args['gtFine_dir'], split_type, city, base_name + "_gtFine_polygons.json")
        image_path = os.path.join(args['leftImg8bit_dir'], split_type, city, base_name + "_leftImg8bit.png")

        if os.path.exists(json_path):
            shutil.copy(json_path, ann_out_dir)
        if os.path.exists(image_path):
            shutil.copy(image_path, img_out_dir)

def zip_dir(folder_path, zip_path):
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, folder_path)
                zipf.write(full_path, arcname=rel_path)

zip_output_path = output_base_dir + ".zip"
zip_dir(output_base_dir, zip_output_path)

# Upload split dataset as artifact
task.upload_artifact('raw_split_dataset', artifact_object=zip_output_path)

print(f'Step 1 done: Train {len(splits["train"])}, Val {len(splits["val"])}, Test {len(splits["test"])} samples')