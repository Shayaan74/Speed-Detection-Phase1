import os
from collections import Counter

# Count images per class
dataset_path = "train/images"  # Adjust to your path
classes = {}

for img_file in os.listdir(dataset_path):
    # Your class label (from filename or label file)
    # This depends on how your dataset is organized
    pass

# Or if you have label files (.txt):
labels_path = "train/labels"
class_counts = Counter()

for label_file in os.listdir(labels_path):
    with open(os.path.join(labels_path, label_file)) as f:
        for line in f:
            class_id = int(line.split()[0])
            class_counts[class_id] += 1

print(class_counts)
