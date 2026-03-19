import os
import csv

# Paths
images_dir = "data/images"
labels_dir = "data/labels"
csv_dir = "data/CSVs"
csv_file = os.path.join(csv_dir, "dataset.csv")

# Make sure CSV folder exists
os.makedirs(csv_dir, exist_ok=True)

# Get list of image files
image_files = sorted(os.listdir(images_dir))

# Open CSV file
with open(csv_file, mode="w", newline="") as file:
    writer = csv.writer(file)

    # Write header
    writer.writerow(["images", "labels"])

    # Loop through images
    for img in image_files:

        # Get file name without extension
        name, ext = os.path.splitext(img)

        # Image path
        image_path = os.path.join(images_dir, img)

        # Corresponding label file (change .txt if needed)
        label_file = name + ".txt"
        label_path = os.path.join(labels_dir, label_file)

        # Check if label exists
        if os.path.exists(label_path):
            writer.writerow([image_path, label_path])
        else:
            print(f"Warning: No label for {img}")

print("dataset.csv created successfully!")
