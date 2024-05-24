import os
import shutil

train_dir = "/home/multi-sy-15/PycharmProjects/BasicProject/kaggle/Cars_Dataset/train"
test_dir = "/home/multi-sy-15/PycharmProjects/BasicProject/kaggle/Cars_Dataset/test"
n_images_to_move = 50  # Specify the number of images to move

# Get the list of subdirectories in the train directory
train_subdirs = os.listdir(train_dir)

# Iterate over each subdirectory in the train directory
for subdir in train_subdirs:
    # Construct the source and destination paths
    src_path = os.path.join(train_dir, subdir)
    dest_path = os.path.join(test_dir, subdir)

    # Create the destination directory if it doesn't exist
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    # Get the list of image files in the source directory
    image_files = [f for f in os.listdir(src_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

    # Move up to n images from the source directory to the corresponding destination directory
    for i, image_file in enumerate(image_files):
        if i >= n_images_to_move:
            break
        src_image_path = os.path.join(src_path, image_file)
        dest_image_path = os.path.join(dest_path, image_file)
        shutil.move(src_image_path, dest_image_path)
        print(f"Moved {src_image_path} to {dest_image_path}")
