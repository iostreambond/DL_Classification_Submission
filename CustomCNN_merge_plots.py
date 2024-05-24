from PIL import Image
import os


# Function to create a grid of images
def create_image_grid(images, grid_size, save_path):
    # Calculate width and height of the grid based on grid_size
    grid_width = images[0].width * grid_size[1]
    grid_height = images[0].height * grid_size[0]

    # Create a new blank image with the calculated grid size
    grid_image = Image.new('RGB', (grid_width, grid_height))

    # Paste each image into the grid
    for i in range(len(images)):
        row = i // grid_size[1]
        col = i % grid_size[1]
        grid_image.paste(images[i], (col * images[0].width, row * images[0].height))

    # Save the grid image
    grid_image.save(save_path)


# Folder paths
folders = ['plot_256', 'plot_320', 'plot_384', 'plot_448', 'plot_512']
output_folder = 'plot_all_merge'

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Iterate over images in plot_256 folder
for filename in os.listdir(folders[0]):
    if filename.endswith('.png') or filename.endswith('.jpg'):
        images = []
        # Open the corresponding image from each folder
        for folder in folders:
            img_path = os.path.join(folder, filename)
            if os.path.exists(img_path):
                img = Image.open(img_path)
                images.append(img)
            else:
                print(f"Image {filename} not found in folder {folder}")

        # Create the image grid and save it
        if len(images) == len(folders):
            output_path = os.path.join(output_folder, filename)
            create_image_grid(images, (3, 2), output_path)
            print(f"Grid image created and saved: {output_path}")
        else:
            print(f"Not all images found for {filename}")

print("All grid images created and saved.")
