import os
import random
from tqdm import tqdm


def make_folder(folder):
    # Check existence and make
    if not os.path.exists(folder):
        os.makedirs(folder)


# Split the folder of training images into train/val/test sets randomly

# Inputs:
# Folder of images
# Percentage of the split
input_folder = "data/train_images_resized_matlab"


# Output folder with subfolders (train/val/test)
train_folder = "data/train_images_resized_matlab/train"
val_folder = "data/train_images_resized_matlab/validation"
test_folder = "data/train_images_resized_matlab/test"

make_folder(train_folder)
make_folder(val_folder)
make_folder(test_folder)

# Loop through and get all the images, with jpg extension
imglist = []
for root, folders, files in os.walk(input_folder):
    for f in files:
        if ".jpg" in f:
            # The file is an image
            imglist.append(os.path.join(root, f))
        # break

train_pct = 0.7
val_pct = 0.15

n_train = round(train_pct * len(imglist))
n_val = round(val_pct * len(imglist))
n_test = len(imglist) - n_train - n_val

# Create the subsamples
random.seed(0)
train_imgs = random.sample(imglist, n_train)
imgs_leftover = list(set(imglist) - set(train_imgs))
val_imgs = random.sample(imgs_leftover, n_val)
test_imgs = list(set(imgs_leftover) - set(val_imgs))

# Move the images into the subsamples
for img in tqdm(train_imgs):
    # Replace the name
    new_folder = img.replace(input_folder, train_folder)
    os.rename(img, new_folder)

for img in tqdm(val_imgs):
    # Replace the name
    new_folder = img.replace(input_folder, val_folder)
    os.rename(img, new_folder)

for img in tqdm(test_imgs):
    # Replace the name
    new_folder = img.replace(input_folder, test_folder)
    os.rename(img, new_folder)
