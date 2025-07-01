import os
import random
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
from tqdm import tqdm
from pathlib import Path

import platform

os_name = platform.system()

if os_name == "Linux":
    data_root = "/Data/Python/AI/level_detection/split_dataset"
elif os_name == "Windows":
    # Windows needs the r before the string
    data_root = r"D:\Data\Python\AI\level_detection\split_dataset"

train_root = os.path.join(data_root, "train")
validation_root =  os.path.join(data_root, "validation")
test_root =  os.path.join(data_root, "test")

train_images_per_perc = 80
validation_images_per_perc = 20
test_images_per_perc = 5
image_height = 100
image_width = 25

colors = ['red', 'green', 'blue']

#
# You can also train with more colors however...............
#
# DO NOT DO THIS UNTIL YOU FINISHED THE FIRST validationS AFTER TRAINING
#
#colors = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta']
#

color_map = {
    'red': (255, 0, 0),
    'green': (0, 255, 0),
    'blue': (0, 0, 255),
    'yellow': (255, 255, 0),
    'cyan': (0, 255, 255),
    'magenta': (255, 0, 255)
}

os.makedirs(data_root, exist_ok=True)

def add_noise(img, amount=0.05):
    """Add salt-and-pepper noise"""
    arr = np.array(img)
    noise = np.random.rand(*arr.shape[:2])
    salt = noise < (amount / 2)
    pepper = noise > (1 - amount / 2)
    arr[salt] = [255, 255, 255]
    arr[pepper] = [0, 0, 0]
    return Image.fromarray(arr)

def random_background():
    """Return a random light background color"""
    base = random.randint(220, 255)
    return (base - random.randint(0, 20),
            base - random.randint(0, 20),
            base - random.randint(0, 20))
skipped = 0

print("Generate train images")
for a in range(train_images_per_perc):
    for c in range(len(colors)):
        color = colors[c]
        for perc in range(1,101):
            # Create a full size image with 1 color
            bg_color = random_background()
            img = Image.new("RGB", (image_width, image_height), bg_color)
            draw = ImageDraw.Draw(img)
            # Draw the lower rectangle with the color
            draw.rectangle([0 , 100 - perc, image_width, image_height], fill=color)
            # Add some noise
            if a >  train_images_per_perc * 0.1 :
                img = add_noise(img, amount=0.03)
            filename = Path(os.path.join(train_root, color+"_"+ str(perc), str(random.randint(100000,999999)) + ".png"))
            # Save the file
            filename.parent.mkdir(parents=True, exist_ok=True)
            try:
#                print(f"Saving {filename}")
                img.save(filename)
            except:
                print("Almost impossible but we hit a existing filename!")
                skipped = skipped + 1
                
print("Generate validation images")
for a in range(validation_images_per_perc):
    for c in range(len(colors)):
        color = colors[c]
        for perc in range(1,101):
            # Create a full size image with 1 color
            bg_color = random_background()
            img = Image.new("RGB", (image_width, image_height), bg_color)
            draw = ImageDraw.Draw(img)
            # Draw the lower rectangle with the color
            draw.rectangle([0 , 100 - perc, image_width, image_height], fill=color)
            # Add some noise
            if a >  validation_images_per_perc * 0.1 :
                img = add_noise(img, amount=0.03)
            filename = Path(os.path.join(validation_root, color+"_"+ str(perc), str(random.randint(100000,999999)) + ".png"))
            # Save the file
            filename.parent.mkdir(parents=True, exist_ok=True)
            try:
#                print(f"Saving {filename}")
                img.save(filename)
            except:
                print("Almost impossible but we hit a existing filename!")
                skipped = skipped + 1

print("Generate test images")
for a in range(test_images_per_perc):
    for c in range(len(colors)):
        color = colors[c]
        for perc in range(1,101):
            # Create a full size image with 1 color
            bg_color = random_background()
            img = Image.new("RGB", (image_width, image_height), bg_color)
            draw = ImageDraw.Draw(img)
            # Draw the lower rectangle with the color
            draw.rectangle([0 , 100 - perc, image_width, image_height], fill=color)
            # Add some noise
            if a >  test_images_per_perc * 0.5 :
                img = add_noise(img, amount=0.03)
            filename = Path(os.path.join(test_root, color+"_"+ str(perc), str(random.randint(100000,999999)) + ".png"))
            # Save the file
            filename.parent.mkdir(parents=True, exist_ok=True)
            try:
#                print(f"Saving {filename}")
                img.save(filename)
            except:
                print("Almost impossible but we hit a existing filename!")
                skipped = skipped + 1

print(f"Skipped {skipped} files")

