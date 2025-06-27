import os
import random
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
from tqdm import tqdm

data_root = "/Data/Python/AI/level_detection/dataset"
images_per_perc = 10
image_height = 100
image_width = 25

colors = ['red', 'green', 'blue']

#
# You can also train with more colors however...............
#
# DO NOT DO THIS UNTIL YOU FINISHED THE FIRST TESTS AFTER TRAINING
#
colors = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta']
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
            
for a in range(images_per_perc):
    for c in range(len(colors)):
        color = colors[c]
        for i in range(101):
            perc = i
            label = f"{color}_{perc}"
            if perc == 0: # I put this in because I might want something with it anyway.
                path = f"{data_root}/blanco"
            else:
                path = f"{data_root}/{label}"
            os.makedirs(path, exist_ok=True)
            
            bg_color = random_background()
            img = Image.new("RGB", (image_width, image_height), bg_color)
            draw = ImageDraw.Draw(img)
            
            # Draw the lower rectangle=
            draw.rectangle([0 , 100 - perc, image_width, image_height], fill=color)
            
            if a > 1 : # add some random noise to the picture but skip first 2 pictures so we also have solids
                img = add_noise(img, amount=0.03)
            filename = f"{path}/{random.randint(100000,999999)}.png"
            img.save(filename)
