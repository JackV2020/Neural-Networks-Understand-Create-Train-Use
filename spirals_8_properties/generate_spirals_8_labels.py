import os
import sys
import random
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
from pathlib import Path
import math
from concurrent.futures import ThreadPoolExecutor
'''
filename = Path(os.path.join(root, f"{direction_label}_{bg_color_label}_{spiral_color_label}_{radius_steps_label}_{line_width_label}_{turns_label}_{angle_start_label}", 
                                 f"{random.randint(100000, 999999)}.png"))
'''
data_root = os.path.abspath(os.path.join(os.sep, "Data", "Python", "AI", "spirals", "split_dataset_8"))
train_root = os.path.join(data_root, "train")
validation_root =  os.path.join(data_root, "validation")
test_root =  os.path.join(data_root, "test")

roots = [ train_root, validation_root, test_root ]
images = [ 20, 4, 2]

image_width = 100
image_height = 100

angle_step_values = [ 0.01, 2]
angle_step_labels = ['archimedian', 'triangular']

direction_values = [ -1, 1]
direction_labels = ['clock', 'counter']

bg_color_values = ['red', 'green', 'blue']
bg_color_labels = ['red', 'green', 'blue']

spiral_color_values = ['cyan', 'magenta', 'yellow']
spiral_color_labels = ['cyan', 'magenta', 'yellow']

radius_steps_values = [2, 3]
radius_steps_labels = ['small', 'big' ]

line_width_values = [2, 6]
line_width_labels = ['thin', 'thick' ]

turns_values = [1, 2]
turns_labels = ['one', 'two']

# the 8 below is for the 8 angles 

tasks_to_create = 8 * (images[0] + images[1] + images[2]) * len(angle_step_values) * len(direction_values) * len(bg_color_values) * len(spiral_color_values) * len(radius_steps_values) * len (line_width_values) * len(turns_values)

num_threads = os.cpu_count()

def add_noise(img, amount=0.05):
    """Add salt-and-pepper noise"""
    arr = np.array(img)
    noise = np.random.rand(*arr.shape[:2])
    salt = noise < (amount / 2)
    pepper = noise > (1 - amount / 2)
    arr[salt] = [255, 255, 255]
    arr[pepper] = [0, 0, 0]
    return Image.fromarray(arr)

def mirror_angle_if_needed(angle, direction):
    if direction == -1:
        return (360 - angle) % 360
    return angle
    
def spiral(image_width, image_height, line_width, start_radius, radius_step, angle_start, angle_step, turns, direction='clock', background='white', linefill='white'):
    if direction == 'clock':
        direction_i = 1
    else:
        direction_i = -1
    image = Image.new("RGB", (image_width, image_height), background)
    draw = ImageDraw.Draw(image)

    # Starting point (centre of spiral)
    cx, cy = image_width // 2 , image_height // 2
    # or move center a bit
    cx, cy = image_width // 2 + random.randint(-2,2), image_height // 2 + random.randint(-2,2)

    # Generate points from starting angle 0
    points = []
    theta = 0
    # or move starting angle a bit
    theta = 0 + random.randint(-1, 1) / 10 / math.pi
    
    max_theta = turns * 2 * math.pi + theta # convert turns to angle
#    radius_step = radius_step + (random.random() - 0.5)
    while theta <= max_theta:
        r = start_radius + radius_step * theta  # Archimedes-spiral: r = a + bθ
        x = cx + r * math.cos(theta + angle_start)
        y = cy + r * math.sin(theta + angle_start) * direction_i
        points.append((x, y))
        theta += angle_step
    # Draw the small line segments
    draw.line(points, fill=linefill, width=line_width)    
    # Markeer het beginpunt
    if points:
        r = 3  # straal van de stip
        x0, y0 = cx, cy
#        draw.ellipse((x0 - r, y0 - r, x0 + r, y0 + r), fill='black')
        x1, y1 = x, y
#        draw.ellipse((x1 - r, y1 - r, x1 + r, y1 + r), fill='black')
#        draw.line( [(x0, y0) , (x1, y1)], fill=linefill, width=r)
    return image

def generate_image(angle_start_value, angle_step_value, direction_label, bg_color_value, spiral_color_value, radius_steps_value, line_width_value, turns_value, filename):

    spiral_image = spiral(image_width=image_width, image_height=image_height, line_width=line_width_value, 
                          start_radius=0, radius_step=radius_steps_value, angle_start=angle_start_value * 2 * math.pi / 360, 
                          angle_step=angle_step_value, turns=turns_value, direction=direction_label, background=bg_color_value, linefill=spiral_color_value)
#    spiral_image = add_noise(spiral_image, amount=0.02)

    filename.parent.mkdir(parents=True, exist_ok=True)
    try:
#        print(f"Save: {filename}")
        spiral_image.save(filename)
        return True
    except:
        print("Almost impossible but we hit an existing filename!")
        return False

workers_started = 0
def worker_task(args):
    global workers_started # shared between all task threads so needs to be global 
    root, angle_start_label, angle_start_value, angle_step_label, angle_step_value, direction_label, direction_value, bg_color_label, bg_color_value, spiral_color_label, spiral_color_value, radius_steps_label, radius_steps_value, line_width_label, line_width_value, turns_label, turns_value = args
    angle_start_label = mirror_angle_if_needed(angle_start_value, direction_value)

    filename = Path(os.path.join(root, f"{angle_step_label}_{direction_label}_{bg_color_label}_{spiral_color_label}_{radius_steps_label}_{line_width_label}_{turns_label}_{angle_start_label}", 
                                 f"{random.randint(100000, 999999)}.png"))

    if line_width_label == 'thin':
        radius_steps_value = radius_steps_value / 1.5

    workers_started += 1
    print_progress_bar("Tasks progress", workers_started, tasks_to_create, 50)
    
    return generate_image(angle_start_value, angle_step_value, direction_label, bg_color_value, spiral_color_value, radius_steps_value, line_width_value, turns_value, filename)

def print_progress_bar(header, iteration, total, length=30):
    # a progress bar
    if iteration == 0:
        print()  # new line at start
    percent = int(100 * (iteration + 1) / total)
    if percent == 0:
        return
    filled_length = int(length * (iteration + 1) / total)
    bar_color_1 = '\033[0;32m'  # light green 
    bar_color_2 = '\033[0;31m'  # light red 
    perc_color = '\033[1;37m'  # bold white
    reset = '\033[0m'
    bar = bar_color_1 + '━' * filled_length + bar_color_2 + '━' * (length - filled_length)
#    or use a simple bar :
#    bar = '█' * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{header}: {bar} {perc_color}{percent}%{reset}')
    sys.stdout.flush()

#    if iteration + 1 == total:
#        print()  # new line at end


def main():
    test=0
    skipped = 0
    created = 0
    tasks_created = 0
    print("Generate train, validate and test images")

    # Generate all tasks (for a batch of multiple parallel tasks)
    tasks = []

    for phase in range(len(roots)):
        root=roots[phase]   
        nr_images=images[phase]    

        for t in range(len(turns_values)):
            turns_value = turns_values[t]     
            turns_label = turns_labels[t]     
                       
            for lw in range(len(line_width_values)):
                line_width_value = line_width_values[lw]
                line_width_label = line_width_labels[lw]
    
                for rs in range(len(radius_steps_values)):
                    radius_steps_value = radius_steps_values[rs]
                    radius_steps_label = radius_steps_labels[rs]
                
                    for bg in range(len(bg_color_values)):
                        bg_color_value = bg_color_values[bg]
                        bg_color_label = bg_color_labels[bg]

                        for spc in range(len(spiral_color_values)):
                            spiral_color_value = spiral_color_values[spc]
                            spiral_color_label = spiral_color_labels[spc]

                            for d in range(len(angle_step_values)):
                                angle_step_value = angle_step_values[d]   
                                angle_step_label = angle_step_labels[d]          

                                for d in range(len(direction_values)):
                                    direction_value = direction_values[d]   
                                    direction_label = direction_labels[d]          

                                    for angle_start_base in range(0, 360 // 45 ):
                                        angle_start_value = angle_start_base * 45
                                        angle_start_label = angle_start_value

                                        for n in range(0,nr_images):

    #                                                tasks.append((root, angle_start, direction, bg, lf))
                                            tasks.append((root, angle_start_label, angle_start_value, angle_step_label, angle_step_value, direction_label, direction_value, bg_color_label, bg_color_value, spiral_color_label, spiral_color_value, radius_steps_label, radius_steps_value, line_width_label, line_width_value, turns_label, turns_value))
                                            tasks_created += 1
    print(f"\nCreated list with {tasks_created} tasks\n")
    # Use aThreadPoolExecutor to spread the tasks over the threads
    with ThreadPoolExecutor(max_workers=num_threads) as executor:  # use num_threads threads
        results = list(executor.map(worker_task, tasks))

    created = sum(results)
    skipped = len(results) - created
    import time
    time.sleep(1)
    print(f"\n\nCreated {created} images, skipped {skipped}\n")

if __name__ == "__main__":
    main()

