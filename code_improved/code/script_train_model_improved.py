print("\n\n I need some time to get started, please wait....\n\n")
import os
import sys
import platform
import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing import image
from sklearn.preprocessing import LabelEncoder
import pickle
import random
import albumentations as A
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow.keras.callbacks import Callback

# ==== Platform path settings ====

os_name = platform.system()
if os_name == "Linux":
    data_root = "/Data/Python/AI/level_detection/split_dataset"
elif os_name == "Windows":
    data_root = r"D:\Data\Python\AI\level_detection\split_dataset"

train_root = os.path.join(data_root, "train")
validation_root = os.path.join(data_root, "validation")

# ==== Files to generate ====

keras_model_path = "model.keras"
encoders_path = "label_encoders.pkl"
tflite_model_path = "model.tflite"

# ==== Picture sizes ====

image_height = 100
image_width = 25

# ==== Training parameters ====

max_epochs = 20
validation_batch_size = 32
train_batch_size = 32
# Avoid taking too long and overfitting......
# Stop training when validation results get worse:
validation_losses_in_a_row = 5
# Stop training when all the next validation results are reached:
val_color_accuracy_target=1
val_perc_accuracy_target=1

# ==== Stop training when validation results get worse or validation results are good enough ====

# EarlyStopping is a standard function we activate like this:

early_stop_val_loss = EarlyStopping(monitor='val_loss', patience=validation_losses_in_a_row, restore_best_weights=True)

# Stop when validation of color accuracy target and  perc accuracy target requirements are met
# EarlyStopOnPerfectAccuracy is a custom function which we define and activate.

class EarlyStopOnPerfectAccuracy(Callback):
    def __init__(self, val_color_accuracy_target=1.0, val_perc_accuracy_target=1.0):
        super().__init__()
        self.val_color_accuracy_target = val_color_accuracy_target
        self.val_perc_accuracy_target = val_perc_accuracy_target

    def on_epoch_end(self, epoch, logs=None):
        # Get values from the logs
        color_accuracy = logs.get('val_color_accuracy')
        perc_accuracy = logs.get('val_perc_accuracy')

        # Stop training when both are perfect enough
        if color_accuracy >= self.val_color_accuracy_target and perc_accuracy >= self.val_perc_accuracy_target:
            print(f"\nTraining stopped after epoch {epoch+1} validation criteria:\n\n  val_color_accuracy: {color_accuracy:6.4f}\n\n  val_perc_accuracy : {perc_accuracy:6.4f}.\nEnd results:\n")
            self.model.stop_training = True

early_stop_perfect = EarlyStopOnPerfectAccuracy(val_color_accuracy_target=val_color_accuracy_target, val_perc_accuracy_target=val_perc_accuracy_target)

# ==== Collect list for colors and percentages and list for every 'color_perc' directory ====

def get_labels_from_dir_name(dir_name):
    try:
        color, perc = dir_name.split("_")
        return color, perc
    except ValueError:
        return None

def collect_labels_and_dirs(directory):
    all_labels = []
    all_dirs = []
    for dname in os.listdir(directory):
        labels = get_labels_from_dir_name(dname)
        if labels:
            all_labels.append(labels)
            all_dirs.append(dname)
    return all_labels, all_dirs

all_labels, all_dirs = collect_labels_and_dirs(train_root)

# Create encoders for the labels

le_color = LabelEncoder()
le_perc = LabelEncoder()

# Split the list in colors and percentages lists

colors = [l[0] for l in all_labels]
percs = [l[1] for l in all_labels]

# Encode the colors and percentages into the classes of their LabelEncoder

le_color.fit(colors)
le_perc.fit(percs)

# Note that the labels are not in the same order as in the image generation script.
# (directories were found in alphabetical order in collect_labels_and_dirs above)

#print(type(le_color))
#print(type(le_color.classes_))
print("Train for classes:\n")
print(le_color.classes_)
print()
print(le_perc.classes_)
print()

n_color = len(le_color.classes_)
n_perc = len(le_perc.classes_)

# ==== ImageLabelGenerator ====

def print_progress_bar(header, iteration, total, length=30):
    # a progress bar shown during validation
    if iteration == 0:
        print()  # new line at end
    percent = int(100 * (iteration + 1) / total)
    if percent == 0:
        return
    filled_length = int(length * (iteration + 1) / total)
    bar_color = '\033[0;32m'  # light green 
    perc_color = '\033[1;97m'  # bold white
    reset = '\033[0m'
    bar = bar_color + '━' * filled_length + reset + '━' * (length - filled_length)
#    or use a simple bar :
#    bar = '█' * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{header}: {bar} {perc_color}{percent}%{reset}')
    sys.stdout.flush()

    if iteration + 1 == total:
        print()  # new line at end
        
# ImageLabelGenerator below creates batches of:
#   numpy array with image data     normalized RGB pixel values (values between 0 and 1)
#   numpy array with colors         indexes 0, 1, 2 for values ['blue' 'green' 'red']
#   numpy array with percentages    indexes 0, 1, .. 99 for values
#     ['1' '10' '100' '11' '12' '13' '14' '15' '16' '17' '18' '19' '2' '20' '21' .. '99']

# The image data is the input of the network
# The colors data explains which color-output-neuron should get the highest output
# The percentages data explains which percentages-output-neuron should get the highest output

#
# An instance for training and one for validation are created from this
#
#   Each instance controls the ordering and handling of its complete dataset and
#   splits up its complete data set in batches which are read in memory
#

class ImageLabelGenerator(Sequence):
    def __init__(self, phase, root, dirs, batch_size, label_encoders, augment_type='skip', shuffle=True, validation_progress_bar=False, **kwargs):
        super().__init__(**kwargs)

        self.phase = phase
        self.augment_type = augment_type
        self.root = root
        self.dirs = dirs
        self.batch_size = batch_size
        self.label_encoders = label_encoders
        self.shuffle = shuffle
        self.validation_progress_bar = validation_progress_bar

        self.image_paths = self._load_image_paths()
        # shuffle all file names before the first epoch starts
        if self.shuffle:
            random.shuffle(self.image_paths)

        # define augmentation for training ( used in __getitem__ below )
        self.augment_train = A.Compose([
            A.Affine(rotate=(-3, 3), p=0.7), # maybe the camera is not exactly horizontal so rotate the image a bit
            A.GaussNoise(p=0.2),
            A.MotionBlur(p=0.2)
        ])
        # just to show that you can add label specific augmentation 
        self.augment_train_red = A.Compose([
            A.Affine(rotate=(-3, 3), p=0.7), # maybe the camera is not exactly horizontal so rotate the image a bit
            A.GaussNoise(p=0.1),             # for red a little less noise
            A.MotionBlur(p=0.1)              # for red a little less blur
        ])

        # just to show that you can use augmentation for validation         
        # define augmentation for validation ( used in __getitem__ below )
        self.augment_val = A.Compose([
            A.Affine(rotate=(-5, 5), p=0.7), # maybe the camera is not exactly horizontal so rotate the image a bit
            A.GaussNoise(p=0.2)
        ])

    def _load_image_paths(self):
        # create a list with full filenames inluding paths ( used in __init__ above )
        image_paths = []
        for dir in self.dirs:
            dir_path = os.path.join(self.root, dir)
            if os.path.isdir(dir_path):
                for fname in os.listdir(dir_path):
                    if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                        image_paths.append(os.path.join(dir_path, fname))
        return image_paths

    def __len__(self):
        # Tell tensorflow / keras how many batches there are
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, idx):

        # __getitem__ produces the next batch with the 3 numpy 

        if (not self.phase == 'training') and self.validation_progress_bar:
            print_progress_bar(f'Validation', idx, self.__len__(),16)

        # batch_files = all file names for this batch
        total_images = len(self.image_paths)
        num_batches = total_images // self.batch_size
        idx = idx % num_batches  # Ensure idx stays within num_batches
        start = idx * self.batch_size
        end = min((idx + 1) * self.batch_size, total_images) # Ensure end is not more than number of images
        batch_files = self.image_paths[start:end]

        # batch_images = list with the contents of the files
        # y_color, y_perc = lists with indexes to labels for corresponding batch_images
        batch_images = []
        y_color, y_perc = [], []
        
        # these are copied from what we created at the beginnning of the script
        le_color, le_perc = self.label_encoders

        # split each filepath to get the labels for the file
        # load the file
        # do some harm, augmentation, to the image
        # add the image contents to a list
        # add the label indexes to a list
        for f in batch_files:
            parent_dir = os.path.basename(os.path.dirname(f))
            try:
                color, perc = parent_dir.split("_")
            except ValueError:
                continue

            img = image.load_img(f, target_size=(image_height, image_width))
            img = image.img_to_array(img).astype("float32") / 255.0

            if self.augment_type == 'training':
                if color == 'red':
                    img = self.augment_train_red(image=img)["image"]
                else:
                    img = self.augment_train(image=img)["image"]
            if self.augment_type == 'validate':
                img = self.augment_val(image=img)["image"]

            batch_images.append(img)    # add image RGB pixels
            y_color.append(le_color.transform([color])[0])  # add color index
            y_perc.append(le_perc.transform([perc])[0])     # add percentage index

        # return the numpy arrays
        return np.array(batch_images), {
            'color': np.array(y_color),
            'perc': np.array(y_perc)
        }

    def on_epoch_end(self):
        # reshuffle all file ( used at end of each epoch )
        if self.shuffle:
            random.shuffle(self.image_paths)

# Just to show how you could add something at the begining of validation

class ValidationStartCallback(tf.keras.callbacks.Callback):
    def on_test_begin(self, logs=None):
        dummy = 1
#        print("\nValidation...")
        
# ==== Model ====

# Now without all the comments

inputs = layers.Input(shape=(image_height, image_width, 3))
x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2,2))(x)
#x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
#x = layers.MaxPooling2D((2,2))(x)
x = layers.Flatten()(x)
x = layers.Dense(n_color + n_perc, activation='relu')(x)
out_color = layers.Dense(n_color, activation='softmax', name='color')(x)
out_perc = layers.Dense(n_perc, activation='softmax', name='perc')(x)

model = Model(inputs=inputs, outputs=[out_color, out_perc])

model.compile(
    optimizer='adam',
    loss={
        'color': 'sparse_categorical_crossentropy',
        'perc': 'sparse_categorical_crossentropy'
    },
    metrics={
        'color': 'accuracy',
        'perc': 'accuracy'
    }
)

# ==== Generators ====

train_generator = ImageLabelGenerator(
    phase='training',
    root=train_root,
    dirs=all_dirs,
    batch_size=train_batch_size,
    label_encoders=(le_color, le_perc),
    augment_type='training',
    shuffle=True,        # validation with the same set in same order every time
    validation_progress_bar=True
)

validation_generator = ImageLabelGenerator(
    phase='validation',
    root=validation_root,
    dirs=all_dirs,
    batch_size=validation_batch_size,
    label_encoders=(le_color, le_perc),
    augment_type='skip',  # you could use 'validate' but that would change the validation data
    shuffle=False,        # validation with the same set in same order every time
    validation_progress_bar=True
)

# ==== Train ====

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=max_epochs,
    verbose=1,
    callbacks=[
        early_stop_val_loss,
        early_stop_perfect,
        ValidationStartCallback()
    ])
# Note that ValidationStartCallback() is added above but does not add anything

# ==== Save files ====
print(f"[INFO] Saving model and encoders... {keras_model_path}  {encoders_path}")
model.save(keras_model_path)

with open(encoders_path, "wb") as f:
    pickle.dump({
        'color': le_color,
        'perc': le_perc
    }, f)

model.summary()

input("Press enter to convert to tflite (please ignore all output)...")

# ==== Convert to TFLite ====

def quiet_model_conversion(model, tflite_model_path):
# Convert model to tflite and avoid as much screen output as possible
    with open(os.devnull, 'w') as devnull:
        sys.stdout = devnull
        sys.stderr = devnull
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = []
        converter.target_spec.supported_types = [tf.float32]
        tflite_model = converter.convert()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)

quiet_model_conversion(model, tflite_model_path)
