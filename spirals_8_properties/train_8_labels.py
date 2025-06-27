print("\n\n I need some time to get started, please wait....\n\n")
import os
import sys
import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing import image
from sklearn.preprocessing import LabelEncoder
import pickle
import random
import albumentations as A
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from PIL import Image
import cv2
# ==== Platform path settings ====

data_root = os.path.abspath(os.path.join(os.sep, "Data", "Python", "AI", "spirals", "split_dataset_8"))
train_root = os.path.join(data_root, "train")
validation_root = os.path.join(data_root, "validation")

# ==== Files to generate ====

keras_model_path = "model_8_labels.keras"
encoders_path = "label_encoders_8_labels.pkl"
tflite_model_path = "model_8_labels.tflite"

# ==== Training parameters ====

max_epochs = 20
validation_batch_size = 16
train_batch_size = 16
# Avoid taking too long and overfitting......
# Stop training when validation results get worse:
validation_losses_in_a_row = 5
# Stop training when all the next validation results are reached:
val_label_1_accuracy_target=1
val_label_2_accuracy_target=1
val_label_3_accuracy_target=1
val_label_4_accuracy_target=1
val_label_5_accuracy_target=1
val_label_6_accuracy_target=1
val_label_7_accuracy_target=1
val_label_8_accuracy_target=1

# ==== Stop training when validation results get worse or validation results are good enough ====

# EarlyStopping is a standard function we activate like this:

early_stop_val_loss = EarlyStopping(monitor='val_loss', patience=validation_losses_in_a_row, restore_best_weights=True)

# Stop when validation of label_1 accuracy target and  label_2 accuracy target requirements are met
# EarlyStopOnPerfectAccuracy is a custom function which we define and activate.

class EarlyStopOnPerfectAccuracy(Callback):
    def __init__(self, val_label_1_accuracy_target=1.0, val_label_2_accuracy_target=1.0, val_label_3_accuracy_target=1.0, val_label_4_accuracy_target=1.0, val_label_5_accuracy_target=1.0, val_label_6_accuracy_target=1.0, val_label_7_accuracy_target=1.0, val_label_8_accuracy_target=1.0):
        super().__init__()
        self.val_label_1_accuracy_target = val_label_1_accuracy_target
        self.val_label_2_accuracy_target = val_label_2_accuracy_target
        self.val_label_3_accuracy_target = val_label_3_accuracy_target
        self.val_label_4_accuracy_target = val_label_4_accuracy_target
        self.val_label_5_accuracy_target = val_label_5_accuracy_target
        self.val_label_6_accuracy_target = val_label_6_accuracy_target
        self.val_label_7_accuracy_target = val_label_7_accuracy_target
        self.val_label_8_accuracy_target = val_label_8_accuracy_target

    def on_epoch_end(self, epoch, logs=None):
        # Get values from the logs
        label_1_accuracy = logs.get('val_label_1_accuracy')
        label_2_accuracy = logs.get('val_label_2_accuracy')
        label_3_accuracy = logs.get('val_label_3_accuracy')
        label_4_accuracy = logs.get('val_label_4_accuracy')
        label_5_accuracy = logs.get('val_label_5_accuracy')
        label_6_accuracy = logs.get('val_label_6_accuracy')
        label_7_accuracy = logs.get('val_label_7_accuracy')
        label_8_accuracy = logs.get('val_label_8_accuracy')

        # Stop training when both are perfect enough
        if  label_1_accuracy >= self.val_label_1_accuracy_target and \
          label_2_accuracy >= self.val_label_2_accuracy_target and \
          label_3_accuracy >= self.val_label_3_accuracy_target and \
          label_4_accuracy >= self.val_label_4_accuracy_target and \
          label_5_accuracy >= self.val_label_5_accuracy_target and \
          label_6_accuracy >= self.val_label_6_accuracy_target and \
          label_6_accuracy >= self.val_label_7_accuracy_target and \
          label_7_accuracy >= self.val_label_8_accuracy_target:
            print(f"\n\nTraining stopped after epoch {epoch+1} validation criteria reached\n\n")
            self.model.stop_training = True

early_stop_perfect = EarlyStopOnPerfectAccuracy(val_label_1_accuracy_target=val_label_1_accuracy_target, \
    val_label_2_accuracy_target=val_label_2_accuracy_target, \
    val_label_3_accuracy_target=val_label_3_accuracy_target, \
    val_label_4_accuracy_target=val_label_4_accuracy_target, \
    val_label_5_accuracy_target=val_label_5_accuracy_target, \
    val_label_6_accuracy_target=val_label_6_accuracy_target, \
    val_label_7_accuracy_target=val_label_7_accuracy_target, \
    val_label_8_accuracy_target=val_label_8_accuracy_target )

# ==== Collect list for label_1s,label_2s... and list for every 'label_1'_'label_2'... directory ====

def get_labels_from_dir_name(dir_name):
    try:
        label_1, label_2, label_3, label_4, label_5, label_6, label_7, label_8 = dir_name.split("_") # label_2 should be a color
        return label_1, label_2, label_3, label_4, label_5, label_6, label_7, label_8
    except ValueError:
        return None
first_directory = 'x'
def collect_labels_and_dirs(directory):
    all_labels = []
    all_dirs = []
    global first_directory
    for dname in os.listdir(directory):
        labels = get_labels_from_dir_name(dname)
        if labels:
            all_labels.append(labels)
            all_dirs.append(dname)
            if first_directory == 'x':
                first_directory = dname
    return all_labels, all_dirs

all_labels, all_dirs = collect_labels_and_dirs(train_root)
all_labels = list(set(all_labels))

# get image width and height

def get_image_dimenstions(train_root,all_labels):
    global first_directory
    first_directory=train_root + "/" + first_directory
    for file_name in os.listdir(first_directory):
        # Volledig pad naar het bestand
        image_path = os.path.join(first_directory, file_name)
        # Open de afbeelding
        with Image.open(image_path) as img:
            width, height = img.size
        break  # Stop de loop zodra we de eerste afbeelding hebben gevondensys.exit(0)
    return width, height

image_width, image_height = get_image_dimenstions(train_root,all_labels)

# Create encoders for the labels

le_label_1 = LabelEncoder()
le_label_2 = LabelEncoder()
le_label_3 = LabelEncoder()
le_label_4 = LabelEncoder()
le_label_5 = LabelEncoder()
le_label_6 = LabelEncoder()
le_label_7 = LabelEncoder()
le_label_8 = LabelEncoder()

# Split the all_labels list in labels_1, labels_2... lists

labels_1 = [l[0] for l in all_labels]
labels_2 = [l[1] for l in all_labels]
labels_3 = [l[2] for l in all_labels]
labels_4 = [l[3] for l in all_labels]
labels_5 = [l[4] for l in all_labels]
labels_6 = [l[5] for l in all_labels]
labels_7 = [l[6] for l in all_labels]
labels_8 = [l[7] for l in all_labels]

# Encode the labels_1, labels_2... into the classes of their LabelEncoder

le_label_1.fit(labels_1)
le_label_2.fit(labels_2)
le_label_3.fit(labels_3)
le_label_4.fit(labels_4)
le_label_5.fit(labels_5)
le_label_6.fit(labels_6)
le_label_7.fit(labels_7)
le_label_8.fit(labels_8)

# Note that the labels are not in the same order as in the image generation script.
# (directories were found in alphabetical order in collect_labels_and_dirs above)

#print(type(le_label_1))
#print(type(le_label_1.classes_))
print("Train for label classes:\n")
print(le_label_1.classes_)
print()
print(le_label_2.classes_)
print()
print(le_label_3.classes_)
print()
print(le_label_4.classes_)
print()
print(le_label_5.classes_)
print()
print(le_label_6.classes_)
print()
print(le_label_7.classes_)
print()
print(le_label_8.classes_)
print()

n_label_1 = len(le_label_1.classes_)
n_label_2 = len(le_label_2.classes_)
n_label_3 = len(le_label_3.classes_)
n_label_4 = len(le_label_4.classes_)
n_label_5 = len(le_label_5.classes_)
n_label_6 = len(le_label_6.classes_)
n_label_7 = len(le_label_7.classes_)
n_label_8 = len(le_label_8.classes_)

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
#   numpy array with label_1        indexes 0, 1, 2 like for values ['blue' 'green' 'red']
#   numpy array with label_2        indexes 0, 1, .. 99 like for values
#     ['1' '10' '100' '11' '12' '13' '14' '15' '16' '17' '18' '19' '2' '20' '21' .. '99']
#   numpy array ...

# The image data is the input of the network
# The label_1 data explains which label_1-output-neuron should get the highest output
# The label_2 data explains which label_2-output-neuron should get the highest output
# ...


# Custom Rotation transform
class RandomRotateCustom(A.ImageOnlyTransform):
    def __init__(self, p=1.0):
        super().__init__(p)
    
    def apply(self, image, **params):
        # Kies willekeurig aantal graden
        angles = [0, 45, 90, 135, 180, 225, 270, 315 ]
        angle = random.choice(angles)
        # Rotatie toepassen
        return np.rot90(image, k=round(angle / 90,0))

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

        # define augmentation for default training ( used in __getitem__ below )
        self.augment_train = A.Compose([
#            A.Affine(rotate=(-3, 3), p=0.7), # maybe the camera is not exactly horizontal so rotate the image a bit
            A.GaussNoise(p=0.1),           # 10% kans op GaussNoise
            A.MotionBlur(p=0.1),            # 10% kans op MotionBlur
#            A.HorizontalFlip(p=0.5),  # 50% kans op horizontale flip
#            A.VerticalFlip(p=0.5),    # 50% kans op verticale flip
#            RandomRotateCustom(p=1.0),      # Altijd rotatie (0, 90, 180, 270)
        ])

#            A.Rotate(limit=2, p=0.7, border_mode=cv2.BORDER_CONSTANT, value=(255, 0, 0)),
        self.augment_train_red = A.Compose([
            A.GaussNoise(p=0.1),           # 10% kans op GaussNoise
            A.MotionBlur(p=0.1),            # 10% kans op MotionBlur
#            A.HorizontalFlip(p=0.5),  # 50% kans op horizontale flip
#            A.VerticalFlip(p=0.5),    # 50% kans op verticale flip
#            RandomRotateCustom(p=1.0),      # Altijd rotatie (0, 90, 180, 270)
        ])
        self.augment_train_green = A.Compose([
            A.GaussNoise(p=0.1),           # 10% kans op GaussNoise
            A.MotionBlur(p=0.1),            # 10% kans op MotionBlur
#            A.HorizontalFlip(p=0.5),  # 50% kans op horizontale flip
#            A.VerticalFlip(p=0.5),    # 50% kans op verticale flip
#            RandomRotateCustom(p=1.0),      # Altijd rotatie (0, 90, 180, 270)
        ])
        self.augment_train_blue = A.Compose([
            A.GaussNoise(p=0.1),           # 10% kans op GaussNoise
            A.MotionBlur(p=0.1),            # 10% kans op MotionBlur
#            A.HorizontalFlip(p=0.5),  # 50% kans op horizontale flip
#            A.VerticalFlip(p=0.5),    # 50% kans op verticale flip
#            RandomRotateCustom(p=1.0),      # Altijd rotatie (0, 90, 180, 270)
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
        # y_label_1, y_label_2 = lists with indexes to labels for corresponding batch_images
        batch_images = []
        y_label_1, y_label_2, y_label_3, y_label_4, y_label_5, y_label_6, y_label_7, y_label_8 = [], [], [], [], [], [], [], []
        
        # these are copied from what we created at the beginnning of the script
        le_label_1, le_label_2, le_label_3, le_label_4, le_label_5, le_label_6, le_label_7, le_label_8  = self.label_encoders

        # split each filepath to get the labels for the file
        # load the file
        # do some harm, augmentation, to the image
        # add the image contents to a list
        # add the label indexes to a list
        for f in batch_files:
            parent_dir = os.path.basename(os.path.dirname(f))
            try:
                label_1, label_2, label_3, label_4, label_5, label_6, label_7, label_8 = parent_dir.split("_") # label_2 should be a color
            except ValueError:
                continue

            img = image.load_img(f, target_size=(image_height, image_width))
            img = image.img_to_array(img).astype("float32") / 255.0

            if self.augment_type == 'training':
                if label_2 == 'red':
                    img = self.augment_train_red(image=img)["image"]
                elif label_2 == 'green':
                    img = self.augment_train_green(image=img)["image"]
                elif label_2 == 'blue':
                    img = self.augment_train_blue(image=img)["image"]
                else:
                    img = self.augment_train(image=img)["image"]
            if self.augment_type == 'validate':
                img = self.augment_val(image=img)["image"]

            batch_images.append(img)    # add image RGB pixels
            y_label_1.append(le_label_1.transform([label_1])[0])  # add label_1 index
            y_label_2.append(le_label_2.transform([label_2])[0])  # add label_2 index
            y_label_3.append(le_label_3.transform([label_3])[0])  # add label_3 index
            y_label_4.append(le_label_4.transform([label_4])[0])  # add label_4 index
            y_label_5.append(le_label_5.transform([label_5])[0])  # add label_5 index
            y_label_6.append(le_label_6.transform([label_6])[0])  # add label_6 index
            y_label_7.append(le_label_7.transform([label_7])[0])  # add label_7 index
            y_label_8.append(le_label_8.transform([label_8])[0])  # add label_8 index

        # return the numpy arrays
        return np.array(batch_images), {
            'label_1': np.array(y_label_1),
            'label_2': np.array(y_label_2),
            'label_3': np.array(y_label_3),
            'label_4': np.array(y_label_4),
            'label_5': np.array(y_label_5),
            'label_6': np.array(y_label_6),
            'label_7': np.array(y_label_7),
            'label_8': np.array(y_label_8)
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
# https://www.upgrad.com/blog/basic-cnn-architecture/
# Now without all the comments

inputs = layers.Input(shape=(image_height, image_width, 3))
'''
# Feature extractor
x = layers.Rescaling(1./255)(inputs)
x = layers.Conv2D(32, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(64, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(128, (3, 3), activation='relu')(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
'''


# Feature extractor
#x = layers.Rescaling(1./255)(inputs)

x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)
x = layers.Conv2D(64, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)
x = layers.Conv2D(128, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)

x = layers.Flatten()(x)

x = layers.Dense((n_label_1 + n_label_2 + n_label_3 + n_label_4 + n_label_5 + n_label_6 + n_label_7 + n_label_8)*20, activation='relu')(x)
x = layers.Dropout(0.5)(x) #
x = layers.Dense(n_label_1 + n_label_2 + n_label_3 + n_label_4 + n_label_5 + n_label_6 + n_label_7 + n_label_8, activation='relu')(x)
out_label_1 = layers.Dense(n_label_1, activation='softmax', name='label_1')(x)
out_label_2 = layers.Dense(n_label_2, activation='softmax', name='label_2')(x)
out_label_3 = layers.Dense(n_label_3, activation='softmax', name='label_3')(x)
out_label_4 = layers.Dense(n_label_4, activation='softmax', name='label_4')(x)
out_label_5 = layers.Dense(n_label_5, activation='softmax', name='label_5')(x)
out_label_6 = layers.Dense(n_label_6, activation='softmax', name='label_6')(x)
out_label_7 = layers.Dense(n_label_7, activation='softmax', name='label_7')(x)
out_label_8 = layers.Dense(n_label_8, activation='softmax', name='label_8')(x)

model = Model(inputs=inputs, outputs=[out_label_1, out_label_2, out_label_3, out_label_4, out_label_5, out_label_6, out_label_7, out_label_8])

model.compile(
    optimizer='adam',
    loss={
        'label_1': 'sparse_categorical_crossentropy',
        'label_2': 'sparse_categorical_crossentropy',
        'label_3': 'sparse_categorical_crossentropy',
        'label_4': 'sparse_categorical_crossentropy',
        'label_5': 'sparse_categorical_crossentropy',
        'label_6': 'sparse_categorical_crossentropy',
        'label_7': 'sparse_categorical_crossentropy',
        'label_8': 'sparse_categorical_crossentropy'
    },    loss_weights={
        'label_1': 1.0,   # raise value to give more attention during training
        'label_2': 1.0,
        'label_3': 1.0,
        'label_4': 1.0,
        'label_5': 1.0,
        'label_6': 1.0,
        'label_7': 1.0,
        'label_8': 1.0
    },
    metrics={
        'label_1': 'accuracy',
        'label_2': 'accuracy',
        'label_3': 'accuracy',
        'label_4': 'accuracy',
        'label_5': 'accuracy',
        'label_6': 'accuracy',
        'label_7': 'accuracy',
        'label_8': 'accuracy'
    }
)

# ==== Generators ====

train_generator = ImageLabelGenerator(
    phase='training',
    root=train_root,
    dirs=all_dirs,
    batch_size=train_batch_size,
    label_encoders=(le_label_1, le_label_2, le_label_3, le_label_4, le_label_5, le_label_6, le_label_7, le_label_8),
    augment_type='training', # you can use 'training' to change the trainning data
    shuffle=True,
    validation_progress_bar=False
)

validation_generator = ImageLabelGenerator(
    phase='validation',
    root=validation_root,
    dirs=all_dirs,
    batch_size=validation_batch_size,
    label_encoders=(le_label_1, le_label_2, le_label_3, le_label_4, le_label_5, le_label_6, le_label_7, le_label_8),
    augment_type='training',  # you could use 'validate' but that would change the validation data
    shuffle=False,
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
#        ModelCheckpoint('best_model.h5', save_best_only=True),
        ValidationStartCallback()
    ])
# Note that ValidationStartCallback() is added above but does not add anything

# ==== Save files ====
print(f"[INFO] Saving model and encoders... {keras_model_path}  {encoders_path}")
model.save(keras_model_path)

with open(encoders_path, "wb") as f:
    pickle.dump({
        'label_1': le_label_1,
        'label_2': le_label_2,
        'label_3': le_label_3,
        'label_4': le_label_4,
        'label_5': le_label_5,
        'label_6': le_label_6,
        'label_7': le_label_7,
        'label_8': le_label_8
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
