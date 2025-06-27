# script_train_model.py
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.saving import save_model

import pickle

import platform

os_name = platform.system()

if os_name == "Linux":
    data_root = "/Data/Python/AI/level_detection/dataset"
elif os_name == "Windows":
    # Windows needs the r before the string
    data_root = r"D:\Data\Python\AI\level_detection\dataset"

model_path = "model.keras"
encoders_path = "label_encoders.pkl"
image_height = 100
image_width = 25

def load_images_and_labels(dataset_path):
    # Loop throug all folders, extract color and percentage from the names
    # For each folder add the image content, color and percentage to arrays
    # The training uses the index from the images array to find the right color and percentage
    X, y_color, y_perc = [], [], []
    for folder in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder)
        if not os.path.isdir(folder_path):
            continue
        try:
            color, perc = folder.split("_")
        except ValueError:
            continue
        for file in os.listdir(folder_path):
            if file.endswith(".png"):
                img = Image.open(os.path.join(folder_path, file)).resize((image_width, image_height))
                X.append(np.array(img) / 255.0)  # normalize the RGB values
                y_color.append(color)
                y_perc.append(perc)
    return np.array(X), y_color, y_perc

print("[INFO] Loading data...")
X, y_color, y_perc = load_images_and_labels(data_root)
# Now we have 3 arrays of the same length
# Note that the image content in X is the content of the files and this consumes memory.........

print("[INFO] Encoding labels to integers...")
le_color = LabelEncoder()
le_perc = LabelEncoder()

y_color_enc = le_color.fit_transform(y_color)
y_perc_enc = le_perc.fit_transform(y_perc)

n_color = len(le_color.classes_)
n_perc = len(le_perc.classes_)

X_train, X_test, y_color_train, y_color_test, y_perc_train, y_perc_test = train_test_split(
    X, y_color_enc, y_perc_enc, test_size=0.2, random_state=42
)

print("[INFO] Building model...")

# Below we create the variable input
#  from input we create the variables out_color and out_perc via variable x
#  and use input, out_color and out_perc to create the model.

# In the comments we calculate the total number of trainable parameters
# which are weights and biases ( biases are extra offsets in the model)
# I put this in because it may gives me better insight

# model.summary() at the end of this script reports these same numbers and more

# At the end of the summary is a list like :
#
# Total params: 3,145,322 (12.00 MB)
# Trainable params: 1,048,440 (4.00 MB)
# Non-trainable params: 0 (0.00 B)
# Optimizer params: 2,096,882 (8.00 MB)
#
# Only the trainable parameters are saved in the model which influences the file size of the model.
# The more parameters to save the bigger file

# ----------------------------------------------------------------------

# Input layer that defines the shape of the input images
# image_height and image_width are the height and width of the images, and 3 represents the RGB color channels
inputs = layers.Input(shape=(image_height, image_width, 3))

# ----------------------------------------------------------------------

# Convolutional layer with 32 filters, each of kernel size (3 x 3 pixels) and depth 3 (RGB) , followed by ReLU activation function
x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
# This helps extract features like edges or textures from the image
# This gives → 32 × ( 3 depth × ( 3 × 3 weights for pixels ) + 1 bias ) = 896 parameters
# (padding='same' keeps resizing in control)

# MaxPooling layer with pool size (2x2), downsampling the image and reducing its spatial dimensions

x = layers.MaxPooling2D((2,2))(x)

# This helps reduce computation and makes the model more invariant to small translations in the input
# image from 100 x 25 →  50 x 12 = 600

# output is 32 channels from the filters x 50 x 12 = 19200 values

# ----------------------------------------------------------------------

# Second Convolutional layer with 64 filters and (3x3) kernel size and depth 32 (feature maps previous layer) , followed by ReLU activation

x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
# This layer will extract more complex features (higher-level patterns)
# This gives → 64 × ( 32 dept × 3 × 3 weights + 1 bias ) = 18496 parameters

# MaxPooling layer again to downsample the feature maps after the second convolution

x = layers.MaxPooling2D((2,2))(x)
# This will again reduce the spatial size, making the model more computationally efficient
# image from 50 x 12 → 25 x 6 = 150

# output is 64 channels from the filters x 25 x 6 = 9600 values

# ----------------------------------------------------------------------

# Third Convolutional layer with 128 filters and (3x3) kernel size and depth 64 (feature maps previous layer) , followed by ReLU activation

#x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
# This layer will extract more complex features (higher-level patterns)
# This gives → 128 × ( 64 dept × 3 × 3 weights + 1 bias ) = 73856 parameters

# MaxPooling layer again to downsample the feature maps after the third convolution

#x = layers.MaxPooling2D((2,2))(x)
# This will again reduce the spatial size, making the model more computationally efficient
# image from 25 x 6 → 12 x 3 = 36

# output is 128 channels from the filters x 12 x 3 = 4608 values

# ----------------------------------------------------------------------

# Flatten layer to convert the 2D feature maps into a 1D vector for the fully connected layers
# This prepares the data to be passed into the fully connected Dense layers
x = layers.Flatten()(x)

# ----------------------------------------------------------------------

# Fully connected layer with 103 neurons ( when you have 100 for percentages and 3 colors)
# Fully connected layer with 106 neurons ( when you have 100 for percentages and 6 colors)

# The output will have 100 values representing percentages and 3 values for color probabilities (red, green, blue)
# This can be followed by a softmax activation to ensure that all outputs are probabilities
# in the range of 0 to 1 and that the sum of all outputs equals 1
x = layers.Dense(n_color + n_perc, activation='relu')(x)
# inputs to this layer is the output of the last MaxPooling2D
# neurons in last layer  = 103
# each neuron also has a bias so the number of biases = neurons = 103
# incoming connections from the previous layer = output values previous layer x neurons
# parameters for this layer = connections + biases
# with 1 layer  : 19200 x 103 + 103 = 1977703 or  19200 x 106 + 106 =  2035306
# with 2 layers :  9600 x 103 + 103 =  988903 or   9600 x 106 + 106 =  1017706
# with 3 layers :  4608 x 103 + 103 =  474727 or   4608 x 106 + 106 =   488554
# ----------------------------------------------------------------------

# Define the output layers of the network

out_color = layers.Dense(n_color, activation='softmax', name='color')(x)
# parameters = inputs * number of colors + number of colors(biases)
# 3 colors : 103 * 3 + 3 = 312
# 6 colors : 106 * 6 + 6 = 642

out_perc = layers.Dense(n_perc, activation='softmax', name='perc')(x)
# parameters = inputs * number of percentages + number of percentages(biases)
# 3 colors : 103 * 100 + 100 = 10400
# 6 colors : 106 * 100 + 100 = 10700

# ----------------------------------------------------------------------

# -------- total number of weights to be stored in the model -----------

# parameters layer 1 : 896
# parameters layer 2 : 18496
# parameters layer 3 : 73856

# parameters first dense layer : for 3 and 6 colors from above
# with 1 layer  : 19200 x 103 + 103 = 1977703 or  19200 x 106 + 106 =  2035306
# with 2 layers :  9600 x 103 + 103 =  988903 or   9600 x 106 + 106 =  1017706
# with 3 layers :  4608 x 103 + 103 =  474727 or   4608 x 106 + 106 =   488554

# parameters colors : 312 or 642 for 6 colors
# parameters percentages : 10400 or 10700 for 6 colors

# totals for 3 colors :
# total 1 layer  : 1977703 + 896                 + 312 + 10400 = 1989311
# total 2 layers :  988903 + 896 + 18496         + 312 + 10400 = 1019007
# total 3 layers :  474727 + 896 + 18496 + 73856 + 312 + 10400 =  578687

# for 6 colors :
# total 1 layer  : 2035306 + 896                 + 642 + 10700 = 2047544
# total 2 layers : 1017706 + 896 + 18496         + 642 + 10700 = 1048440
# total 3 layers :  488554 + 896 + 18496 + 73856 + 642 + 10700 =  593144

# the total number of weights to store is  : 1 for each parameter above (parameter is connection or bias)
# the result is that the number of weights to be stored in the model decreases dramatically when you add a layer

# when you add some colors the amount does not grow that much because you also have the 100 percentages.

# model.summary() at the end of this script reports these numbers and more

# ----------------------------------------------------------------------

# Create the model

model = Model(inputs=inputs, outputs=[out_color, out_perc])

# ----------------------------------------------------------------------

# Below
# Adam stands for Adaptive Moment Estimation — it’s a popular optimizer in deep learning.
# Think of loss as the “error score” during validation
#  we use sparse_categorical_crossentropy because we encoded the color and percentage labels as integers
# Metrics are values we want to see updating during training.

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


print("[INFO] Training model...")


early_stopping = EarlyStopping(monitor='val_loss',          # follow validation loss
                               patience=5,                  # Stop after 5 epochs without improvement
                               restore_best_weights=True)   # Put best weights back
 
history = model.fit(
    X_train,
    {'color': y_color_train, 'perc': y_perc_train},
    epochs=20,
    batch_size=64,
    validation_data=(X_test, {
        'color': y_color_test,
        'perc': y_perc_test
    }),
    callbacks=[early_stopping]
)

print(f"[INFO] Saving model and encoders... {model_path}  {encoders_path}")
save_model(model, model_path )

# We will use a script to analyse images using the model
# The output will be n_proc + n_color values between 0 and 1
# When you have 100 percentages and 3 colors you will end up with 103 values between 0 an 1
# We save labels for al these outputs to be able to translate them to red, green, blue and numbers 1..100
with open(encoders_path, "wb") as f:
    pickle.dump({
        'color': le_color,
        'perc': le_perc
    }, f)

# Show the summary including the numbers calculated in the comments above

model.summary()
