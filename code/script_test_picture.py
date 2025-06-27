import sys
import numpy as np
from PIL import Image
import tensorflow as tf
import pickle

model_path = "model.keras"
encoders_path = "label_encoders.pkl"
image_height = 100
image_width = 25

if len(sys.argv) != 2:
    print("Usage: python script_test_picture.py <path-to-picture>")
    sys.exit(1)

image_path = sys.argv[1]

print("[INFO] Load model and encoders...")

model = tf.keras.models.load_model(model_path)

with open(encoders_path, "rb") as f:
    encoders = pickle.load(f)

print("[INFO] Prepare the image...")

img = Image.open(image_path).convert("RGB").resize((image_width, image_height))
# normalize to 0..1 inputs
img_array = np.array(img, dtype=np.float32) / 255.0
# add a dimension because the model needs this (1, image_width, image_height, 3)
# the 1 means that the model can process this as a batch
img_array = np.expand_dims(img_array, axis=0)  

print("[INFO] Predict...")

preds = model.predict(img_array, verbose=0)

# First some info on the color output
print("------------------ Some info for the color ------------------")

# This prints the labels which we read from the pickle file

print(f"labels        : {encoders['color'].classes_}")

# preds[0] from the model output contains the probability output for the colors

print(f"probabilities : {preds[0]}")

# This prints the maximum probability

print(f"maximum       : {max(preds[0][0]):.6e}")

# This prints the index of the maximum probability

print(f"index of max  : {np.argmax(preds[0])} (index starts at 0)")

# and with that we can print the label itself

print(f"color label   : {encoders['color'].classes_[np.argmax(preds[0])]}")

print("-------------------------------------------------------------")

# Decode to labels

# Bassed on the above we could use:

#color_pred = encoders['color'].classes_[np.argmax(preds[0])]
#perc_pred = encoders['perc'].classes_[np.argmax(preds[1])]

# but we use the model output direct:

color_pred = encoders['color'].inverse_transform([np.argmax(preds[0])])[0]
perc_pred = encoders['perc'].inverse_transform([np.argmax(preds[1])])[0]

print(f"{'Color'     :<12}: {color_pred:<10}  Certainty: {100 * max(preds[0][0]):6.2f}%")
print(f"{'Percentage':<12}: {perc_pred:<3}%        Certainty: {100 * max(preds[1][0]):6.2f}%")


