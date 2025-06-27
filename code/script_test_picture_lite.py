import sys
import numpy as np
from PIL import Image
import pickle
# We do not user tensorflow but import the LiteRTInterpreter for the tflite model
from ai_edge_litert import interpreter as interpreter_lib

model_path_2 = "model.tflite"
encoders_path = "label_encoders.pkl"
image_height = 100
image_width = 25

if len(sys.argv) != 2:
    print("Usage: python script_test_picture_lite.py <path-to-picture>")
    sys.exit(1)

image_path = sys.argv[1]

print("[INFO] Load model and encoders...")

interpreter = interpreter_lib.InterpreterWithCustomOps(
    custom_op_registerers=["pywrap_genai_ops.GenAIOpsRegisterer"],
    model_path=model_path_2,
    num_threads=2,
    experimental_default_delegate_latest_features=True)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Check if the model expects channel shape [1, image_height, image_width, 3]
if tuple(input_details[0]['shape'][1:4]) != (image_height, image_width, 3):
    raise ValueError(f"Unexpected shape: {input_details[0]['shape']}")

with open(encoders_path, "rb") as f:
    encoders = pickle.load(f)
    
print("[INFO] Prepare the image...")

img = Image.open(image_path).convert("RGB").resize((image_width, image_height))
# normalize to 0..1 inputs
img_array = np.array(img, dtype=np.float32) / 255.0
# add a dimension because the model needs this (1, image_width, image_height, 3)
# the 1 means that the model can process this as a batch
img_array = np.expand_dims(img_array, axis=0)

print("[INFO] Predict with LiteRT...")

interpreter.set_tensor(input_details[0]['index'], img_array)
interpreter.invoke() # this fills output_details

# Process outputs, note that perc is first and color is second in output_details

color_output = interpreter.get_tensor(output_details[1]['index'])
perc_output  = interpreter.get_tensor(output_details[0]['index'])

# First some info on the color output

print("------------------ Some info for the color ------------------")

# This prints the labels which we read from the pickle file

print(f"labels        : {encoders['color'].classes_}")

# preds[0] from the model output contains the probability output for the colors

print(f"probabilities : {interpreter.get_tensor(output_details[1]['index'])}")

# This prints the maximum probability

print(f"maximum       : {max(interpreter.get_tensor(output_details[1]['index'])[0]):.6e}")

# This prints the index of the maximum probability

print(f"index of max  : {np.argmax(interpreter.get_tensor(output_details[1]['index']))} (index starts at 0)")

# and with that we can print the label itself

print(f"color label   : {encoders['color'].classes_[np.argmax(interpreter.get_tensor(output_details[1]['index']))]}")

print("-------------------------------------------------------------")

# Decode to labels

# Bassed on the above we could use:

#color_pred = encoders['color'].classes_[np.argmax(interpreter.get_tensor(output_details[1]['index']))]
#perc_pred  =  encoders['perc'].classes_[np.argmax(interpreter.get_tensor(output_details[0]['index']))]

# but we use the model output direct:

color_pred = encoders['color'].inverse_transform([np.argmax(color_output[0])])[0]
perc_pred = encoders['perc'].inverse_transform([np.argmax(perc_output[0])])[0]

print(f"{'Color'     :<12}: {color_pred:<10}  Certainty: {100 * max(color_output[0]):6.2f}%")
print(f"{'Percentage':<12}: {perc_pred:<3}%        Certainty: {100 * max(perc_output[0]):6.2f}%")

