import sys
import numpy as np
from PIL import Image
import pickle
# We do not user tensorflow but import the LiteRTInterpreter for the tflite model
from ai_edge_litert import interpreter as interpreter_lib

model_path = "model_8_labels.tflite"
encoders_path = "label_encoders_8_labels.pkl"

if len(sys.argv) != 2:
    print("Usage: python script_test_picture_lite.py <path-to-picture>")
    sys.exit(1)

image_path = sys.argv[1]

print("[INFO] Load model and encoders...")

interpreter = interpreter_lib.InterpreterWithCustomOps(
    custom_op_registerers=["pywrap_genai_ops.GenAIOpsRegisterer"],
    model_path=model_path,
    num_threads=2,
    experimental_default_delegate_latest_features=True)

# Get the input tensor of the model
input_details = interpreter.get_input_details()

# The first is the image
input_shape = input_details[0]['shape']

# expected shape is [batch_size, height, width, channels] most of the time
batch_size, image_height, image_width, channels = input_shape

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

label_1_output = interpreter.get_tensor(output_details[5]['index'])
label_2_output = interpreter.get_tensor(output_details[1]['index'])
label_3_output = interpreter.get_tensor(output_details[6]['index'])
label_4_output = interpreter.get_tensor(output_details[4]['index'])
label_5_output = interpreter.get_tensor(output_details[0]['index'])
label_6_output = interpreter.get_tensor(output_details[3]['index'])
label_7_output = interpreter.get_tensor(output_details[7]['index'])
label_8_output = interpreter.get_tensor(output_details[2]['index'])

print(label_1_output)
print(label_2_output)
print(label_3_output)
print(label_4_output)
print(label_5_output)
print(label_6_output)
print(label_7_output)
print(label_8_output)
# First some info on the label_1 output

print("----------------- Some info for the labels ------------------")

# This prints the labels which we read from the pickle file

print(f"labels  1      : {encoders['label_1'].classes_}")
print(f"labels  2      : {encoders['label_2'].classes_}")
print(f"labels  3      : {encoders['label_3'].classes_}")
print(f"labels  4      : {encoders['label_4'].classes_}")
print(f"labels  5      : {encoders['label_5'].classes_}")
print(f"labels  6      : {encoders['label_6'].classes_}")
print(f"labels  7      : {encoders['label_7'].classes_}")
print(f"labels  8      : {encoders['label_8'].classes_}")
print("-------------------------------------------------------------")

# Decode to labels

# Bassed on the above we could use the next to predict the label_1 and label_2:

#label_1_pred = encoders['label_1'].classes_[np.argmax(interpreter.get_tensor(label_1_details[1]['index']))]
#label_2_pred = encoders['label_2'].classes_[np.argmax(interpreter.get_tensor(label_2_details[0]['index']))]

# but we use the model output direct:

label_1_pred = encoders['label_1'].inverse_transform([np.argmax(label_1_output[0])])[0]
label_2_pred = encoders['label_2'].inverse_transform([np.argmax(label_2_output[0])])[0]
label_3_pred = encoders['label_3'].inverse_transform([np.argmax(label_3_output[0])])[0]
label_4_pred = encoders['label_4'].inverse_transform([np.argmax(label_4_output[0])])[0]
label_5_pred = encoders['label_5'].inverse_transform([np.argmax(label_5_output[0])])[0]
label_6_pred = encoders['label_6'].inverse_transform([np.argmax(label_6_output[0])])[0]
label_7_pred = encoders['label_7'].inverse_transform([np.argmax(label_7_output[0])])[0]
label_8_pred = encoders['label_8'].inverse_transform([np.argmax(label_8_output[0])])[0]

print(f"{'label_1':<16}: {label_1_pred:>12} ; Certainty: {100 * max(label_1_output[0]):6.2f}%")
print(f"{'label_2':<16}: {label_2_pred:>12} ; Certainty: {100 * max(label_2_output[0]):6.2f}%")
print(f"{'label_3':<16}: {label_3_pred:>12} ; Certainty: {100 * max(label_3_output[0]):6.2f}%")
print(f"{'label_4':<16}: {label_4_pred:>12} ; Certainty: {100 * max(label_4_output[0]):6.2f}%")
print(f"{'label_5':<16}: {label_5_pred:>12} ; Certainty: {100 * max(label_5_output[0]):6.2f}%")
print(f"{'label_6':<16}: {label_6_pred:>12} ; Certainty: {100 * max(label_6_output[0]):6.2f}%")
print(f"{'label_7':<16}: {label_7_pred:>12} ; Certainty: {100 * max(label_7_output[0]):6.2f}%")
print(f"{'label_8':<16}: {label_8_pred:>12} ; Certainty: {100 * max(label_8_output[0]):6.2f}%")


'''
# 2nd choice label x
# Zorg ervoor dat de label_4_output[0] een numpy array is
#for x in (2,0,3,1):
for l in range(1,8):
    if l == 1:
        y=2
    elif l == 2:
        y=0
    elif l == 3:
        y=3
    elif l == 4:
        y=1
    elif l == 5:
        y=1
    elif l == 6:
        y=1
    elif l == 7:
        y=1
            
    label_output = interpreter.get_tensor(output_details[y]['index'])
    values = label_output[0]

    # Vind de indices van de twee grootste waarden
    sorted_indices = np.argsort(values)[::-1]  # Sorteer in aflopende volgorde
    max_index = sorted_indices[0]
    second_max_index = sorted_indices[1]

    # Haal de twee grootste waarden
    max_value = encoders['label_'+str(l)].classes_[max_index]
    max_perc = values[max_index]
    second_max_value = encoders['label_'+str(l)].classes_[second_max_index]
    second_max_perc = values[second_max_index]

    # Afdrukken van de waarden en de zekerheid
    print(f"label_{str(l)} 2nd     : {second_max_value:>12} ; Certainty: {100 * second_max_perc:6.2f}%")
'''
