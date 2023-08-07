import tflite_runtime.interpreter as tflite
from PIL import Image
import numpy as np
import os
import sys


def accuracy_score(original_labels, predicted_labels):
    correct_predictions = 0
    total_samples = len(original_labels)

    for original_label, predicted_label in zip(original_labels, predicted_labels):
        if original_label == predicted_label:
            correct_predictions += 1

    accuracy = correct_predictions / total_samples
    return accuracy

script_directory = os.path.dirname(os.path.abspath(sys.argv[0]))
os.chdir(script_directory)

interpreter = tflite.Interpreter(model_path="mobilenet_v2_1.0_224_quantized_1_default_1.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def preprocess(image_path):
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    img = Image.open(image_path).resize((width, height))
    img = img.convert('RGB')
    input_data = np.array(img)
    input_data = np.expand_dims(img, axis=0)
    return input_data


def predict(folder_path):
    predicted_label = []
    true_labels = []
    for label_number, label_folder in enumerate(os.listdir(folder_path)):
        label_folder_path = os.path.join(folder_path, label_folder)
        for filename in os.listdir(label_folder_path):
                image_path = os.path.join(label_folder_path, filename)
                img_array = preprocess(image_path)  
                true_labels.append(label_number) 
                interpreter.set_tensor(input_details[0]['index'], img_array)

                # Make a prediction!
                interpreter.invoke()
                #  Get and print the result
                output_data = interpreter.get_tensor(output_details[0]['index'])
                prediction = np.argmax(output_data[0])
                
                predicted_label.append(prediction)
    accuracy = accuracy_score(true_labels, predicted_label)
    print(f"Model Accuracy tflite: {accuracy:.2f}")  


predict("test")
