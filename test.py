import tensorflow as tf
from PIL import Image
import numpy as np
import os
import sys
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

script_directory = os.path.dirname(os.path.abspath(sys.argv[0]))
os.chdir(script_directory)

INPUT_IMG_SIZE = 224
INPUT_IMG_SHAPE = (224, 224, 3)
classes = os.listdir("test")

def plot_confusion_matrix(true_labels, predicted_labels, classes):
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix")
    plt.show()

def preprocess(image):
    image_path = image
    img = Image.open(image_path).convert('RGB')
    img = img.resize((INPUT_IMG_SIZE, INPUT_IMG_SIZE), Image.BICUBIC)
    input_data = np.array(img, dtype=np.float32)/255.0
    input_data = input_data.reshape(1, INPUT_IMG_SIZE, INPUT_IMG_SIZE, 3)
    image_tensor = tf.convert_to_tensor(input_data)
    return image_tensor


def predict(model_path, folder_path):
    OUTPUT_SAVED_MODEL_DIR = f'./optimized_models_{model_path[:-3]}'
    loaded_model = tf.saved_model.load(OUTPUT_SAVED_MODEL_DIR)
    infer = loaded_model.signatures['serving_default']
    predicted_label = []
    true_labels = []
    for label_number, label_folder in enumerate(os.listdir(folder_path)):
        label_folder_path = os.path.join(folder_path, label_folder)
        for filename in os.listdir(label_folder_path):
                image_path = os.path.join(label_folder_path, filename)
                img_array = preprocess(image_path)  # Replace target_size with your model's input size
                true_labels.append(label_number)  # Use the current folder name as the true label
                output = infer(img_array)
                predictions = next(iter(output))
                prediction = np.argmax(output[predictions].numpy().squeeze())
                
                predicted_label.append(prediction)
    accuracy = accuracy_score(true_labels, predicted_label)
    print(f"Model Accuracy: {accuracy:.2f}")   
    plot_confusion_matrix(true_labels, predicted_label, classes=classes) 
    del infer

predict("keras_model.h5", "test")
   
