from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)

# Load your custom model
model_path = r"C:\Users\DELL\eye-classifier-main\eye-classifier-main\models\my_model.h5"
model = load_model(model_path)

# Class labels
class_labels = ["Normal (N)", "Diabetes (D)", "Glaucoma (G)", "Cataract (C)",
                "Age-related Macular Degeneration (A)", "Hypertension (H)",
                "Pathological Myopia (M)", "Other diseases/abnormalities (O)"]

# Path to save uploaded images
image_save_folder = r"C:\Users\DELL\eye-classifier-main\eye-classifier-main\preprocessed_images"

# Ensure the Training_Images directory exists
if not os.path.exists(image_save_folder):
    os.makedirs(image_save_folder)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = os.path.join(image_save_folder, imagefile.filename)
    imagefile.save(image_path)

    # Preprocess the image
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image /= 255.0  # Normalize the image if necessary

    # Make prediction
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction, axis=1)[0]
    classification = class_labels[predicted_class]

    return render_template('index.html', prediction=classification)

if __name__ == '__main__':
    app.run(port=3000, debug=True)
