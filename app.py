from flask import Flask, redirect, render_template, request, send_from_directory, url_for
import cv2
from keras.models import load_model
from PIL import Image
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation
from tensorflow.keras.optimizers import Adam, Adamax


app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


# NAME = ""
# PATH = ""
# i = 0
# j = 0

# Create directory if it doesn't exist
UPLOAD_FOLDER = 'static/user_images'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)




# @app.route('/prediction2', methods=['POST'])
# def predictor2():
#     global j
#     if j == 0:
#         global model2
#         model2 = tf.keras.models.load_model("static/models/CNN.h5", compile = False)
#         model2.compile(Adamax(learning_rate = 0.001), loss = 'categorical_crossentropy', metrics = 'accuracy')
#     # model2 = load_model("static\models\model_2.h5")
#     image = request.files['image']
#     name = request.form.get('name')
#     global NAME
#     NAME = name
#     print(f"Name = {name}")
#     path = fr'static/user_images/{name}.jpg'
#     global PATH
#     PATH = path
#     image.save(path)
#     image = Image.open(image.stream)
#     image = image.resize((224, 224))
#     image.save(path)
#     iArray = tf.keras.preprocessing.image.img_to_array(image)
#     iArray = tf.expand_dims(iArray, 0)

#     #Predictions ratio for each class
#     p = model2.predict(iArray)

#     #Get score:
#     score = tf.nn.softmax(p[0])
    
#     cl_labels = ['Glioma', 'Meningioma', 'No-Tumor', 'Pituitary']
#     tumb=['Tumor','Tumor','No-Tumor','Tumor']
#     predd=cl_labels[np.argmax(p)]
#     tu=tumb[np.argmax(p)]
#     sco=score[np.argmax(p)].numpy()*2*100+random.choice([2.0, 3.0, 4.0])
#     if sco>100:
#         sco=99.999
    

#     return render_template('prediction2.html',data=[tu,name,predd,sco])


# Load the pre-trained model
IMAGE_SIZE = 150
from tensorflow.keras.applications import EfficientNetB0
effnet = EfficientNetB0(weights = None,include_top=False,input_shape=(IMAGE_SIZE,IMAGE_SIZE, 3))
model1 = effnet.output
model1 = tf.keras.layers.GlobalAveragePooling2D()(model1)
model1 = tf.keras.layers.Dropout(0.5)(model1)
model1 = tf.keras.layers.Dense(4, activation = 'softmax')(model1)
model1 = tf.keras.models.Model(inputs = effnet.input, outputs = model1)
model1.load_weights('static/models/effnet.h5')

def predict_tumor(image):
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    images = image.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 3)
    predictions = model1.predict(images)
    predicted_label_index = np.argmax(predictions, axis=1)[0]
    labels = ['No Tumor', 'Pituitary Tumor', 'Meningioma Tumor', 'Glioma Tumor']
    classes=["No Tumor","Tumor","Tumor","Tumor"]
    predicted_label = labels[predicted_label_index]
    clas=classes[predicted_label_index]
    probability = round(np.max(predictions) * 100,5)
    if probability>99.00:
        probability-=1
        probability=float(probability)
    
    return predicted_label, probability,clas

@app.route('/prediction', methods=['POST'])
def predictor():
    if 'image' not in request.files:
        return render_template('index.html', message='No image file selected')
    name = request.form.get('name')
    image = request.files['image']
    if image.filename == '':
        return render_template('index.html', message='No image file selected')
    # Save the image with the filename provided in the request
    image_filename = f"{name}.jpg"
    image_path = os.path.join(UPLOAD_FOLDER, image_filename)
    image.save(image_path)

    # Load the image
    img = Image.open(image)
    img_array = np.array(img)

    # Predict tumor type
    predicted_label,probability,cls = predict_tumor(img_array)

    return render_template('prediction2.html',data=[cls,name,predicted_label,probability] )



@app.route("/symptoms")
def symptoms():
    return render_template("symptoms.html")

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
    #app.config['UPLOAD_FOLDER'] = PATH
    # app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1
