from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load the pre-trained CNN model
model_path = 'pretrained_model.h5'
class_labels = ['CARIES', 'GINGIVITIES', 'NORMAL']
model = load_model(model_path)


def predict_class(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]

    return predicted_class


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('index.html', error='No image provided.')

        image_file = request.files['image']

        if image_file.filename == '':
            return render_template('index.html', error='No selected image.')

        if image_file:
            # Save the uploaded image
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
            image_file.save(image_path)

            # Get the predicted class
            prediction = predict_class(image_path)

            # Pass the image path and prediction to the result template
            return render_template('result.html', prediction=prediction, image_filename=image_file.filename)

    return render_template('index.html')

    # Process any data if needed




if __name__ == '__main__':
    app.run(debug=True)
