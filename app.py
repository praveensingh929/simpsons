from flask import Flask,request,render_template,request
from model import predict
import tensorflow as tf
import numpy as np
import jsonpickle
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
model = tf.keras.models.load_model('model/model_inception.h5')
# create the flask object
app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

ALLOWED_EXT = set(['jpg' , 'jpeg' , 'png'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT

def read_image(filename):
    img=image.load_img(filename,target_size = (421,418))
    x=image.img_to_array(img)
    x=np.expand_dims(x,axis=0)
    a=img_data=preprocess_input(x)
    return a

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename): #Checking file format
            filename = file.filename
            file_path = os.path.join('static/images', filename)
            file.save(file_path)
            img = read_image(file_path) #prepressing method
            x1=model.predict(img)
            x=np.argmax(model.predict(img), axis=1)
            if x == [2]:
                character = "Bart"
            elif x == [0]:
                character = "Grampa Simpson"
            elif x == [1]:
                character = "APU"
            return render_template('predict.html', character = character,prob=x1, user_image = file_path)
        else:
            return "Unable to read the file. Please check file extension"


if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True)