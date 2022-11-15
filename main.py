#  To start app: uvicorn main:app --reload
#  Then open docs path for tests /docs

import tensorflow as tf
from PIL import Image
from io import BytesIO
import numpy as np
from utils import preprocess_image
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from fastapi import FastAPI, File, UploadFile


MODEL = tf.keras.models.load_model('./model/model.h5')


app = FastAPI()

def load_image_into_numpy_array(data):
    return np.array(Image.open(BytesIO(data)))

def one_hot_decode(pred_oh):
    # One_hot decoding
    oh_to_true_label = {0: 0,
                        1: 8,
                        2: 15,
                        3: 25}
    pred = oh_to_true_label[np.argmax(pred_oh)]
    return pred

@app.get('/')
async def index():
    return {"Message": "Welcome to PFAND CLASSIFIER 1.0"}




@app.post("/predict_ready/")
async def make_prediction(file: UploadFile = File(...)):
    """ Takes a ready preprocessed image and makes prediction
    """
    # print(type(file))
    image = load_image_into_numpy_array(await file.read())
    # print(type(image))
    # print(image.shape)

    prediction_oh = MODEL.predict(np.array([image]))[0]
    # print(prediction)

    # One_hot decoding
    prediction = one_hot_decode(prediction_oh)

    return {"prediction": prediction}


@app.post("/predict_raw/")
async def make_prediction(file: UploadFile = File(...)):
    """ Takes a raw image, preprocess it and then makes prediction
    """
    img_arr = load_image_into_numpy_array(await file.read())
    img = Image.fromarray(img_arr)
    
    # free memory
    del img_arr

    img = preprocess_image(img, resize_factor=0.1, cropped_size=200)

    # to numpy array
    img = np.array(img)

    # make a prediction
    prediction_oh = MODEL.predict(np.array([img]), verbose=0)[0]

    # free memory
    del img

    print(prediction_oh)
    # One_hot decoding
    prediction = one_hot_decode(prediction_oh)


    # get confidence
    confidence = np.round(float(np.max(prediction_oh)), decimals=2)

    

    return {"prediction": prediction, "confidence": confidence}