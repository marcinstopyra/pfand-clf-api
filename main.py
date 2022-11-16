#  To start app: uvicorn main:app --reload
#  Then open docs path for tests /docs

from PIL import Image
from io import BytesIO
import numpy as np
from utils import preprocess_image
import os

import tflite_runtime.interpreter as tflite


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse


MODEL = tflite.Interpreter("./model/model.tflite")
MODEL.allocate_tensors()
input_details = MODEL.get_input_details()
output_details = MODEL.get_output_details()



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

def predictLite(image, model):
    """Makes a prediction on a sample with a tensorflow Lite converted model
    ---
    args:
        img:        input sample
    returns:
        pred:       prediction class probabilities
    """
    model.set_tensor(input_details[0]['index'], image.reshape((1, 200, 200, 3)))
    model.invoke()
    pred = model.get_tensor(output_details[0]['index']).reshape((4,))

    return pred

@app.get('/', response_class=HTMLResponse)
async def index():
    # return {"Message": "Welcome to PFAND CLASSIFIER 1.0"}
    content = """
        <h1 style="text-align:center">Welcome to PFAND Classifier!</h1>
    """
    return content



@app.post("/predict_lite/")
async def make_prediction(file: UploadFile = File(...)):
    """ Takes a raw image, preprocess it and then makes prediction
    """
    img_arr = load_image_into_numpy_array(await file.read())
    img = Image.fromarray(img_arr)
    

    img = preprocess_image(img, resize_factor=0.1, cropped_size=200)

    # to numpy array
    img = np.array(img)

    # convert img to FLOAT32 - required by TF Lite model
    img = img.astype(np.float32)

    # make a prediction
    prediction_oh = predictLite(img, MODEL)

    print(prediction_oh)
    # One_hot decoding
    prediction = one_hot_decode(prediction_oh)


    # get confidence
    confidence = np.round(float(np.max(prediction_oh)), decimals=2)

    

    return {"prediction": prediction, "confidence": confidence}