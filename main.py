#  To start app: uvicorn main:app --reload
#  Then open docs path for tests /docs

from PIL import Image
from io import BytesIO
import numpy as np
from utils import preprocess_image
import os

import tflite_runtime.interpreter as tflite


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse




MODEL = tflite.Interpreter("./model/model.tflite")
MODEL.allocate_tensors()
input_details = MODEL.get_input_details()
output_details = MODEL.get_output_details()

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

def predictFullProcess(img_arr, model):
    """ Takes an image in binary form
    TODO
    TODO
    TODO
    """
    img = Image.fromarray(img_arr)

    img = preprocess_image(img, resize_factor=0.1, cropped_size=200)

    # to numpy array
    img = np.array(img)

    # convert img to FLOAT32 - required by TF Lite model
    img = img.astype(np.float32)

    # make a prediction
    prediction_oh = predictLite(img, model)

    print(prediction_oh)
    # One_hot decoding
    prediction = one_hot_decode(prediction_oh)


    # get confidence
    confidence = np.round(float(np.max(prediction_oh)), decimals=2)

    

    return {"prediction": prediction, "confidence": confidence}

## ------------------------------------------------------------------------------------
## Loading model and bulding API
api_app = FastAPI(title="api_app")

@api_app.post("/predict_lite/")
async def make_prediction(file: UploadFile = File(...)):
    """ Takes an image raw of a bottle/can, preprocesses it so that it matches the requirenments of the model input
    and makes a prediction using a TF Lite model
    """
    img_arr = load_image_into_numpy_array(await file.read())
    result = predictFullProcess(img_arr, MODEL)

    return result

## ------------------------------------------------------------------------------------
## webpage

templates = Jinja2Templates(directory="ui/templates")

app = FastAPI(title="main app")

app.mount("/api", api_app)
app.mount("/static", StaticFiles(directory="ui/static"), name="static")
# app.mount("/", StaticFiles(directory="ui", html=True), name="ui")

# # Render index
@app.get('/', response_class=HTMLResponse)
def index(request: Request):
    # return {"Message": "Welcome to PFAND CLASSIFIER 1.0"}
    return templates.TemplateResponse("index.html", {"request": request})

@app.get('/about', response_class=HTMLResponse)
def about(request: Request):
    # return {"Message": "Welcome to PFAND CLASSIFIER 1.0"}
    return templates.TemplateResponse("about.html", {"request": request})

@app.get('/howtouse', response_class=HTMLResponse)
def howtouse(request: Request):
    # return {"Message": "Welcome to PFAND CLASSIFIER 1.0"}
    return templates.TemplateResponse("howtouse.html", {"request": request})




@app.post("/result") #, response_class=HTMLResponse)
async def make_prediction(request: Request, file1: UploadFile = File(default=None), file2: UploadFile = File(default=None)):
    """ TODO
    """
    file1 = await file1.read()
    file2 = await file2.read()

    # img_arr = load_image_into_numpy_array(file1)
    if len(file1) != 0:
        file = file1
    elif len(file2) != 0:
        file = file2
    else:
        return {'msg': "error"}

    img_arr = load_image_into_numpy_array(file)
    result = predictFullProcess(img_arr, MODEL)

    context = {"request": request, 
               "result": result}

    return templates.TemplateResponse("result.html", context)

@app.get("/result")
async def redirect_to_index():
    return RedirectResponse("/")
    