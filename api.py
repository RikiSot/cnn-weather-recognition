# 1. Library Imports
from http.client import HTTPException
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_file
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from tensorflow import expand_dims
import numpy as np
import json

import uvicorn
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi import HTTPException
import os
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '.'))


# Load labels from json file
labels_path = os.path.join(ROOT_DIR, 'data', 'labels.json')
labels = json.load(open(labels_path))
IMG_SIZE = (120, 120)  # Same size as the model's input during training

# 2. Create the app object
app = FastAPI()

# 3. Load models
model_path = os.path.join(ROOT_DIR, 'models', 'ResNet50.h5')
model = tf.keras.models.load_model(model_path)


# 4. API Endpoints and methods
@app.get("/")
async def home():
    return {"message": "Technical test"}


@app.post("/predict")
async def predict_image(image_link: str = ''):
    """
    Predict the label from a given image (url)
    """
    if image_link == '':
        raise HTTPException(status_code=400, detail='No image link provided')
    try:
        img_path = get_file(origin=image_link)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    image = load_img(img_path, target_size=IMG_SIZE)
    image = img_to_array(image)
    image = expand_dims(image, axis=0)

    score = model.predict(image)
    model_score = round(score.max() * 100, 2)

    prediction = np.argmax(score, axis=1)
    label = labels[str(prediction[0])]
    response = {
        'label': label,
        'confidence': model_score,
    }
    return JSONResponse(content=response)


def start_server():
    "Launch the server with poetry run start at root level"
    uvicorn.run(app, host="127.0.0.1", port=8000, debug=True)


if __name__ == "__main__":
    start_server()
