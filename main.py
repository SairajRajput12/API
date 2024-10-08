from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import requests
from io import BytesIO


class Plant(BaseModel):
    img_url: str
    name: str


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Class indices for crops
others = {
    0: 'Apple___Apple_scab',
    1: 'Apple___Black_rot',
    # (rest of your class mappings...)
}

peas = {
    0: 'Hispa',
    1: 'LeafBlast',
    2: 'BrownSpot',
    3: 'Healthy'
}


def load_and_preprocess_image(image_path_or_url, target_size=(256, 256)):
    # Determine if it's a URL
    if image_path_or_url.startswith("http"):
        # Download the image from the URL
        response = requests.get(image_path_or_url)
        img = Image.open(BytesIO(response.content))
    else:
        # Load image from local path
        img = Image.open(image_path_or_url)

    # Resize the image
    img = img.resize(target_size)
    # Convert to numpy array
    img_array = np.array(img)
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Normalize values to [0, 1]
    img_array = img_array.astype('float32') / 255.
    return img_array


def predict_image_class(model, image_url, class_indices, name_of_crop):
    print('Start prediction...')

    # Load and preprocess the image
    preprocessed_img = load_and_preprocess_image(image_url, (224, 224))

    # Get model predictions
    predictions = model.predict(preprocessed_img)

    # Find the predicted class
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[predicted_class_index]

    return predicted_class_name


@app.get("/")
def hello_world():
    return {"message": "This is the Plant Disease Detection System!"}


@app.post("/predict")
def predict(data: Plant):
    name_of_crop = data.name
    img_url = data.img_url

    if not name_of_crop:
        raise HTTPException(status_code=400, detail="Invalid crop name")

    # Load appropriate model
    if name_of_crop == 'pea':
        model = load_model('peas.h5')
        prediction = predict_image_class(model, img_url, peas, name_of_crop)
    else:
        model = load_model('others.h5')
        prediction = predict_image_class(model, img_url, others, name_of_crop)

    return {"prediction": prediction}


if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=8000)
