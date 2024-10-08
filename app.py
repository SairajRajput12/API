from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import requests
import io
from io import BytesIO
import dropbox

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Read API Key from environment variables
DROPBOX_ACCESS_TOKEN = os.getenv('DROPBOX_ACCESS_TOKEN')



class Plant(BaseModel):
    img_url: str
    name: str


app = FastAPI()

origins = {"*"} 

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
    2: 'Apple___Cedar_apple_rust',
    3: 'Apple___healthy',
    4: 'Blueberry___healthy',
    5: 'Cherry_(including_sour)___Powdery_mildew',
    6: 'Cherry_(including_sour)___healthy',
    7: 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    8: 'Corn_(maize)___Common_rust_',
    9: 'Corn_(maize)___Northern_Leaf_Blight',
    10: 'Corn_(maize)___healthy',
    11: 'Grape___Black_rot',
    12: 'Grape___Esca_(Black_Measles)',
    13: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    14: 'Grape___healthy',
    15: 'Orange___Haunglongbing_(Citrus_greening)',
    16: 'Peach___Bacterial_spot',
    17: 'Peach___healthy',
    18: 'Pepper,_bell___Bacterial_spot',
    19: 'Pepper,_bell___healthy',
    20: 'Potato___Early_blight',
    21: 'Potato___Late_blight',
    22: 'Potato___healthy',
    23: 'Raspberry___healthy',
    24: 'Soybean___healthy',
    25: 'Squash___Powdery_mildew',
    26: 'Strawberry___Leaf_scorch',
    27: 'Strawberry___healthy',
    28: 'Tomato___Bacterial_spot',
    29: 'Tomato___Early_blight',
    30: 'Tomato___Late_blight',
    31: 'Tomato___Leaf_Mold',
    32: 'Tomato___Septoria_leaf_spot',
    33: 'Tomato___Spider_mites Two-spotted_spider_mite',
    34: 'Tomato___Target_Spot',
    35: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    36: 'Tomato___Tomato_mosaic_virus',
    37: 'Tomato___healthy'
}

peas = {
    0: 'Hispa',
    1: 'LeafBlast',
    2: 'BrownSpot',
    3: 'Healthy'
}


def load_and_preprocess_image(image_path_or_url, target_size=(256, 256)):
    # Determine if it's a URL
    if image_path_or_url.startswith("hsttp"):
        # Download the image from the URL
        response = requests.get(image_path_or_url)
        img = Image.open(BytesIO(response.content))
    else:
        # Load image from local path
        img = Image.open(image_path_or_url)

    # Resize the imagesz
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
    ans = ""
    if name_of_crop == 'pea':  
        preprocessed_img = load_and_preprocess_image(image_url) 
        predictions = model.predict(preprocessed_img) 
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        ans = class_indices[predicted_class_index]
    else: 
        preprocessed_img = load_and_preprocess_image(image_url, (224, 224)) 
        # Get model predictions
        predictions = model.predict(preprocessed_img)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        ans = class_indices[predicted_class_index]

    return ans



def load_model_from_dropbox(dropbox_path, access_token):
    # Initialize Dropbox client
    dbx = dropbox.Dropbox(access_token)
    
    # Ensure the Dropbox path starts with "/"
    if not dropbox_path.startswith('/'):
        dropbox_path = '/' + dropbox_path
    
    # Download the file content from Dropbox
    metadata, res = dbx.files_download(path=dropbox_path)
    
    # Use a BytesIO stream to simulate a file for TensorFlow's load_model
    model_stream = io.BytesIO(res.content)
    
    # Load the model directly from the BytesIO stream
    try:
        # Save the stream content temporarily to a file
        with open("temp_model.h5", "wb") as f:
            f.write(model_stream.getbuffer())
        
        # Load the model from the temporary file
        model = load_model("temp_model.h5")
        print('Model loaded successfully!')
        return model
    except OSError as e:
        print(f"Error loading model: {e}")



@app.get("/")
def hello_world():
    return {"message": "This is the Plant Disease Detection System!"}


@app.post("/predict")
def predict(data: Plant):
    name_of_crop = data.name
    img_url = data.img_url

    if not name_of_crop:
        raise HTTPException(status_code=400, detail="Invalid crop name")

    

    # Load model from Dropbox
    if name_of_crop == 'pea':
        model = load_model_from_dropbox('model.h5', DROPBOX_ACCESS_TOKEN)
    else:
        model = load_model_from_dropbox('plant_disease_prediction_model.h5',DROPBOX_ACCESS_TOKEN)


    # Proceed with prediction if model is valid
    prediction = predict_image_class(model, img_url, peas if name_of_crop == 'pea' else others, name_of_crop)
    
    return {"prediction": prediction}
