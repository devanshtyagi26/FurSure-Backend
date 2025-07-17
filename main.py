from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import uvicorn
import os
from fastapi.middleware.cors import CORSMiddleware

# Load model at startup
MODEL_PATH = "data/cat_dog_classifier.keras"
model = load_model(MODEL_PATH)

app = FastAPI()


# Load allowed origins from env var or fallback to default list
origins = os.getenv("FRONTEND_ORIGINS").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET"],  # only GET requests allowed
    allow_headers=["Authorization", "Content-Type", "Accept", "Origin", "User-Agent", "X-Requested-With"],
)


IMG_SIZE = (256, 256)

@app.get("/")
def read_root():
    return {"message": "Welcome to FurSure"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Validate file is an image
        if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
            raise HTTPException(status_code=400, detail="Invalid file type. Only JPEG and PNG images are allowed.")

        # Save the uploaded image temporarily
        contents = await file.read()
        temp_file_path = "temp_img.jpg"
        with open(temp_file_path, "wb") as f:
            f.write(contents)

        # Load and preprocess image
        img = image.load_img(temp_file_path, target_size=IMG_SIZE)
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        pred = model.predict(img_array)[0][0]
        label = "Dog" if pred > 0.5 else "Cat"

        # Delete the temp file
        os.remove(temp_file_path)

        return JSONResponse({
            "prediction": label,
            "confidence": round(float(pred), 4)
        })

    except HTTPException as he:
        raise he
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})