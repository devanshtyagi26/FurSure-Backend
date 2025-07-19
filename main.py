from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.lite.python.interpreter import Interpreter
from tensorflow.keras.preprocessing import image
import numpy as np
from io import BytesIO
import os

# Load TFLite model at startup
MODEL_PATH = "./data/cat_dog_classifier.tflite"
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


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
        if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
            raise HTTPException(status_code=400, detail="Invalid file type. Only JPEG and PNG allowed.")

        contents = await file.read()
        img = image.load_img(BytesIO(contents), target_size=IMG_SIZE)
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array.astype(np.float32), axis=0)

        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]

        label = "Dog" if prediction > 0.5 else "Cat"

        return JSONResponse({
            "prediction": label,
            "confidence": round(float(prediction), 4)
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})