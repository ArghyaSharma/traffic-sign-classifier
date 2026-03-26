import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow requests from anywhere (useful for frontend later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model
model = None

def get_model():
    global model
    if model is None:
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')
        model = tf.keras.models.load_model("model.h5", compile=False)
    return model

IMG_WIDTH = 30
IMG_HEIGHT = 30

# Class labels (important for real output)
SIGN_NAMES = {
    0: "Speed limit 20", 1: "Speed limit 30", 2: "Speed limit 50",
    3: "Speed limit 60", 4: "Speed limit 70", 5: "Speed limit 80",
    6: "End of 80 limit", 7: "Speed limit 100", 8: "Speed limit 120",
    9: "No passing", 10: "No passing (trucks)", 11: "Right of way",
    12: "Priority road", 13: "Yield", 14: "Stop", 15: "No vehicles",
    16: "No trucks", 17: "No entry", 18: "General caution",
    19: "Left curve", 20: "Right curve", 21: "Double curve",
    22: "Bumpy road", 23: "Slippery road", 24: "Road narrows",
    25: "Road work", 26: "Traffic signals", 27: "Pedestrians",
    28: "Children crossing", 29: "Bicycles crossing", 30: "Ice/snow",
    31: "Wild animals", 32: "End of limits", 33: "Turn right",
    34: "Turn left", 35: "Go straight", 36: "Go straight or right",
    37: "Go straight or left", 38: "Keep right", 39: "Keep left",
    40: "Roundabout", 41: "End no passing", 42: "End no passing (trucks)"
}

@app.get("/")
def root():
    return {"message": "Traffic Sign Classifier is running 🚀"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read uploaded file
        contents = await file.read()

        # Convert to numpy array
        nparr = np.frombuffer(contents, np.uint8)

        # Decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Resize + normalize
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        # Predict
        model = get_model()
        predictions = model.predict(img, verbose=0)
        category = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))

        return {
            "category": category,
            "sign_name": SIGN_NAMES.get(category, "Unknown"),
            "confidence": round(confidence * 100, 2)
        }

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
