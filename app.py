from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from model.model import EMNISTClassifier
import torch
import numpy as np
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Loading the trained weights
model = EMNISTClassifier()
model.load_state_dict(torch.load("model/emnist_model.pth", map_location=torch.device('cpu')))
model.eval()

class PixelData(BaseModel):
    pixels: list  
        
mapping=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't' ]

@app.get("/")
async def read_index():
    return FileResponse(os.path.join("templates", "index.html"))

@app.post("/predict")
async def predict_digit(data: PixelData):
    try:
        if len(data.pixels) != 784:
            raise HTTPException(status_code=400, detail="Invalid image dimensions. Must be 784 pixels.")
        
        # preparing data for model
        img_array = np.array(data.pixels, dtype=np.float32).reshape(1, 1, 28, 28)
        img_array = img_array / 255.0
        img_array = (img_array - 0.1307) / 0.3081
        tensor_input = torch.tensor(img_array)
        
        with torch.no_grad():
            outputs = model(tensor_input)
            _, prediction = outputs.max(1)
            
        return {"prediction": mapping[int(prediction.item())]}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))