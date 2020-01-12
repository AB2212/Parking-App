
import numpy as np

import sys, os

from fastapi import FastAPI, UploadFile, File
from starlette.requests import Request
import io

from pydantic import BaseModel


app = FastAPI(__name__)

class ImageType(BaseModel):
    url: str

@app.get("/")
def home():
	return "Home"

@app.post("/predict/")    
def prediction(request: Request, 
	file: bytes = File(...)):
	import cv2
	import cvlib as cv
	from cvlib.object_detection import draw_bbox

	if request.method == "POST":
		image_stream = io.BytesIO(file)
		image_stream.seek(0)
		file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
		frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
		bbox, label, conf = cv.detect_common_objects(frame)
		output_image = draw_bbox(frame, bbox, label, conf)
		num_cars = label.count('car')
		print('Number of cars in the image is '+ str(num_cars))
		return {"num_cars":num_cars}
	return "No post request found"