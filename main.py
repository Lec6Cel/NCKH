from fastapi import FastAPI, UploadFile, Request, File
from fastapi.middleware.cors import CORSMiddleware
from database.db import db
from contextlib import asynccontextmanager
from routes import user
from modelsAI.effB1 import effB1
import torch
from torchvision import transforms
from PIL import Image
from dotenv import load_dotenv
load_dotenv()


model = effB1()
data_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])])
classes = ['Pizza', 'Steak', 'Sushi']


@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.connect()
    model.load_state_dict(torch.load('./modelsAI/effB1.pth'))
    yield
    await db.disconnect()

origins = ['*']
app = FastAPI(lifespan=lifespan)
app.include_router(user.router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post('/predict')
async def get_img(file: UploadFile = File(...)):
    contents = await file.read()
    with open('./modelsAi/image/predict.jpg', 'wb') as f:
        f.write(contents)
    img = Image.open('./modelsAi/image/predict.jpg')
    img_transform = data_transforms(img).unsqueeze(dim=0)
    model.eval()
    with torch.inference_mode():
        y_logits = model(img_transform)
        y_predict = torch.softmax(y_logits, dim=1).argmax(dim=1)
    return {'predict': classes[y_predict]}
