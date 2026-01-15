import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import requests

st.title("Real-Time Image Classification (ResNet-18)")

@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=True)
    model.eval()
    return model

model = load_model()

labels_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
labels = requests.get(labels_url).text.splitlines()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])

img_file = st.camera_input("Capture an image")

if img_file:
    image = Image.open(img_file)
    st.image(image, caption="Captured Image", use_column_width=True)

    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs[0], dim=0)

    top5 = torch.topk(probs, 5)

    results = {
        "Label": [labels[i] for i in top5.indices],
        "Probability": [float(p) for p in top5.values]
    }

    st.subheader("Top 5 Predictions")
    st.table(results)