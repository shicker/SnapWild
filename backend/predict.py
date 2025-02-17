import torch
from torchvision import transforms
from PIL import Image
from model import AnimalRecognitionModel

# Load the trained model
model = AnimalRecognitionModel(num_classes=90)
model.load_state_dict(torch.load('backend/models/animal_model.pth'))
model = model.to('cuda')
model.eval()

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_animal(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to('cuda')

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        return predicted.item()