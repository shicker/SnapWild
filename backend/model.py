import torch
import torch.nn as nn
import torchvision.models as models

class AnimalRecognitionModel(nn.Module):
    def __init__(self, num_classes=90):
        super(AnimalRecognitionModel, self).__init__()
        # Use a pre-trained ResNet18 model as the base
        self.base_model = models.resnet18(pretrained=False)
        # Replace the final fully connected layer to match the number of classes
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)

# Utility function to load the model
def load_model(model_path, num_classes=90):
    model = AnimalRecognitionModel(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model