import torch
import torch.nn as nn
from torchvision.transforms import v2
import cv2
import json
from pathlib import Path


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.convolution_part = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.flatten = nn.Flatten()

        self.dense_layers = nn.Sequential(
            nn.Linear(12*12*128, 120),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(84, 3)
        )

        # We do not use Softmax, because the loss function CrossEntropyLoss needs pure logits. 
    
    def forward(self, x):
        x = self.convolution_part(x)
        x = self.flatten(x)
        x = self.dense_layers(x)
        return x


def get_data():
    path_images = Path("signs/classification/images/")
    path_labels = Path("signs/classification/annotations/")

    transform = v2.Compose([
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True), 
        v2.Resize((96, 96))
    ])

    list_images = []
    list_labels = []

    class_map = {"speed_limit_30": 0, "speed_limit_60": 1, "speed_limit_90": 2}

    all_image_files = sorted([f for f in path_images.rglob('*') if f.is_file()])

    for img_path in all_image_files:
        json_path = path_labels / img_path.relative_to(path_images).with_suffix('.json')
        
        if json_path.exists():
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)

            label_name = json_data.get("class")
            if label_name in class_map:
                image = cv2.imread(str(img_path))
                if image is None:
                    continue

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
                image = transform(image)

                list_images.append(image)
                # On ajoute le label en même temps, au même index
                list_labels.append(class_map[label_name])
    print(f"{len(list_images)}, {len(list_labels)}")
    
    # On convertit les labels en tenseur pour manipuler du PyTorch pur
    return torch.stack(list_images), torch.tensor(list_labels)

def process_image(image):
    '''
    transform the input image (BGR format) to be able to go to the model. 
    '''
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

    transform = v2.Compose([
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True), 
        v2.Resize((96, 96))
    ])
    image = transform(image)

    return torch.stack([image])

def validation_test():
    model = CNN()
    model.load_state_dict(torch.load('mejor_weights.pth', weights_only=True))
    model.eval()

    validation_batch, labels = get_data()

    # Passe tout le batch d'un coup dans le modèle
    with torch.no_grad():
        pred = model(validation_batch)
        guess = pred.argmax(1)

    # Compteurs pour vérifier l'efficacité par classe
    for c in [0, 1, 2]:
        total_classe = (labels == c).sum().item()
        vrais_positifs = ((labels == c) & (guess == c)).sum().item()
        pourcentage = (vrais_positifs / total_classe * 100) if total_classe > 0 else 0
        print(f"Classe {c} -> Précision: {vrais_positifs}/{total_classe} ({pourcentage:.1f}%)")

    print("\n--- Détail des prédictions (15 premiers éléments) ---")
    for i in range(len(validation_batch)):
        print(f"Image {i:02d} | Vrai Label: {labels[i].item()} -> Prédiction Modèle: {guess[i]}, logits: {pred[i]}")

if __name__ == "__main__":
    model = CNN()
    model.load_state_dict(torch.load('mejor_weights.pth', weights_only=True))
    model.eval()

    path = "test_dataset/class_0/img_0.png" # Example
    image = cv2.imread(path)
    image = process_image(image)

    with torch.no_grad():
        pred = model(image)
        guess = pred.argmax(1)
    print(guess)