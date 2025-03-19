import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from ai_library.base_model import BaseModel

class RCNN(BaseModel):
    def __init__(self, num_classes=10):
        """
        Erstellt ein R-CNN für Bildklassifikation mit PyTorch.
        :param num_classes: Anzahl der Klassen im Datensatz.
        """
        super().__init__()
        self.num_classes = num_classes
        self.build_model()

    def build_model(self):
        """
        Erstellt das R-CNN-Modell und definiert die Schichten.
        """
        self.model = models.resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, self.num_classes)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        print("RCNN wurde erfolgreich erstellt!")

    def train(self, X_train, y_train, epochs=10, batch_size=32):
        """
        Trainingsfunktion für das R-CNN-Modell.
        :param X_train: Tensor mit Trainingsbildern (Batch, 3, 224, 224)
        :param y_train: Tensor mit Labels
        :param epochs: Anzahl der Trainingsdurchläufe
        :param batch_size: Größe der Mini-Batches
        """
        if self.model is None:
            raise ValueError("Das Modell wurde nicht initialisiert!")

        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0
            for images, labels in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

            accuracy = correct / total
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}")

        print("Training abgeschlossen!")

    def predict(self, X):
        """
        Gibt die Vorhersagen für neue Eingaben zurück.
        :param X: Eingabebilder (Tensor, Batch, 3, 224, 224)
        :return: Vorhergesagte Labels
        """
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X)
            _, predictions = torch.max(outputs, 1)
        return predictions
