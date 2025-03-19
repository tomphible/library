import sys
import os
import torch
sys.path.append(os.path.join(os.path.dirname(__file__), 'ai-library'))
from ai_library.models.cnn import CNN
from ai_library.models.rcnn import RCNN
import matplotlib.pyplot as plt

def train_model(model_class, input_size):
    # Dummy data for example training
    X_train = torch.rand(100, 3, input_size, input_size)  # 100 RGB images
    y_train = torch.randint(0, 2, (100,))  # 100 random labels for 10 classes

    model = model_class(num_classes=2)
    model.train(X_train, y_train, epochs=5, batch_size=32)
    return model

def make_predictions(model, input_size):
    # Predictions on new data
    X_test = torch.rand(10, 3, input_size, input_size)  # 10 new images
    predictions = model.predict(X_test)
    print(f"Predictions from {model.__class__.__name__}:", predictions)
    show_image_with_prediction(X_test[0], predictions)

def show_image_with_prediction(image, prediction):
    plt.imshow(image.permute(1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
    plt.title(f"Prediction: {prediction}")
    plt.axis('off')
    plt.show()

def main():
    # Train and evaluate CNN model
    cnn_model = train_model(CNN, 32)
    make_predictions(cnn_model, 32)

    # Train and evaluate RCNN model
    rcnn_model = train_model(RCNN, 224)
    make_predictions(rcnn_model, 224)


if __name__ == "__main__":
    main()
