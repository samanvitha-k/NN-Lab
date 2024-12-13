import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Load and preprocess CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Normalize pixel values between 0 and 1
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# One-hot encode labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Function to build model using VGG16 or VGG19
def build_vgg_model(base_model):
    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False

    # Build the new model
    model = models.Sequential([
        base_model,
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')  # 10 classes for CIFAR-10
    ])
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Load VGG16 and VGG19 models pre-trained on ImageNet
vgg16_base = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
vgg19_base = VGG19(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# Build the models
vgg16_model = build_vgg_model(vgg16_base)
vgg19_model = build_vgg_model(vgg19_base)

# Train and evaluate function
def train_and_evaluate_model(model, model_name):
    print(f"\nTraining {model_name} model...")
    history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))
    
    # Evaluate the model
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"\n{model_name} Test Accuracy: {test_acc:.4f}")
    
    # Plot accuracy
    plt.plot(history.history['accuracy'], label=f'{model_name} Training Accuracy')
    plt.plot(history.history['val_accuracy'], label=f'{model_name} Validation Accuracy')
    plt.title(f'{model_name} Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

# Train and evaluate VGG16 model
train_and_evaluate_model(vgg16_model, "VGG16")

# Train and evaluate VGG19 model
train_and_evaluate_model(vgg19_model, "VGG19")