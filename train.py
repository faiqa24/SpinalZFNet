# train.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from model import spinal_zfnet


# Define the data loading and preprocessing functions
def load_images_and_labels(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img = io.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
            labels.append(extract_label_from_filename(filename))
    return np.array(images), np.array(labels)

def extract_label_from_filename(filename):
    if 'normal' in filename:
        return 0
    elif 'cyst' in filename:
        return 1
    elif 'tumor' in filename:
        return 2
    elif 'stone' in filename:
        return 3

def preprocess_image(image):
    image = transform.resize(image, (224, 224), mode='reflect')
    image = filters.median(image)
    return image

# Load dataset
DATASET_DIR = 'path_to_your_dataset'
images, labels = load_images_and_labels(DATASET_DIR)
images = np.array([preprocess_image(img) for img in images])
labels = np.array(labels)
images = np.expand_dims(images, axis=-1)

# Split dataset into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Assuming extracted_features_train and extracted_features_test are also defined
# Placeholder for feature extraction (should be replaced with actual feature extraction)
extracted_features_train = np.random.rand(len(X_train), 100)
extracted_features_test = np.random.rand(len(X_test), 100)

# Define the model
input_shape = (224, 224, 1)
extracted_feature_shape = (100, )
model = spinal_zfnet(input_shape, extracted_feature_shape)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, min_delta=0.0001, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_delta=0.0001)
model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max')

# Train the model
history = model.fit(
    [X_train, extracted_features_train], y_train,
    epochs=100,
    batch_size=32,
    validation_data=([X_test, extracted_features_test], y_test),
    callbacks=[early_stopping, reduce_lr, model_checkpoint]
)

# Evaluate the model
y_pred = model.predict([X_test, extracted_features_test])
y_pred_classes = np.argmax(y_pred, axis=1)

def calculate_metrics(y_true, y_pred):
    tp = np.sum(np.logical_and(y_pred == 1, y_true == 1))
    tn = np.sum(np.logical_and(y_pred == 0, y_true == 0))
    fp = np.sum(np.logical_and(y_pred == 1, y_true == 0))
    fn = np.sum(np.logical_and(y_pred == 0, y_true == 1))
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn)
    precision = tp / (tp + fp)
    specificity = tn / (tn + fp)
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)
    
    return accuracy, sensitivity, precision, specificity, f1_score

accuracy, sensitivity, precision, specificity, f1_score = calculate_metrics(y_test, y_pred_classes)

print(f"Accuracy: {accuracy:.4f}")
print(f"Sensitivity: {sensitivity:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"F1 Score: {f1_score:.4f}")

# Plot training & validation accuracy and loss curves
import matplotlib.pyplot as plt

def plot_learning_curves(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    plt.tight_layout()
    plt.show()

plot_learning_curves(history)
