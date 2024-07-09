import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from skimage import io, transform
from skimage.filters import median
from skimage.morphology import disk
import cv2
import matplotlib.pyplot as plt
import pywt
from skimage.feature import greycomatrix, greycoprops
from scipy.ndimage import gaussian_filter
from skimage import measure


# Set dataset directory
DATASET_DIR = 'path_to_dataset_directory'
IMG_SIZE = 224
BATCH_SIZE = 32
SEED = 42

# Load and preprocess dataset
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
    image = transform.resize(image, (IMG_SIZE, IMG_SIZE), mode='reflect')
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = median(image, disk(3))
    return image

images, labels = load_images_and_labels(DATASET_DIR)
images = np.array([preprocess_image(img) for img in images])
labels = np.array(labels)
images = np.expand_dims(images, axis=-1)

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=SEED)

# Define EfficientNet-based segmentation model (ENet) in TensorFlow
def enet_model(input_shape):
    base_model = tf.keras.applications.EfficientNetB0(input_shape=input_shape, include_top=False, weights='imagenet')

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = tf.keras.layers.Conv2DTranspose(1280, (12, 39), strides=1, padding='valid', activation='relu')(x)
    x = tf.keras.layers.Conv2DTranspose(672, (1, 1), strides=1, padding='valid', activation='relu')(x)
    x = tf.keras.layers.Conv2DTranspose(672, (12, 39), strides=1, padding='valid', activation='relu')(x)
    x = tf.keras.layers.Conv2DTranspose(480, (1, 1), strides=1, padding='valid', activation='relu')(x)
    x = tf.keras.layers.Conv2DTranspose(240, (1, 1), strides=1, padding='valid', activation='relu')(x)
    x = tf.keras.layers.Conv2DTranspose(240, (24, 79), strides=1, padding='valid', activation='relu')(x)
    x = tf.keras.layers.Conv2DTranspose(144, (1, 1), strides=1, padding='valid', activation='relu')(x)
    x = tf.keras.layers.Conv2DTranspose(144, (46, 155), strides=1, padding='valid', activation='relu')(x)
    x = tf.keras.layers.Conv2DTranspose(96, (93, 310), strides=1, padding='valid', activation='relu')(x)
    x = tf.keras.layers.Conv2DTranspose(32, (46, 155), strides=1, padding='valid', activation='relu')(x)
    x = tf.keras.layers.Conv2DTranspose(32, (185, 620), strides=1, padding='valid', activation='relu')(x)
    outputs = tf.keras.layers.Conv2DTranspose(1, (185, 620), strides=1, padding='valid', activation='sigmoid')(x)
    
    model = tf.keras.models.Model(inputs=base_model.input, outputs=outputs)
    return model

# Define input shape for ENet model
input_shape = (IMG_SIZE, IMG_SIZE, 1)
enet = enet_model((IMG_SIZE, IMG_SIZE, 3))

# Compile ENet model
enet.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Placeholder: Train ENet model on the dataset (assuming labels for segmentation are available)
# enet.fit(X_train, y_train_segmentation, batch_size=BATCH_SIZE, epochs=50, validation_data=(X_test, y_test_segmentation))

# Perform segmentation on the test dataset using ENet model
segmented_images = enet.predict(X_test)

# Define data augmentation for segmented images
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Apply augmentation to segmented images
def augment_images(images):
    augmented_images = []
    for img in images:
        img = np.expand_dims(img, axis=0)  # Expand dims to fit ImageDataGenerator
        aug_iter = datagen.flow(img)
        aug_img = next(aug_iter)[0].astype(np.uint8)
        augmented_images.append(aug_img)
    return np.array(augmented_images)

augmented_images = augment_images(segmented_images)

# Save preprocessed and augmented data to disk 
np.save('X_train_augmented.npy', augmented_images)
np.save('X_test_seg.npy', segmented_images)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)


# Function to extract SURF features
def extract_surf_features(image):
    surf = cv2.xfeatures2d.SURF_create()
    keypoints, descriptors = surf.detectAndCompute(image, None)
    return descriptors

# Function to extract WLD-based DWT with HOG features
def extract_wld_dwt_hog_features(image):
    # Discrete Wavelet Transform
    coeffs = pywt.dwt2(image, 'haar')
    LL, (LH, HL, HH) = coeffs

    # Weber Local Descriptor (WLD)
    def weber_local_descriptor(image):
        differential_excitation = np.arctan((image - np.mean(image)) / (image + 0.0001))
        orientation = np.arctan2(gaussian_filter(image, sigma=1, order=(1, 0)), gaussian_filter(image, sigma=1, order=(0, 1)))
        return differential_excitation, orientation

    wld_LL = weber_local_descriptor(LL)
    wld_LH = weber_local_descriptor(LH)
    wld_HL = weber_local_descriptor(HL)
    wld_HH = weber_local_descriptor(HH)

    # Histogram of Oriented Gradients (HOG)
    def hog_descriptor(image):
        hog = cv2.HOGDescriptor()
        h = hog.compute(image)
        return h

    hog_LL = hog_descriptor(LL)
    hog_LH = hog_descriptor(LH)
    hog_HL = hog_descriptor(HL)
    hog_HH = hog_descriptor(HH)

    features = np.concatenate((wld_LL[0].flatten(), wld_LL[1].flatten(),
                               wld_LH[0].flatten(), wld_LH[1].flatten(),
                               wld_HL[0].flatten(), wld_HL[1].flatten(),
                               wld_HH[0].flatten(), wld_HH[1].flatten(),
                               hog_LL.flatten(), hog_LH.flatten(),
                               hog_HL.flatten(), hog_HH.flatten()))
    return features

# Function to extract shape features
def extract_shape_features(image):
    labeled_img = measure.label(image)
    regions = measure.regionprops(labeled_img)

    shape_features = []
    for region in regions:
        area = region.area
        perimeter = region.perimeter
        major_axis_length = region.major_axis_length
        minor_axis_length = region.minor_axis_length
        convex_hull = region.convex_area

        compactness = perimeter ** 2 / (4 * np.pi * area)
        solidity = area / convex_hull
        eccentricity = major_axis_length / minor_axis_length

        shape_features.extend([area, perimeter, major_axis_length, minor_axis_length, convex_hull, compactness, solidity, eccentricity])

    return shape_features

# Function to extract statistical features
def extract_statistical_features(image):
    mean = np.mean(image)
    entropy = -np.sum(image * np.log2(image + 0.0001))
    
    glcm = greycomatrix(image.astype(np.uint8), distances=[1], angles=[0], symmetric=True, normed=True)
    contrast = greycoprops(glcm, 'contrast')[0, 0]
    correlation = greycoprops(glcm, 'correlation')[0, 0]

    return [mean, entropy, contrast, correlation]

# Example of feature extraction on augmented CT images
def extract_features_from_images(images):
    surf_features = []
    wld_dwt_hog_features = []
    shape_features = []
    statistical_features = []

    for img in images:
        surf_features.append(extract_surf_features(img))
        wld_dwt_hog_features.append(extract_wld_dwt_hog_features(img))
        shape_features.append(extract_shape_features(img))
        statistical_features.append(extract_statistical_features(img))

    return {
        'surf_features': surf_features,
        'wld_dwt_hog_features': wld_dwt_hog_features,
        'shape_features': shape_features,
        'statistical_features': statistical_features
    }

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input

import tensorflow as tf
from tensorflow.keras import layers, models, Input

# Define SpinalNet block in TensorFlow/Keras
def spinal_net_block(inputs, units, dropout_rate=0.5):
    split_inputs = tf.split(inputs, num_or_size_splits=2, axis=1)
    spinal_outputs = []
    
    for i in range(len(split_inputs)):
        if i == 0:
            x = layers.Dense(units, activation='relu')(split_inputs[i])
        else:
            x = layers.Concatenate()([split_inputs[i], spinal_outputs[-1]])
            x = layers.Dense(units, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)
        spinal_outputs.append(x)
    
    spinal_output = layers.Concatenate()(spinal_outputs)
    return spinal_output

def spinal_net_model(input_shape, layer_width=128, num_classes=4):
    inputs = Input(shape=input_shape)
    x = layers.Flatten()(inputs)
    spinal_output = spinal_net_block(x, units=layer_width)
    spinal_output = spinal_net_block(spinal_output, units=layer_width)
    spinal_output = spinal_net_block(spinal_output, units=layer_width)
    spinal_output = spinal_net_block(spinal_output, units=layer_width)
    outputs = layers.Dense(num_classes, activation='softmax')(spinal_output)
    return models.Model(inputs, outputs, name='SpinalNet')

# Define SpinalZFNet layer
def spinal_zfnet_layer(spinal_output, extracted_features):
    def regression_model(inputs):
        x = layers.Dense(128, activation='relu')(inputs)
        x = layers.Dense(4, activation='softmax')(x)
        return x
    
    combined_inputs = layers.Concatenate()([spinal_output, extracted_features])
    combined_output = regression_model(combined_inputs)
    return combined_output

# Define ZFNet model
def zf_net_model(input_shape):
    inputs = Input(shape=input_shape)
    x = layers.Conv2D(96, (7, 7), strides=2, activation='relu')(inputs)
    x = layers.MaxPooling2D((3, 3), strides=2)(x)
    
    x = layers.Conv2D(256, (5, 5), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=2)(x)
    
    x = layers.Conv2D(384, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(384, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=2)(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    
    outputs = layers.Dense(4, activation='softmax')(x)
    return models.Model(inputs, outputs, name='ZFNet')

# Define the complete SpinalZFNet model
def spinal_zfnet(input_shape, extracted_feature_shape):
    spinal_input = Input(shape=input_shape)
    extracted_feature_input = Input(shape=extracted_feature_shape)
    
    # SpinalNet
    spinal_model = spinal_net_model(input_shape)
    spinal_output = spinal_model(spinal_input)
    
    # SpinalZFNet layer
    spinal_zfnet_output = spinal_zfnet_layer(spinal_output, extracted_feature_input)
    
    # ZFNet
    zfnet_model = zf_net_model((150, 150, 3))
    final_output = zfnet_model(spinal_zfnet_output)
    
    return models.Model(inputs=[spinal_input, extracted_feature_input], outputs=final_output, name='SpinalZFNet')

# Assuming the input shape of the CT images and the extracted features
input_shape = (150, 150, 1)
extracted_feature_shape = (100, 100)  

# Create the SpinalZFNet model
model = spinal_zfnet(input_shape, extracted_feature_shape)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Display the model summary
model.summary()
