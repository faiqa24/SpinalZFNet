# model.py
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

def spinal_zfnet_layer(spinal_output, extracted_features):
    def regression_model(inputs):
        x = layers.Dense(128, activation='relu')(inputs)
        x = layers.Dense(4, activation='softmax')(x)
        return x
    
    combined_inputs = layers.Concatenate()([spinal_output, extracted_features])
    combined_output = regression_model(combined_inputs)
    return combined_output

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

def spinal_zfnet(input_shape, extracted_feature_shape):
    spinal_input = Input(shape=input_shape)
    extracted_feature_input = Input(shape=extracted_feature_shape)
    
    # SpinalNet
    spinal_model = spinal_net_model(input_shape)
    spinal_output = spinal_model(spinal_input)
    
    # SpinalZFNet layer
    spinal_zfnet_output = spinal_zfnet_layer(spinal_output, extracted_feature_input)
    
    # ZFNet
    zfnet_model = zf_net_model((224, 224, 3))
    final_output = zfnet_model(spinal_zfnet_output)
    
    return models.Model(inputs=[spinal_input, extracted_feature_input], outputs=final_output, name='SpinalZFNet')
