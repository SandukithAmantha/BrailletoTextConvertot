import os
import json
import zipfile
import numpy as np
import cv2
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.backend as K
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Concatenate, AveragePooling2D, GlobalAveragePooling2D, Lambda

from collections import defaultdict

from tensorflow.keras.utils import Sequence

class BrailleDataGenerator(Sequence):
    def __init__(self, image_generator, mask_generator, batch_size):
        self.image_generator = image_generator
        self.mask_generator = mask_generator
        self.batch_size = batch_size

    def __len__(self):
        return len(self.image_generator)

    def __getitem__(self, index):
        images_batch = self.image_generator[index]
        masks_batch = self.mask_generator[index]
        return images_batch, masks_batch


with zipfile.ZipFile("/content/dataset.zip", "r") as zip_ref:
    zip_ref.extractall("images")


def preprocess_image_mask(image, mask, target_size=(224, 224)):
    img_resized = resize(image, target_size, preserve_range=True, mode='reflect', anti_aliasing=True).astype(np.uint8)
    mask_resized = resize(mask, target_size, order=0, preserve_range=True, mode='reflect', anti_aliasing=False).astype(
        np.uint8)
    return img_resized, mask_resized


def create_mask_from_annotations(annotations, img_shape):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    for annotation in annotations:
        segmentation = annotation['segmentation'][0]
        for i in range(0, len(segmentation), 2):
            x, y = int(segmentation[i]), int(segmentation[i + 1])
            if 0 <= x < img_shape[1] and 0 <= y < img_shape[0]:
                mask[y, x] = 1
    return mask


with open("annotations.json") as f:
    annotations = json.load(f)

annotations_by_image_id = defaultdict(list)
for img, anno in zip(annotations["images"], annotations["annotations"]):
    annotations_by_image_id[img["id"]].append(anno)

image_dir = "images/dataset"
image_paths = sorted([os.path.join(image_dir, file) for file in os.listdir(image_dir)])

images, masks = [], []
for img_path in image_paths:
    img_name = os.path.basename(img_path)

    # Find the corresponding image ID and annotations
    image_id = None
    for img in annotations["images"]:
        if img["file_name"] == img_name:
            image_id = img["id"]
            break
    image_annotations = annotations_by_image_id[image_id]

    img = imread(img_path)
    mask = create_mask_from_annotations(image_annotations, img.shape)

    img, mask = preprocess_image_mask(img, mask)

    images.append(img)
    masks.append(mask)

images = np.array(images)
masks = np.array(masks)
masks = masks[..., np.newaxis]  # Add a channel dimension

# //////////////////////////////////////////////////////////////////////////////////////////

# Split the data into training and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(images, masks, test_size=0.2, random_state=42)

# Data augmentation
data_gen_args = dict(rotation_range=20, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2)
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

image_datagen.fit(X_train, augment=True, seed=1)
mask_datagen.fit(Y_train, augment=True, seed=1)

image_generator = image_datagen.flow(X_train, batch_size=32, seed=1)
mask_generator = mask_datagen.flow(Y_train, batch_size=32, seed=1)
train_generator = zip(image_generator, mask_generator)


def aspp(x, filters):
    x1 = DepthwiseConv2D(kernel_size=3, dilation_rate=6, padding='same')(x)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = Conv2D(filters, kernel_size=1)(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)

    x2 = DepthwiseConv2D(kernel_size=3, dilation_rate=12, padding='same')(x)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x2 = Conv2D(filters, kernel_size=1)(x2)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)

    x3 = DepthwiseConv2D(kernel_size=3, dilation_rate=18, padding='same')(x)
    x3 = BatchNormalization()(x3)
    x3 = Activation('relu')(x3)
    x3 = Conv2D(filters, kernel_size=1)(x3)
    x3 = BatchNormalization()(x3)
    x3 = Activation('relu')(x3)

    x4 = GlobalAveragePooling2D()(x)
    x4 = K.expand_dims(x4, axis=1)
    x4 = K.expand_dims(x4, axis=2)
    x4 = Conv2D(filters, kernel_size=1)(x4)
    x4 = BatchNormalization()(x4)
    x4 = Activation('relu')(x4)
    x4 = UpSampling2D(size=(int(x.shape[1]), int(x.shape[2])), interpolation='bilinear')(x4)

    return Concatenate()([x1, x2, x3, x4])


def custom_mobilenetv2(input_shape):
    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

    # Create the new input layer
    inputs = Input(input_shape)

    # Convert single-channel input to three-channel input
    def single_to_three_channel(x):
        return K.concatenate([x, x, x], axis=-1)

    x = Lambda(single_to_three_channel)(inputs)

    # Replace the first Conv2D layer and fix the Add layers
    for i, layer in enumerate(base_model.layers[1:]):
        if isinstance(layer, tf.keras.layers.Add):
            input_tensors = [x, x]
            x = layer(input_tensors)
        else:
            x = layer(x)

    return Model(inputs=inputs, outputs=x)


# DeepLab v3+ model with a simple encoder-decoder architecture
def Deeplabv3plus(input_size=(224, 224, 1)):
    inputs = Input(input_size)

    # Encoder
    base_model = custom_mobilenetv2(input_shape=input_size)

    x = inputs
    for layer in base_model.layers[1:]:
        x = layer(x)

    x = base_model.get_layer('block_13_expand_relu').output

    # ASPP (Atrous Spatial Pyramid Pooling)
    aspp_output = aspp(x, 256)

    # Decoder
    x = Conv2D(256, (1, 1), padding='same', use_bias=False)(aspp_output)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D(size=(4, 4), interpolation='bilinear')(x)
    low_level_features = base_model.get_layer('block_1_expand_relu').output
    low_level_features = Conv2D(48, (1, 1), padding='same', use_bias=False)(low_level_features)
    low_level_features = BatchNormalization()(low_level_features)
    low_level_features = Activation('relu')(low_level_features)

    x = Concatenate()([x, tf.image.resize(low_level_features, (56, 56))])
    x = Conv2D(256, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D(size=(4, 4), interpolation='bilinear')(x)

    outputs = Conv2D(2, (1, 1), activation='softmax')(x)

    return Model(inputs=inputs, outputs=outputs)


train_datagen = ImageDataGenerator(**data_gen_args)
val_datagen = ImageDataGenerator()

train_image_generator = train_datagen.flow(X_train, seed=1)
train_mask_generator = train_datagen.flow(Y_train, seed=1)
val_image_generator = val_datagen.flow(X_val, seed=1)
val_mask_generator = val_datagen.flow(Y_val, seed=1)

train_generator = zip(train_image_generator, train_mask_generator)
val_generator = zip(val_image_generator, val_mask_generator)

# Compile and train the model
model = Deeplabv3plus(input_size=(224, 224, 1))
model.compile(optimizer=Adam(learning_rate=1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

callbacks = [
    ModelCheckpoint('deeplab_braille.h5', save_best_only=True, monitor='val_loss'),
    EarlyStopping(monitor='val_loss', patience=10)
]

train_generator = BrailleDataGenerator(train_image_generator, train_mask_generator, batch_size=32)
val_generator = BrailleDataGenerator(val_image_generator, val_mask_generator, batch_size=32)

model.fit(train_generator,
          steps_per_epoch=len(train_generator),
          epochs=100,
          callbacks=callbacks,
          validation_data=val_generator,
          validation_steps=len(val_generator))