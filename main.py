import random
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2

def preprocess_image(image):
    blurred_image = cv2.GaussianBlur(image, (3, 3), 0)
    denoised_image = cv2.medianBlur(blurred_image, 3)
    return denoised_image

train_data = "./baseDeDados/train"
test_data = "./baseDeDados/test"
image_size = (244, 244)
batch_size = 16
classes = ['Covid', 'Normal', 'Viral Pneumonia']

fig, axes = plt.subplots(1, len(classes), figsize=(12, 4))
for i, cls in enumerate(classes):
    img_path = os.path.join(train_data, cls, random.choice(os.listdir(os.path.join(train_data, cls))))
    img = plt.imread(img_path)
    axes[i].imshow(img)
    axes[i].set_title(cls)
    axes[i].axis('off')

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1.0/255.0, rotation_range = 0.5, width_shift_range = 0.2,
    height_shift_range = 0.2, preprocessing_function=preprocess_image, shear_range = 0.2, zoom_range = 0.1, horizontal_flip = True, fill_mode = 'nearest')
train_ds = train_datagen.flow_from_directory(train_data, target_size=image_size, batch_size=batch_size, class_mode='categorical', shuffle=True)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255, preprocessing_function=preprocess_image)
test_ds = test_datagen.flow_from_directory( test_data, target_size=image_size, batch_size=batch_size, class_mode='categorical', shuffle=False)

cnn_model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu", input_shape=(image_size[0], image_size[1], 3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu", input_shape=(image_size[0], image_size[1], 3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(3, activation="softmax")
])

early_stopping = tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)

cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

base_model = tf.keras.applications.ResNet50(
    include_top=False,
    weights='imagenet',
    input_shape=(image_size[0], image_size[1], 3)
)

for layer in base_model.layers:
    layer.trainable = False

resnet_model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

resnet_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

resnet_history = resnet_model.fit(train_ds, validation_data=test_ds, epochs=20, callbacks=[early_stopping])

cnn_history = cnn_model.fit(train_ds, validation_data=test_ds, epochs=20, callbacks=[early_stopping])

cnn_model.summary()
resnet_model.summary()

plt.figure(figsize=(12, 6))


plt.subplot(1, 2, 1)
plt.plot(cnn_history.history['accuracy'], label='CNN Training Accuracy')
plt.plot(cnn_history.history['val_accuracy'], label='CNN Validation Accuracy')
plt.plot(resnet_history.history['accuracy'], label='ResNet Training Accuracy')
plt.plot(resnet_history.history['val_accuracy'], label='ResNet Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(cnn_history.history['loss'], label='CNN Training Loss')
plt.plot(cnn_history.history['val_loss'], label='CNN Validation Loss')
plt.plot(resnet_history.history['loss'], label='ResNet Training Loss')
plt.plot(resnet_history.history['val_loss'], label='ResNet Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

def classify_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=image_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)

    cnn_prediction = cnn_model.predict(img_array)

    cnn_class = np.argmax(cnn_prediction[0])

    classes = ['Covid', 'Normal', 'Viral Pneumonia']  

    print("CNN Model Prediction:")
    print(f"Class: {classes[cnn_class]}, Confidence: {cnn_prediction[0][cnn_class] * 100:.2f}%")