import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

img_size = 128

dataset_path = r"C:\Users\Ramavath kushwetha\fingerprint\BloodGroupProject\dataset"

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    image_size=(img_size, img_size),
    batch_size=32
)

print("Classes:", train_ds.class_names)

# ✅ MODEL
model = models.Sequential([
    layers.Input(shape=(128,128,3)),   # ✅ FIXED
    layers.Rescaling(1./255),

    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(8, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ✅ TRAIN
history = model.fit(train_ds, epochs=15)

# ✅ SAVE MODEL
model.save(r"C:\Users\Ramavath kushwetha\fingerprint\models\cnn_model.h5")

# ✅ PLOT
plt.plot(history.history['accuracy'])
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()
