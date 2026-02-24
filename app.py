# -----------------------------
# 0. INSTALL REQUIRED PACKAGES
# -----------------------------
!pip install -q kaggle tensorflow pillow scikit-learn matplotlib

# -----------------------------
# 1. KAGGLE DATASET DOWNLOAD
# -----------------------------
import os, zipfile

os.environ['KAGGLE_CONFIG_DIR'] = os.path.expanduser("~/.kaggle")

!kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p /content

with zipfile.ZipFile("/content/chest-xray-pneumonia.zip", 'r') as zip_ref:
    zip_ref.extractall("/content/chest_xray_data")

BASE_PATH = "/content/chest_xray_data/chest_xray"

# -----------------------------
# 2. IMAGE PREPROCESSING
# -----------------------------
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (160,160)
BATCH_SIZE = 32

train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)

test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    BASE_PATH + "/train",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

val_data = test_gen.flow_from_directory(
    BASE_PATH + "/val",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

test_data = test_gen.flow_from_directory(
    BASE_PATH + "/test",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

# -----------------------------
# 3. MOBILE NET MODEL
# -----------------------------
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(160,160,3)
)

base_model.trainable = False

cnn_model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])

cnn_model.compile(
    optimizer=Adam(0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

cnn_model.summary()

# -----------------------------
# 4. CALLBACKS
# -----------------------------
from tensorflow.keras.callbacks import EarlyStopping

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )
]

# -----------------------------
# 5. INITIAL TRAINING
# -----------------------------
history = cnn_model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    steps_per_epoch=50,
    validation_steps=10,
    callbacks=callbacks
)

# -----------------------------
# 6. FINE-TUNING
# -----------------------------
for layer in cnn_model.layers[0].layers[-30:]:
    layer.trainable = True

cnn_model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

fine_tune_history = cnn_model.fit(
    train_data,
    validation_data=val_data,
    epochs=5,
    steps_per_epoch=50,
    validation_steps=10
)

# -----------------------------
# 7. FINAL EVALUATION
# -----------------------------
loss, accuracy = cnn_model.evaluate(test_data)
print(f"🔥 Fine-Tuned Test Accuracy: {accuracy*100:.2f}%")

# -----------------------------
# 8. SAVE FINAL MODEL FOR STREAMLIT
# -----------------------------
cnn_model.save("lung_model.keras")
print("✅ Final Model Saved as: lung_model.keras")

# Download to local system
from google.colab import files
files.download("lung_model.keras")

print("🎓 TRAINING COMPLETED SUCCESSFULLY")
