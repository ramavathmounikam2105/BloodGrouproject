
import cv2
import tensorflow as tf
import numpy as np
import os
import sys

model_path = r"C:\Users\Ramavath kushwetha\fingerprint\models\cnn_model.h5"

if not os.path.exists(model_path):
    print("❌ Model file not found")
    sys.exit()

model = tf.keras.models.load_model(model_path, compile=False)


img_path = r"C:\Users\Ramavath kushwetha\fingerprint\test.BMP"

if not os.path.exists(img_path):
    print("❌ Image file not found")
    sys.exit()


def preprocess_image(path):
    img = cv2.imread(path)   # ✅ KEEP COLOR (3 CHANNEL)

    if img is None:
        return None

    # Resize
    img = cv2.resize(img, (128, 128))

    # Optional: slight blur (safe)
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # Normalize
    img = img / 255.0

    return img


# =========================
# PREPROCESS IMAGE
# =========================
img = preprocess_image(img_path)

if img is None:
    print("❌ Error processing image")
    sys.exit()

# Expand dims → (1,128,128,3)
img_input = np.expand_dims(img, axis=0)


# =========================
# PREDICTION
# =========================
prediction = model.predict(img_input)

classes = ['A+','A-','AB+','AB-','B+','B-','O+','O-']  # ✅ match training order!

result = classes[np.argmax(prediction)]
confidence = np.max(prediction) * 100


# =========================
# OUTPUT
# =========================
print("\n===== RESULT =====")
print("Predicted Blood Group:", result)
print("Confidence:", round(confidence, 2), "%")

if confidence < 60:
    print("⚠️ Low confidence prediction. Try clearer fingerprint.")


# =========================
# DISPLAY IMAGE (NO MARKING)
# =========================
display_img = cv2.imread(img_path)
display_img = cv2.resize(display_img, (400, 400))

cv2.imshow("Fingerprint Image", display_img)
cv2.waitKey(0)
cv2.destroyAllWindows()