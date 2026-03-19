import cv2
import os

# 🔹 Function: Advanced Preprocessing
def preprocess_image(path):
    img = cv2.imread(path, 0)  # grayscale

    if img is None:
        print("❌ Error loading:", path)
        return None

    # Resize
    img = cv2.resize(img, (128,128))

    # Noise removal
    img = cv2.GaussianBlur(img, (5,5), 0)

    # Contrast enhancement
    img = cv2.equalizeHist(img)

    return img


# 🔹 Main execution (same as your previous working logic)
path = r"C:\Users\Ramavath kushwetha\fingerprint\BloodGroupProject\dataset\A+"

files = os.listdir(path)

if len(files) == 0:
    print("❌ Folder is empty")
else:
    print("✅ Total images:", len(files))


for img_name in files:
    img_path = os.path.join(path, img_name)

    processed = preprocess_image(img_path)

    if processed is None:
        continue

    # Show processed image
    cv2.imshow("Processed Fingerprint", processed)

    key = cv2.waitKey(0)   # wait until key press
    if key == 27:          # press ESC to exit
        break

cv2.destroyAllWindows()
