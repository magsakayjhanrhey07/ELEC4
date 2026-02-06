import cv2
import os

INPUT_DIR = "input_images"
OUTPUT_DIR = "output_images"

def process_images():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for filename in os.listdir(INPUT_DIR):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(INPUT_DIR, filename)

            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            clahe_img = clahe.apply(gray)

            # Gaussian Blur
            blurred = cv2.GaussianBlur(clahe_img, (5,5), 0)

            # Adaptive Threshold
            thresh = cv2.adaptiveThreshold(
                blurred, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )

            # Invert Colors
            inverted = cv2.bitwise_not(thresh)

            output_path = os.path.join(OUTPUT_DIR, filename)
            cv2.imwrite(output_path, inverted)

    return True


if __name__ == "__main__":
    process_images()
    print("Image processing completed.")
