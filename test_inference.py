import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best.pt"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_default_image() -> Path:
    """Return the first real test image, excluding generated mask files."""
    for image_path in sorted((BASE_DIR / "test").rglob("*")):
        if image_path.suffix.lower() in SUPPORTED_EXTENSIONS and "_mask" not in image_path.stem.lower():
            return image_path
    raise FileNotFoundError("No test image was found under the test directory.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run classification inference on a test image.")
    parser.add_argument("image", nargs="?", type=Path, help="Image path, relative to the project root or absolute.")
    args = parser.parse_args()

    image_path = args.image or find_default_image()
    if not image_path.is_absolute():
        image_path = BASE_DIR / image_path
    image_path = image_path.resolve()

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    model = YOLO(str(MODEL_PATH))
    results = model.predict(source=str(image_path), verbose=False)
    probabilities = results[0].probs
    if probabilities is None:
        raise RuntimeError("This model did not return classification probabilities.")

    class_names = model.names
    predicted_class = class_names[int(probabilities.top1)]
    confidence = float(probabilities.top1conf) * 100
    print(f"\nPredicted class: {predicted_class} ({confidence:.2f}% confidence)\n")

    image_rgb = Image.open(image_path).convert("RGB")
    plt.figure(figsize=(6, 6))
    plt.imshow(image_rgb)
    plt.title(f"{predicted_class} ({confidence:.1f}%)", fontsize=12, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()