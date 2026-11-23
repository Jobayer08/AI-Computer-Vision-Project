
from pathlib import Path
import csv

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

ROOT = Path(__file__).resolve().parent

IMAGE_DIR = ROOT / "images" / "train"
CSV_PATH = ROOT / "train.csv"

# Example image from your CSV
TARGET_ID = "9a5094b2563a1ef3ff50dc5c7ff71345"

annotations = []

with open(CSV_PATH, "r", encoding="utf-8") as file:

    for row in csv.DictReader(file):

        if row["image_id"] != TARGET_ID:
            continue

        if not row["x_min"].strip():
            continue

        annotations.append(row)

if not annotations:
    raise ValueError("No bounding boxes found!")

image_path = IMAGE_DIR / f"{TARGET_ID}.jpg"

image = Image.open(image_path).convert("RGB")

width, height = image.size

fig, ax = plt.subplots(figsize=(9, 10))

ax.imshow(image)

for annotation in annotations:

    x_min = float(annotation["x_min"])
    y_min = float(annotation["y_min"])
    x_max = float(annotation["x_max"])
    y_max = float(annotation["y_max"])

    box_width = x_max - x_min
    box_height = y_max - y_min

    rectangle = patches.Rectangle(
        (x_min, y_min),
        box_width,
        box_height,
        linewidth=2,
        edgecolor="red",
        facecolor="none"
    )

    ax.add_patch(rectangle)

    label = (
        f'{annotation["class_name"]} '
        f'({annotation["rad_id"]})'
    )

    ax.text(
        x_min,
        max(0, y_min - 10),
        label,
        color="yellow",
        fontsize=9,
        backgroundcolor="black"
    )

ax.set_title(
    f"Chest X-ray: {TARGET_ID}\n"
    f"Image Size: {width} x {height}"
)

ax.axis("off")

output_path = ROOT / "reports" / "bbox_visualization.png"

plt.tight_layout()
plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.show()

print("Saved to:", output_path)
