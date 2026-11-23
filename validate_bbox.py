
from pathlib import Path
from collections import defaultdict
from PIL import Image
import csv

ROOT = Path(__file__).resolve().parent

IMAGE_DIR = ROOT / "images" / "train"
CSV_PATH = ROOT / "train.csv"

image_dimensions = {}

valid_boxes = 0
invalid_boxes = 0
missing_images = 0
partial_boxes = 0
missing_boxes = 0

errors = []

with open(CSV_PATH, "r", encoding="utf-8") as file:

    reader = csv.DictReader(file)

    for row in reader:

        image_id = row["image_id"]

        coords = [
            row["x_min"],
            row["y_min"],
            row["x_max"],
            row["y_max"]
        ]

        # Skip No finding rows
        if all(not x.strip() for x in coords):
            missing_boxes += 1
            continue

        if any(not x.strip() for x in coords):
            partial_boxes += 1
            continue

        image_path = IMAGE_DIR / f"{image_id}.jpg"

        if image_id not in image_dimensions:

            if not image_path.exists():
                missing_images += 1
                errors.append(
                    f"Missing image: {image_id}"
                )
                continue

            with Image.open(image_path) as image:
                image_dimensions[image_id] = image.size

        width, height = image_dimensions[image_id]

        x_min, y_min, x_max, y_max = map(float, coords)

        is_valid = (
            0 <= x_min < x_max <= width
            and
            0 <= y_min < y_max <= height
        )

        if is_valid:
            valid_boxes += 1
        else:
            invalid_boxes += 1

            if len(errors) < 10:
                errors.append({
                    "image_id": image_id,
                    "class": row["class_name"],
                    "coordinates": coords,
                    "image_size": (width, height)
                })


print("\n========== BBOX VALIDATION ==========")

print("Valid boxes:", valid_boxes)
print("Invalid boxes:", invalid_boxes)
print("Missing boxes:", missing_boxes)
print("Partial boxes:", partial_boxes)
print("Missing images:", missing_images)

print("\nErrors:")

for error in errors:
    print(error)

print("\nUnique images checked:",
      len(image_dimensions))
