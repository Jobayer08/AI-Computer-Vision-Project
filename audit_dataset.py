
from pathlib import Path
import csv
from collections import Counter, defaultdict

# =====================================
# 1. Dataset paths
# =====================================

ROOT = Path(__file__).resolve().parent

TRAIN_DIR = ROOT / "images" / "train"
TEST_DIR = ROOT / "images" / "test"
CSV_PATH = ROOT / "train.csv"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# =====================================
# 2. Read image IDs
# =====================================

def get_image_ids(folder):
    if not folder.exists():
        raise FileNotFoundError(folder)

    return {
        file.stem
        for file in folder.rglob("*")
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
    }


train_ids = get_image_ids(TRAIN_DIR)
test_ids = get_image_ids(TEST_DIR)

# =====================================
# 3. Load CSV annotations
# =====================================

if not CSV_PATH.exists():
    raise FileNotFoundError(CSV_PATH)

with CSV_PATH.open("r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    rows = list(reader)

required_columns = {
    "image_id", "class_name", "class_id", "rad_id",
    "x_min", "y_min", "x_max", "y_max"
}

if not required_columns.issubset(reader.fieldnames or []):
    raise ValueError("CSV is missing required columns")


# =====================================
# 4. Basic statistics
# =====================================

csv_ids = {row["image_id"] for row in rows}

print("\n========== DATASET SUMMARY ==========")

print("Train images:", len(train_ids))
print("Test images:", len(test_ids))
print("Total images:", len(train_ids) + len(test_ids))
print("CSV rows:", len(rows))
print("Unique annotated images:", len(csv_ids))

# =====================================
# 5. Image ID matching
# =====================================

print("\n========== IMAGE ID VALIDATION ==========")

print("CSV IDs missing from train folder:",
      len(csv_ids - train_ids))

print("Train images without CSV annotation:",
      len(train_ids - csv_ids))

print("CSV IDs overlapping test folder:",
      len(csv_ids & test_ids))

print("Train-test ID overlap:",
      len(train_ids & test_ids))

# =====================================
# 6. Class distribution
# =====================================

class_rows = Counter()
class_images = defaultdict(set)
class_mapping = {}

for row in rows:
    name = row["class_name"]
    image_id = row["image_id"]
    class_id = row["class_id"]

    class_rows[name] += 1
    class_images[name].add(image_id)
    class_mapping[name] = class_id

print("\n========== CLASS DISTRIBUTION ==========")

for name, count in class_rows.most_common():
    print(
        f"{name:25s} "
        f"rows={count:6d} "
        f"images={len(class_images[name]):6d} "
        f"class_id={class_mapping[name]}"
    )

# =====================================
# 7. Radiologist analysis
# =====================================

image_readers = defaultdict(set)

for row in rows:
    image_readers[row["image_id"]].add(row["rad_id"])

reader_distribution = Counter(
    len(readers) for readers in image_readers.values()
)

print("\n========== RADIOLOGIST ANALYSIS ==========")

print("Unique radiologists:",
      len(set(row["rad_id"] for row in rows)))

print("Readers per image:")

for readers, count in sorted(reader_distribution.items()):
    print(f"{readers} readers: {count} images")

# =====================================
# 8. Bounding box validation
# =====================================

no_box = 0
complete_box = 0
partial_box = 0
invalid_box = 0

no_finding_with_box = 0
abnormal_without_box = 0

coordinate_columns = [
    "x_min", "y_min", "x_max", "y_max"
]

for row in rows:

    coordinates = [
        row[col].strip()
        for col in coordinate_columns
    ]

    empty_count = sum(not x for x in coordinates)

    is_no_finding = row["class_name"] == "No finding"

    if empty_count == 4:
        no_box += 1

        if not is_no_finding:
            abnormal_without_box += 1

    elif empty_count > 0:
        partial_box += 1

    else:
        complete_box += 1

        if is_no_finding:
            no_finding_with_box += 1

        try:
            x_min, y_min, x_max, y_max = map(
                float, coordinates
            )

            if x_min >= x_max or y_min >= y_max:
                invalid_box += 1

        except ValueError:
            invalid_box += 1

print("\n========== BOUNDING BOX QUALITY ==========")

print("Rows without boxes:", no_box)
print("Rows with complete boxes:", complete_box)
print("Rows with partial boxes:", partial_box)
print("Invalid coordinate boxes:", invalid_box)

print("No finding with boxes:", no_finding_with_box)
print("Abnormal findings without boxes:", abnormal_without_box)

print("\n========== AUDIT COMPLETED ==========")
