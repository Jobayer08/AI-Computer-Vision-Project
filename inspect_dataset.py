
import csv
from collections import Counter

csv_path = "train.csv"

with open(csv_path, "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    rows = list(reader)

print("Total rows:", len(rows))

# 1. Column information
print("\nColumns:")
print(reader.fieldnames)

# 2. First five records
print("\nFirst 5 records:")
for row in rows[:5]:
    print(row)

# 3. Missing values
print("\nMissing values:")
for column in reader.fieldnames:
    missing = sum(
        1 for row in rows
        if not row[column].strip()
    )
    print(f"{column}: {missing}")

# 4. Unique image count
if "image_id" in reader.fieldnames:
    image_ids = set(row["image_id"] for row in rows)
    print("\nUnique images:", len(image_ids))

# 5. Class distribution
if "class_name" in reader.fieldnames:
    classes = Counter(
        row["class_name"] for row in rows
    )

    print("\nClass distribution:")
    for name, count in classes.most_common():
        print(f"{name}: {count}")
