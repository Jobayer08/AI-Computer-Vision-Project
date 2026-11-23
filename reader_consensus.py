
import csv
from collections import defaultdict, Counter

image_readers = defaultdict(set)
image_labels = defaultdict(set)
reader_labels = defaultdict(set)

with open("train.csv", "r") as file:
    for row in csv.DictReader(file):

        image_id = row["image_id"]
        rad_id = row["rad_id"]
        class_name = row["class_name"]

        image_readers[image_id].add(rad_id)
        image_labels[image_id].add(class_name)

        reader_labels[(image_id, rad_id)].add(class_name)


# 1. No finding and abnormal findings together
mixed_images = 0

for image_id, labels in image_labels.items():

    has_no_finding = "No finding" in labels

    has_abnormality = any(
        label != "No finding" for label in labels
    )

    if has_no_finding and has_abnormality:
        mixed_images += 1


# 2. Radiologist voting
no_finding_votes = Counter()
mixed_reader_annotations = 0

for (image_id, rad_id), labels in reader_labels.items():

    if labels == {"No finding"}:
        no_finding_votes[image_id] += 1

    if (
        "No finding" in labels
        and len(labels) > 1
    ):
        mixed_reader_annotations += 1


vote_distribution = Counter(
    no_finding_votes.get(image_id, 0)
    for image_id in image_readers
)


print("========== READER CONSENSUS ==========")

print("Images with mixed No finding and abnormal labels:",
      mixed_images)

print("Reader groups with No finding + abnormality:",
      mixed_reader_annotations)

print("\nNo finding votes per image:")

for votes, count in sorted(vote_distribution.items()):
    print(f"{votes} / 3 radiologists: {count} images")


print("\nImages with all 3 No finding votes:",
      vote_distribution[3])

print("Images with 0 No finding votes:",
      vote_distribution[0])
