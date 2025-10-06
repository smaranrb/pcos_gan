import os
import shutil
import random

def split_dataset(
    source_dir="./data",
    dest_dir="./data_split",
    split_ratio=0.8,  # 80% train, 20% test
    seed=42
):
    random.seed(seed)

    # Create output directories
    train_dir = os.path.join(dest_dir, "train")
    test_dir = os.path.join(dest_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Iterate over each class folder
    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_path):
            continue  # skip non-directories

        images = os.listdir(class_path)
        random.shuffle(images)

        split_index = int(len(images) * split_ratio)
        train_images = images[:split_index]
        test_images = images[split_index:]

        # Create class subfolders
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

        # Move files
        for img in train_images:
            shutil.move(os.path.join(class_path, img),
                        os.path.join(train_dir, class_name, img))

        for img in test_images:
            shutil.move(os.path.join(class_path, img),
                        os.path.join(test_dir, class_name, img))

        print(f"✅ {class_name}: {len(train_images)} train, {len(test_images)} test")

    print("\n✅ Split complete!")
    print(f"Train dir: {train_dir}")
    print(f"Test dir:  {test_dir}")


if __name__ == "__main__":
    # Modify this split ratio as needed (e.g., 0.7 for 70/30 split)
    split_dataset(split_ratio=0.75)
