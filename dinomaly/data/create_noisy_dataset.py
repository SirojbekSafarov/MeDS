"""
Create a noisy MVTec-style dataset from a CSV file.

The CSV specifies which images to include in the training set.
Defective samples (from test/<defect_type>/) are copied into train/good/
to simulate label noise. The test set is copied as-is from the original dataset.

Usage:
    python create_noisy_dataset.py \
        --original_dataset /path/to/mvtec_anomaly_detection \
        --csv_file data/MVTech/train_mvtec_nr10_seed_0.csv \
        --output_dir /path/to/output/MVTech_nr10_seed_0
"""

import argparse
import os
import shutil
import pandas as pd


def create_noisy_dataset(original_dataset, csv_file, output_dir):
    df = pd.read_csv(csv_file)

    # Get all class names from the CSV
    class_names = df["clsname"].unique()
    print(f"Classes found in CSV: {list(class_names)}")

    for cls_name in class_names:
        cls_df = df[df["clsname"] == cls_name]
        original_cls_dir = os.path.join(original_dataset, cls_name)
        output_cls_dir = os.path.join(output_dir, cls_name)

        # --- Copy test set as-is ---
        src_test = os.path.join(original_cls_dir, "test")
        dst_test = os.path.join(output_cls_dir, "test")
        if os.path.exists(src_test):
            shutil.copytree(src_test, dst_test, dirs_exist_ok=True)
            print(f"  [{cls_name}] Copied test set")

        # --- Copy ground_truth as-is ---
        src_gt = os.path.join(original_cls_dir, "ground_truth")
        dst_gt = os.path.join(output_cls_dir, "ground_truth")
        if os.path.exists(src_gt):
            shutil.copytree(src_gt, dst_gt, dirs_exist_ok=True)
            print(f"  [{cls_name}] Copied ground_truth")

        # --- Build noisy train/good/ from CSV ---
        dst_train_good = os.path.join(output_cls_dir, "train", "good")
        os.makedirs(dst_train_good, exist_ok=True)

        n_good = 0
        n_defective = 0

        for _, row in cls_df.iterrows():
            filename = row["filename"]  # e.g. toothbrush/test/defective/017.png
            src_path = os.path.join(original_dataset, filename)
            img_name = os.path.basename(filename)

            # Check if this is a defective sample (not from train/good)
            is_defective = "/train/good/" not in filename

            if is_defective:
                # Rename to avoid collisions: defect_type_originalname.png
                # e.g. test/defective/017.png -> defective_017.png
                parts = filename.split("/")
                # parts: [cls, "test", defect_type, img_file]
                defect_type = parts[-2]  # e.g. "defective", "cut", "color"
                img_name = f"{defect_type}_{os.path.basename(filename)}"
                n_defective += 1
            else:
                n_good += 1

            dst_path = os.path.join(dst_train_good, img_name)

            if not os.path.exists(src_path):
                print(f"  WARNING: Source not found: {src_path}")
                continue

            shutil.copy2(src_path, dst_path)

        print(
            f"  [{cls_name}] Train set: {n_good} good + {n_defective} defective "
            f"= {n_good + n_defective} total"
        )

    print(f"\nNoisy dataset created at: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create noisy MVTec dataset from CSV")
    parser.add_argument(
        "--original_dataset",
        type=str,
        required=True,
        help="Path to the original MVTec-AD dataset",
    )
    parser.add_argument(
        "--csv_file",
        type=str,
        required=True,
        help="Path to the noise CSV file (e.g. train_mvtec_nr10_seed_0.csv)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path for the output noisy dataset",
    )
    args = parser.parse_args()

    create_noisy_dataset(args.original_dataset, args.csv_file, args.output_dir)
