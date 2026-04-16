import os

hr_dir = r"D:\Structa\claritycore\ai-engine\data\train2017"
lr_dir = r"D:\Structa\claritycore\ai-engine\data\train2017_lr"

hr_files = sorted(os.listdir(hr_dir))[:5]
lr_files = sorted(os.listdir(lr_dir))[:5]

print("First 5 HR filenames:")
for f in hr_files:
    print(f" {f}")

print("\nFirst 5 LR filenames:")
for f in lr_files:
    print(f" {f}")

print(f"\nTotal HR: {len(os.listdir(hr_dir))}")
print(f"Total LR: {len(os.listdir(lr_dir))}")