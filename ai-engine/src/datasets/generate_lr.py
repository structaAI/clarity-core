import os
import cv2
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import time

# --- CONFIGURATION ---
HR_DIR = r"D:\Structa\claritycore\ai-engine\data\train2017"
LR_DIR = r"D:\Structa\claritycore\ai-engine\data\train2017_lr"
TARGET_HR = 1024
TARGET_LR = 128  # 8x Upscaling Challenge

def apply_heavy_damage(img: np.ndarray) -> np.ndarray:
    """Creates an extreme 8x degradation for high-grade restoration training."""
    # 1. Random Blur (Focus/Motion simulation)
    if np.random.rand() > 0.5:
        k = np.random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)
    
    # 2. Downsample to 128x128
    img_lr = cv2.resize(img, (TARGET_LR, TARGET_LR), interpolation=cv2.INTER_AREA)
    
    # 3. Add Sensor Noise
    noise_sigma = np.random.uniform(2, 6)
    noise = np.random.normal(0, noise_sigma, img_lr.shape).astype(np.uint8)
    img_lr = cv2.add(img_lr, noise)
    
    # 4. Heavy JPEG Artifacts
    quality = np.random.randint(30, 60)
    _, enc = cv2.imencode('.jpg', img_lr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    
    decoded_img = cv2.imdecode(enc, 1)
    
    if decoded_img is None:
        # Fallback: if decode fails, return the resized img without JPEG artifacts
        return np.ascontiguousarray(img_lr)
        
    return np.ascontiguousarray(decoded_img)

def process_image(img_name: str):
    try:
        hr_path = os.path.join(HR_DIR, img_name)
        lr_path = os.path.join(LR_DIR, img_name)
        
        # Load HR
        img = cv2.imread(hr_path)
        if img is None:
            return

        # Step A: Standardize HR to 1024px (LANCZOS4 for best Ground Truth quality)
        hr_final = cv2.resize(img, (TARGET_HR, TARGET_HR), interpolation=cv2.INTER_LANCZOS4)
        
        # Step B: Create Heavily Damaged LR
        lr_final = apply_heavy_damage(hr_final)

        # Step C: Save both (Overwriting HR with the standardized 1024px version)
        # Using np.ascontiguousarray to resolve the 'MatLike' type-hint error
        cv2.imwrite(hr_path, np.ascontiguousarray(hr_final)) # type: ignore
        cv2.imwrite(lr_path, np.ascontiguousarray(lr_final)) # type: ignore
        
    except Exception:
        pass # Skip corrupted files to maintain momentum

if __name__ == "__main__":
    if not os.path.exists(LR_DIR):
        os.makedirs(LR_DIR)

    print("🚀 Gathering images...")
    images = [f for f in os.listdir(HR_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    total = len(images)
    
    print(f"🔥 Starting Isolated High-Speed Processing of {total} images.")
    print(f"💻 Utilizing all {cpu_count()} threads of the Legion Pro 7.")

    start_time = time.time()

    # Using full cpu_count() with unordered mapping for maximum throughput
    with Pool(processes=cpu_count()) as pool:
        list(tqdm(pool.imap_unordered(process_image, images, chunksize=128), total=total))

    end_time = time.time()
    print(f"\n✅ DONE! Total time: {(end_time - start_time)/60:.2f} minutes.")
    print(f"📂 Pairs ready for 12-epoch training in {LR_DIR}")