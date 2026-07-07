import os
import cv2
import numpy as np

def generate_mock_data(base_dir="data/raw", num_train_per_class=15, num_test_per_class=5):
    """
    Generates a synthetic brain MRI dataset for quick end-to-end pipeline testing.
    Replicates the directory structure expected by get_dataloaders in dataset.py.
    """
    classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
    splits = {
        'Training': num_train_per_class,
        'Testing': num_test_per_class
    }
    
    print(f"Generating synthetic brain MRI dataset at {base_dir}...")
    
    for split, count in splits.items():
        split_dir = os.path.join(base_dir, split)
        for cls_name in classes:
            class_dir = os.path.join(split_dir, cls_name)
            os.makedirs(class_dir, exist_ok=True)
            
            for i in range(count):
                # 1. Create a blank dark gray canvas (256x256) simulating background noise
                img = np.random.normal(15, 3, (256, 256)).astype(np.float32)
                
                # 2. Draw a skull outline (dark/light contrast ellipse)
                cv2.ellipse(img, (128, 128), (80, 100), 0, 0, 360, 40, 4) # outer skull
                cv2.ellipse(img, (128, 128), (76, 96), 0, 0, 360, 20, -1)   # bone marrow
                cv2.ellipse(img, (128, 128), (72, 92), 0, 0, 360, 5, 2)    # inner table
                
                # 3. Draw brain tissue (gray matter contour with folds)
                brain_mask = np.zeros((256, 256), dtype=np.uint8)
                cv2.ellipse(brain_mask, (128, 128), (68, 88), 0, 0, 360, 255, -1)
                
                # Add brain-like intensity structures inside the mask
                brain_texture = np.random.normal(100, 15, (256, 256)).astype(np.float32)
                # Ventricles (dark butterfly shapes in the center)
                cv2.ellipse(brain_texture, (115, 128), (10, 25), 15, 0, 360, 30, -1)
                cv2.ellipse(brain_texture, (141, 128), (10, 25), -15, 0, 360, 30, -1)
                
                # Blend brain texture into the image
                idx = (brain_mask == 255)
                img[idx] = brain_texture[idx]
                
                # 4. Draw class-specific tumor structures
                if cls_name != 'notumor':
                    # Set up center and radius
                    if cls_name == 'pituitary':
                        # Pituitary tumors are at the bottom base of the brain
                        center = (128, 185)
                        radius = np.random.randint(10, 15)
                        color = np.random.randint(200, 240)
                        cv2.circle(img, center, radius, color, -1)
                    elif cls_name == 'glioma':
                        # Gliomas are irregular, infiltrative, and diffuse
                        center = (128 + np.random.randint(-30, 30), 128 + np.random.randint(-30, 30))
                        radius = np.random.randint(15, 25)
                        color = np.random.randint(180, 220)
                        # Irregular shape via drawing overlapping ellipses
                        cv2.circle(img, center, radius, color, -1)
                        cv2.ellipse(img, center, (radius + 8, radius - 5), 45, 0, 360, color - 20, -1)
                    elif cls_name == 'meningioma':
                        # Meningiomas occur near the edge of the brain/meninges
                        center = (128 - 50, 120)
                        radius = np.random.randint(12, 18)
                        color = np.random.randint(210, 250)
                        cv2.circle(img, center, radius, color, -1)
                        # Add a dural tail (comet shape)
                        cv2.ellipse(img, (center[0] + 5, center[1] + 5), (radius + 12, radius - 8), -30, 0, 360, color - 10, 2)
                
                # 5. Add MRI noise and blur
                img = cv2.GaussianBlur(img, (3, 3), 0)
                img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
                
                # Save as JPG
                filename = f"{cls_name}_{i:03d}.jpg"
                filepath = os.path.join(class_dir, filename)
                cv2.imwrite(filepath, img_uint8)
                
    print("Mock dataset generation completed successfully!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/raw")
    args = parser.parse_args()
    
    generate_mock_data(args.data_dir)
