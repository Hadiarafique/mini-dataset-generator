# 📚 Phase 1 Learning Guide

Welcome! This guide will help you understand everything in Phase 1 of the Mini Dataset Generator.

---

## 🎯 What You Built

You created a **Computer Vision pipeline** that:
1. ✅ Removes duplicate images
2. 🎨 Augments images (creates variations)
3. 📦 Builds YOLO-ready datasets

---

## 📂 Project Structure Explained

```
mini_dataset_generator/
│
├── src/                          # All your code modules
│   ├── duplicate_remover.py      # Finds and removes duplicate images
│   ├── augmentation.py           # Creates image variations
│   └── dataset_builder.py        # Creates YOLO folder structure
│
├── test_images/                  # Put your input images here
├── output/                       # Generated datasets go here
│
├── main.py                       # Run this to execute the pipeline
├── requirements.txt              # Python packages needed
└── LEARNING_GUIDE.md            # This file
```

---

## 🧠 Key Concepts You Need to Know

### 1. **Perceptual Hashing** (Duplicate Detection)

**What is it?**
- A way to create a "fingerprint" of an image
- Similar images have similar fingerprints
- Different from regular hash (MD5, SHA) which changes completely with 1 pixel change

**How it works:**
```python
# Regular hash (MD5/SHA)
image1 = "cat.jpg"          → hash: a3f7b8c9d2e1
image1_rotated = "cat.jpg"  → hash: 9d2e1f7b8c3a  # Completely different!

# Perceptual hash
image1 = "cat.jpg"          → phash: 1101010101
image1_rotated = "cat.jpg"  → phash: 1101010100  # Very similar!
```

**Why use it?**
- Detects duplicates even if images are slightly different
- Saves storage and training time
- Prevents model overfitting

**Code location:** `src/duplicate_remover.py`

---

### 2. **Image Augmentation**

**What is it?**
- Creating multiple variations of the same image
- Helps train better ML models with limited data

**Example transformations:**
```
Original image (1)
    ↓
Augmentation
    ↓
├── Rotated 45° (1 new image)
├── Flipped horizontally (1 new image)
├── Brightness increased (1 new image)
├── Color shifted (1 new image)
└── Perspective changed (1 new image)
    ↓
Total: 6 images from 1 original!
```

**Why use it?**
- Increases dataset size: 10 images → 100+ images
- Model learns to handle variations
- Reduces need for manual data collection

**Code location:** `src/augmentation.py`

**Libraries used:**
- `opencv-python` (cv2): Basic image operations
- `albumentations`: Advanced augmentation library (used by pros)

---

### 3. **YOLO Dataset Structure**

**What is YOLO?**
- **Y**ou **O**nly **L**ook **O**nce
- A fast object detection algorithm
- Used in real-time applications (self-driving cars, manufacturing)

**Required folder structure:**
```
my_dataset/
├── train/
│   ├── images/           # Training images (80%)
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   └── labels/           # Label files (bounding boxes)
│       ├── img1.txt      # Matches img1.jpg
│       ├── img2.txt      # Matches img2.jpg
│       └── ...
│
├── valid/
│   ├── images/           # Validation images (20%)
│   └── labels/           # Validation labels
│
└── data.yaml            # Configuration file
```

**What's in a label file? (img1.txt)**
```
# Format: <class_id> <x_center> <y_center> <width> <height>
0 0.5 0.5 0.3 0.4
1 0.7 0.3 0.2 0.2
```
- All values are normalized (0 to 1)
- Each line = one object in the image

**What's in data.yaml?**
```yaml
path: /path/to/my_dataset
train: train/images
val: valid/images

names:
  0: apple
  1: orange

nc: 2  # number of classes
```

**Code location:** `src/dataset_builder.py`

---

## 🔧 How Each Module Works

### Module 1: `duplicate_remover.py`

**Main class:** `DuplicateRemover`

**Key methods:**
1. `calculate_hash(image_path)` - Creates perceptual hash
2. `is_duplicate(hash1, hash2)` - Compares two hashes
3. `find_duplicates(folder)` - Scans folder for duplicates
4. `remove_duplicates(folder)` - Removes duplicates

**How to use:**
```python
from src.duplicate_remover import DuplicateRemover

remover = DuplicateRemover(threshold=5)
unique, duplicates = remover.find_duplicates("my_images/")

print(f"Unique: {len(unique)}")
print(f"Duplicates: {len(duplicates)}")
```

**Threshold parameter:**
- `0` = Only exact matches
- `5` = Default, good balance
- `10` = Very lenient, finds similar images

---

### Module 2: `augmentation.py`

**Main class:** `ImageAugmenter`

**Two approaches:**

**A) Simple transforms** (easier to understand)
```python
augmenter = ImageAugmenter()

# Apply specific transforms
augmented = augmenter.augment_simple(
    "image.jpg", 
    transforms=['rotate_90', 'flip_horizontal']
)
```

Available simple transforms:
- `rotate_90`, `rotate_180`, `rotate_270`
- `flip_horizontal`, `flip_vertical`
- `brightness_up`, `brightness_down`

**B) Advanced transforms** (using albumentations)
```python
# Applies random combinations
augmented = augmenter.augment_image("image.jpg", num_augmentations=5)
```

Includes:
- Random rotation (up to 45°)
- Scale and shift
- Brightness/contrast changes
- Color shifts
- Blur and noise
- Perspective changes

**Which to use?**
- **Simple**: When you want specific, predictable results
- **Advanced**: When you want variety and randomness

---

### Module 3: `dataset_builder.py`

**Main class:** `YOLODatasetBuilder`

**Key methods:**
1. `create_folder_structure()` - Creates train/valid folders
2. `split_train_valid()` - Splits images 80/20
3. `copy_images_to_dataset()` - Copies images to folders
4. `create_empty_labels()` - Creates .txt label files
5. `generate_yaml()` - Creates data.yaml config
6. `build_dataset()` - Does everything in one go

**How to use:**
```python
from src.dataset_builder import YOLODatasetBuilder

builder = YOLODatasetBuilder(dataset_name="apples_dataset")

result = builder.build_dataset(
    image_paths=["img1.jpg", "img2.jpg", ...],
    output_path="output/",
    class_names=['apple', 'orange'],
    train_ratio=0.8
)
```

**Train/Valid split:**
- 80% training = Model learns from these
- 20% validation = Model is tested on these
- **Important**: Validation images should NEVER be in training set

---

## 🚀 How to Run Phase 1

### Step 1: Install dependencies
```powershell
pip install -r requirements.txt
```

### Step 2: Add test images
- Put 10-20 images in `test_images/` folder
- Any format: .jpg, .png, .bmp

### Step 3: Run the pipeline
```powershell
python main.py
```

### Step 4: Follow the prompts
```
How many augmentations per image? (recommended: 5-10): 5
Dataset name (default: my_dataset): apples
Object classes (comma-separated, e.g. apple,orange): apple
```

### Step 5: Check output
- Generated dataset: `output/your_dataset_name/`
- Augmented images: `output/augmented_images/`

---

## 🧪 Test Each Module Individually

### Test duplicate remover:
```powershell
python src/duplicate_remover.py
```

### Test augmentation:
```powershell
python src/augmentation.py
```

### Test dataset builder:
```powershell
python src/dataset_builder.py
```

Each module has test code at the bottom (`if __name__ == "__main__":`)

---

## 💡 Key Python Concepts Used

### 1. **Classes and Objects**
```python
class DuplicateRemover:
    def __init__(self, threshold):
        self.threshold = threshold  # Instance variable
    
    def calculate_hash(self, image_path):  # Instance method
        # ...
```

### 2. **List Comprehensions**
```python
# Long way
image_files = []
for f in folder.iterdir():
    if f.suffix == '.jpg':
        image_files.append(f)

# Short way (comprehension)
image_files = [f for f in folder.iterdir() if f.suffix == '.jpg']
```

### 3. **Pathlib** (Modern file handling)
```python
from pathlib import Path

# Old way
path = "folder/subfolder/file.txt"
filename = path.split('/')[-1]

# New way
path = Path("folder/subfolder/file.txt")
filename = path.name  # file.txt
stem = path.stem      # file (without extension)
```

### 4. **Type Hints**
```python
def process_images(images: List[str]) -> int:
    # images must be a list of strings
    # function returns an integer
    return len(images)
```

### 5. **F-strings** (String formatting)
```python
name = "Dataset"
count = 42

# Old way
print("Found " + str(count) + " images in " + name)

# New way
print(f"Found {count} images in {name}")
```

---

## 🔍 Understanding the Data Flow

```
Input Images (test_images/)
    ↓
[Step 1: Duplicate Remover]
    ↓
Unique Images Only
    ↓
[Step 2: Augmenter]
    ↓
Original + Augmented Images
    ↓
[Step 3: Dataset Builder]
    ↓
YOLO Dataset Structure
├── Train (80%)
└── Valid (20%)
    ↓
Ready for Training!
```

---

## 📊 Example Run

**Input:**
- 15 images in `test_images/`
- 3 are duplicates
- Augmentation: 5x per image

**Processing:**
```
Step 1: Duplicate Detection
  → 15 images → 12 unique (3 duplicates removed)

Step 2: Augmentation
  → 12 unique × 5 augmentations = 60 new images
  → Total: 12 + 60 = 72 images

Step 3: Dataset Build
  → Train: 72 × 80% = 58 images
  → Valid: 72 × 20% = 14 images
```

**Output:**
```
my_dataset/
├── train/
│   ├── images/ (58 images)
│   └── labels/ (58 .txt files)
├── valid/
│   ├── images/ (14 images)
│   └── labels/ (14 .txt files)
└── data.yaml
```

---

## 🎓 What You Learned

### Computer Vision Skills:
- ✅ Image hashing algorithms
- ✅ Image transformations (rotation, flip, color)
- ✅ Data augmentation techniques
- ✅ Dataset organization for ML

### Python Skills:
- ✅ Object-oriented programming (classes)
- ✅ File handling with Pathlib
- ✅ Working with OpenCV
- ✅ Type hints and documentation
- ✅ Error handling

### ML/AI Skills:
- ✅ YOLO dataset format
- ✅ Train/validation splitting
- ✅ Data preprocessing pipeline
- ✅ YAML configuration files

---

## 🚀 Next Steps (Phase 2)

Phase 2 will add a **Streamlit UI** so users can:
- Upload images through web browser
- Configure settings with sliders/buttons
- Preview augmented images
- Download dataset as ZIP

**You'll learn:**
- Web UI development with Streamlit
- File uploads and downloads
- Creating ZIP files
- Progress bars and user feedback

---

## 🐛 Troubleshooting

### Problem: "Module not found"
**Solution:**
```powershell
pip install -r requirements.txt
```

### Problem: "No images found"
**Solution:** Add images to `test_images/` folder

### Problem: "Permission denied"
**Solution:** Run PowerShell as Administrator or change output folder

### Problem: Augmented images look weird
**Solution:** Adjust augmentation parameters in `augmentation.py`
- Lower rotation angles
- Reduce brightness changes
- Disable perspective transforms

---

## 📚 Further Reading

### Perceptual Hashing:
- https://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html

### Image Augmentation:
- Albumentations docs: https://albumentations.ai/docs/

### YOLO:
- Ultralytics YOLOv11: https://docs.ultralytics.com/
- YOLO format explained: https://docs.ultralytics.com/datasets/

### Python:
- Pathlib tutorial: https://realpython.com/python-pathlib/
- OpenCV tutorials: https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html

---

## ❓ Questions to Test Your Understanding

1. **What's the difference between perceptual hash and MD5 hash?**
   <details>
   <summary>Answer</summary>
   Perceptual hash creates similar fingerprints for similar images. MD5 creates completely different hashes even if one pixel changes.
   </details>

2. **Why do we split into train/valid sets?**
   <details>
   <summary>Answer</summary>
   To test the model on unseen data. If we train and test on the same data, we can't know if the model actually learned or just memorized.
   </details>

3. **What does the number in a YOLO label file represent?**
   <details>
   <summary>Answer</summary>
   Format: class_id x_center y_center width height (all normalized 0-1)
   </details>

4. **Why augment images instead of just collecting more?**
   <details>
   <summary>Answer</summary>
   It's faster and cheaper. Augmentation creates variety from existing images, saving time and money on data collection.
   </details>

---

## 🎯 Challenge Exercises

### Beginner:
1. Add a new simple transform (e.g., grayscale conversion)
2. Change the train/valid split to 70/30
3. Add support for .webp images

### Intermediate:
1. Add a duplicate threshold adjustment in main.py
2. Create a function to preview augmented images
3. Add progress bars using `tqdm`

### Advanced:
1. Implement a similarity score (how different two images are)
2. Add batch size parameter to control memory usage
3. Create a "smart" augmentation that adapts to image content

---

## 💬 Need Help?

- Read the comments in each file
- Run individual modules to test
- Check the error messages carefully
- Google specific errors with context

**Remember:** Every expert was once a beginner who didn't give up! 🚀

---

**Good luck with your learning journey!** 🎉
