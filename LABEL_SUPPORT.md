# 🏷️ Label/Annotation Support Guide

## Overview
The Mini Dataset Generator now supports uploading existing labels with your images and exporting them in multiple formats!

## Features Added

### 1. **Upload Existing Labels**
- ✅ Upload YOLO format labels (.txt files) alongside your images
- ✅ Automatic matching of labels to images by filename
- ✅ Validation and matching statistics
- ✅ Labels are preserved during augmentation and splitting

### 2. **Multiple Export Formats**
Support for three industry-standard annotation formats:

#### **YOLO Format** (Default)
```
class_id x_center y_center width height
```
- All values normalized (0.0 to 1.0)
- Most common format for YOLO training
- One line per object

#### **COCO JSON Format**
```json
{
  "images": [...],
  "annotations": [...],
  "categories": [...]
}
```
- Popular for research and competitions
- Rich metadata support
- Single JSON file per split

#### **Pascal VOC XML Format**
```xml
<annotation>
  <object>
    <name>class_name</name>
    <bndbox>
      <xmin>100</xmin>
      <ymin>150</ymin>
      <xmax>200</xmax>
      <ymax>250</ymax>
    </bndbox>
  </object>
</annotation>
```
- Used by many annotation tools
- One XML file per image
- Human-readable format

## How to Use

### Step 1: Enable Label Support
1. Open the sidebar
2. Under "🏷️ Label Support" section
3. Check "I have existing labels"
4. Select your export format (YOLO/COCO/Pascal VOC)

### Step 2: Upload Images and Labels
1. Upload your images as usual
2. A new section appears: "🏷️ Step 1b: Upload Labels"
3. Upload your .txt label files
4. **Important:** Label filenames must match image filenames
   - Example: `dog.jpg` → `dog.txt`
   - Example: `cat_001.png` → `cat_001.txt`

### Step 3: Process Dataset
1. Configure augmentation and split settings
2. Click "Generate Dataset"
3. Labels are automatically:
   - Matched with images
   - Preserved during augmentation (not transformed)
   - Split into train/test/valid folders
   - Converted to your chosen format

### Step 4: Download
- Your ZIP file includes all labels in the selected format
- YOLO: `.txt` files in `labels/` folders
- COCO: `annotations_train.json`, `annotations_test.json`, `annotations_valid.json`
- Pascal VOC: `.xml` files in `annotations_xml/` folders

## Label File Format Requirements

### YOLO Format (.txt)
Each line represents one object:
```
0 0.5 0.5 0.3 0.4
1 0.2 0.3 0.15 0.2
```
- Column 1: Class ID (integer, starting from 0)
- Column 2: X center (normalized 0-1)
- Column 3: Y center (normalized 0-1)
- Column 4: Width (normalized 0-1)
- Column 5: Height (normalized 0-1)

### File Naming Convention
```
images/
  ├── image_001.jpg
  ├── image_002.jpg
  └── image_003.png

labels/
  ├── image_001.txt
  ├── image_002.txt
  └── image_003.txt  ← Must match image names!
```

## Example Workflow

### Scenario: You have annotated images from LabelImg

1. **Export from LabelImg:**
   - Choose YOLO format
   - Get `images/` and `labels/` folders

2. **Upload to Mini Dataset Generator:**
   - Enable "I have existing labels"
   - Upload all images
   - Upload all .txt label files
   - See matching statistics (e.g., "🔗 Matched 50/50 images with labels")

3. **Configure:**
   - Set augmentation: 5 per image
   - Set split: 70/15/15
   - Choose export: COCO JSON (for research)

4. **Download:**
   - Get 250 augmented images with labels
   - COCO JSON files ready for training
   - Properly split into train/test/valid

## Tips & Best Practices

### ✅ DO:
- Match label filenames exactly with image filenames
- Use consistent naming (e.g., `img_001.jpg` → `img_001.txt`)
- Verify label format before uploading
- Check matching statistics after upload

### ❌ DON'T:
- Use different naming schemes for images and labels
- Mix multiple annotation formats
- Forget to enable "I have existing labels" checkbox
- Upload labels for images you didn't upload

## Troubleshooting

**"No labels matched with image filenames"**
- Check that label files have the same name as images (except extension)
- Example: `photo.jpg` needs `photo.txt`, not `photo_label.txt`

**"Label format invalid"**
- Ensure YOLO format: 5 space-separated numbers per line
- All values should be between 0 and 1
- Class IDs should be integers

**"Labels not in ZIP file"**
- Make sure you enabled "I have existing labels"
- Verify labels were uploaded successfully
- Check export format was selected

## Label Validation

The system automatically validates:
- ✅ File format (5 values per line)
- ✅ Normalized coordinates (0-1 range)
- ✅ Integer class IDs
- ✅ Filename matching
- ✅ Empty files detection

## Advanced: Label Augmentation

**Note:** Currently, labels are **not transformed** during augmentation to prevent coordinate mismatches. 

Future versions may include:
- Bounding box transformation during augmentation
- Label verification with augmented images
- Visual label preview

## Support

Questions or issues with labels?
1. Check this guide first
2. Verify your label format matches YOLO specification
3. Test with a small subset (2-3 images) first
4. Report issues on GitHub

---

**Added in Version:** December 2025  
**Compatible Formats:** YOLO, COCO JSON, Pascal VOC XML
