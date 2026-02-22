# PKLot Final Notebook + Streamlit App

This project contains:

- `PKLOT_Final.ipynb`: end-to-end parking occupancy classification pipeline
- `app.py`: Streamlit inference app with custom dark-mode UI
- `best_model.keras`: trained binary classifier (`empty` vs `occupied`)

---

## 1) Notebook (`PKLOT_Final.ipynb`)

The notebook builds a complete workflow to classify parking spots as **Empty** or **Occupied**.

### Objective

- Load PKLot data and COCO annotations
- Extract parking-spot crops from bounding boxes
- Train a CNN on cropped spots
- Evaluate performance
- Export deployable model files

### Workflow Summary

1. **Data Loading**
   - Mount Google Drive in Colab
   - Unzip dataset from `/content/drive/MyDrive/pklot.zip`
   - Extract into `/content/PKLot/`

2. **Annotation Loading**
   - Load COCO annotations for `train`, `valid`, `test`
   - Build image/category ID mappings

3. **EDA**
   - Class distribution
   - Spots per image stats
   - Bounding box stats and aspect ratio
   - Box visualization on original images

4. **Crop Extraction**
   - Crop each parking space from COCO bbox
   - Save into class folders: `empty`, `occupied`
   - Balanced subsets:
     - Train: 20,000 + 20,000
     - Valid: 5,000 + 5,000
     - Test: 5,000 + 5,000

5. **Training Pipeline**
   - `ImageDataGenerator` with augmentation for train split
   - Input size: `96x96`
   - Batch size: `64`

6. **CNN Architecture**
   - Conv(32) + Pool
   - Conv(64) + Pool
   - Conv(128) + Pool
   - Flatten → Dense(256) → Dropout(0.5) → Dense(1, sigmoid)

7. **Training Strategy**
   - Up to 15 epochs
   - Early stopping (`patience=4`)
   - Best checkpoint on `val_accuracy`

8. **Evaluation**
   - Test accuracy around **97.21%**
   - Confusion matrix + classification report

9. **Export**
   - Keras model: `/content/pklot_cropped/best_model.keras`
   - TFLite model: `/content/pklot_occupancy_model.tflite`

---

## 2) Streamlit App (`app.py`) 

The Streamlit app loads `best_model.keras` and predicts whether an uploaded parking-spot image is **Empty** or **Occupied**.

### App Features

- Cached model loading via `@st.cache_resource`
- Image preprocessing to `96x96`, RGB, normalized to `[0,1]`
- Prediction threshold: `0.5`
- Confidence score + raw probability display
- Validation for invalid image uploads
- **Custom dark UI styling**:
  - App background in dark theme
  - Label **“Choose a parking spot image...”** in white
  - **“Browse files”** button in dark color scheme
  - Uploaded filename chip in dark theme styling

### Run Locally

1. Install dependencies:
```bash 
pip install -r requirements.txt
```

2. Start the app:
```bash
streamlit run app.py
```

3. Open browser URL shown by Streamlit and upload an image.

Keep best_model.keras in the same directory as app.py.

## 3) Requirements
From requirements.txt:
- streamlit
- tensorflow
- pillow
- numpy

## Notes
- Notebook paths are written for Google Colab (/content/...).
- If running notebook locally, update dataset/model paths.
- Streamlit app is intended for inference using the already trained .keras model.
