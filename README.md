### 🧠 **Project Summary**

This notebook builds a machine learning model that:

* Extracts features from `.wav` audio files (like MFCCs, chroma, mel spectrogram, etc.)
* Trains a **MLP (Multi-Layer Perceptron)** classifier
* Uses the **RAVDESS dataset**
* Visualizes key feature distributions

---

### 🔍 Key Components

#### ✅ **1. Imports & Setup**

You're using:

* `librosa` & `soundfile` for audio processing
* `sklearn` for model training and evaluation
* `matplotlib` & `seaborn` for visualization

---

#### ✅ **2. Emotion Mapping**

Defined as:

```python
emotions = {
    0: "calm",
    1: "happy",
    2: "neutral",
    3: "sad",
    4: "angry",
    5: "fearful",
    6: "disgust",
    7: "surprised"
}
```

---

#### ✅ **3. Feature Extraction Function**

You're extracting:

* **MFCC**
* **Chroma**
* **Mel-spectrogram**
* **Spectral contrast**
* **Tonnetz**

All of which are concatenated into one feature vector per audio file.

---

#### ✅ **4. Dataset Loading**

* Loops through a dataset directory
* Extracts `.wav` files from subfolders
* Uses the filename format to determine the **emotion label**
* Loads all into `features` and `labels` arrays

```python
features, labels = load_dataset(dataset_path, emotion_mapping)
```

---

#### ✅ **5. Feature Visualization**

You're plotting:

* MFCCs
* Chromograms
* Mel spectrograms
* Spectral contrast
* Tonnetz

Each shown as a **boxplot distribution**, which is great for spotting outliers and spread.

---

