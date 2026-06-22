---
title: EMNIST Character Classifier
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---
# EMNIST Balanced Character Classifier

A fully-connected neural network that recognizes handwritten characters (digits + letters) in real time, draw in the browser, get a prediction instantly.

---

## Demo

> `![Demo](assets/demo.gif)`

---

## Model Architecture

| Layer | Details |
|---|---|
| Input | 28×28 grayscale → Flatten → 784 |
| FC1 | 784 → 256, ReLU |
| FC2 | 256 → 128, ReLU |
| FC3 | 128 → 47 (output) |

- **Parameters:** ~240K trainable
- **Loss:** CrossEntropyLoss
- **Optimizer:** Adam (lr = 0.001)
- **Augmentation:** RandomAffine (rotation ±15°, translate, scale), improves real-world drawing robustness
- **Test Accuracy: 83.93%** on EMNIST Balanced (47 classes, 18,800 test samples)

> Train accuracy (~80%) sits below test accuracy (~84%), this is expected. RandomAffine makes training harder than the clean test set, which is a sign the augmentation is working correctly.

---

## Results

### Training Curves
![Training Curves](assets/training_curves.png)

### Confusion Matrix
![Confusion Matrix](assets/confusion_matrix.png)

Notable observations:
- Digits (0–9) achieve the highest accuracy
- Common confusions: `I↔1`, `O↔0`, `l↔L`, `g↔q` , visually similar pairs
- Lowercase letters are the hardest class overall

---

## Project Structure

```
emnist-classifier/
│
├── app.py                  # FastAPI backend + prediction endpoint
├── requirements.txt
├── .gitignore
├── README.md
│
├── templates/
│   └── index.html          # Canvas UI (draw + predict)
│
├── model/
│   ├── __init__.py
│   ├── model.py            # MLP architecture
│   ├── train.py            # Training script
│   ├── train.ipynb         # Training notebook with plots
│   └── emnist_model.pth    # Saved weights (~900 KB)
│
└── assets/
    ├── demo.gif
    ├── training_curves.png
    ├── confusion_matrix.png
    └── samples.png
```

---

## Test


Link: https://huggingface.co/spaces/notBehzad/character-classifier

> No retraining needed — pretrained weights are included. To retrain from scratch: `cd model && python train.py`

---

## Tech Stack

**PyTorch** · **FastAPI** · **Vanilla JS** · **EMNIST Balanced**
