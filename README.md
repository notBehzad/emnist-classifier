---
title: EMNIST Character Classifier
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---
# EMNIST Balanced Character Classifier

Predicts a character or digit

## Demo
![Project Demo](assets/demo.gif)


## Model Architecture
- Brief description (layers, params, etc.)
- Achieved X% accuracy on test set

## Results
![Confusion Matrix](assets/confusion_matrix.png)
![Training Curves](assets/training_curves.png)

## Project Structure
```text
emnist-classifier/
│
├── app.py                    # FastAPI backend
├── requirements.txt
├── .gitignore
├── README.md
│
├── model/
│   ├── model.py              # Architecture definition (separate from training)
│   ├── train.py              # Training script
│   └── emnist_model.pth
│
├── templates/
│   └── index.html            # Frontend canvas
│
└── assets/                   # All visuals for README
    ├── demo.gif              # Screen recording of drawing + predicting
    ├── confusion_matrix.png
    ├── samples.png
    └── training_curves.png
```

## Setup & Run
pip install -r requirements.txt
uvicorn app:app --reload

## Tech Stack
PyTorch · FastAPI · Vanilla JS
