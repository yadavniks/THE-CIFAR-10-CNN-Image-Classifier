# 🎯 CIFAR-10 Image Classification (Deep Learning Project)

This project builds and trains a neural network to classify images from the **CIFAR-10 dataset** into 10 object categories such as airplanes, cars, animals, and more.  
The model is implemented using **Python, TensorFlow/Keras**, and evaluates performance with metrics like accuracy and loss.

---

**📂 Dataset**

CIFAR-10 contains:

| Feature | Detail |
|--------|--------|
| Image size | 32 × 32 |
| Color | RGB |
| Classes | 10 |
| Total images | 60,000 |
| Train/Test split | 50,000 / 10,000 |

Classes include:
> airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

Dataset is loaded directly from **Keras datasets**.

---
 **🧠 Model**

The model uses a Convolutional Neural Network (CNN) with layers such as:

- Convolutional layers (feature extraction)
- MaxPooling (dimension reduction)
- Dense Fully-Connected layers
- Softmax output (classification into 10 classes)

Input Preprocessing
- Normalization:  
  \`x_norm = x / 255.0 - 0.5\`
- One-hot encoding for labels:  
  \`to_categorical(y)\`

---

**🚀 Training**

- Optimizer: Adam
- Loss: Categorical Crossentropy
- Metrics: Accuracy
- Trained for: _(edit: X epochs you used)_

You can modify hyperparameters to improve accuracy.

---

 **📊 Results**

| Metric | Value |
|-------|------|
| Training Accuracy | _(edit) \_% |
| Testing Accuracy | _(edit) \_% |

> You can also add accuracy/loss plots here

---

**🛠️ Technologies Used**

| Tool | Purpose |
|------|---------|
| Python | Programming |
| TensorFlow / Keras | Deep Learning |
| NumPy | Computation |
| Matplotlib/Seaborn | Visualization |
| Jupyter Notebook | Development |

