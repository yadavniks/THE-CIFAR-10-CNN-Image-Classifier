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
-


---

 **📊 Results**

| Metric | 
|-------|------|
| Training Accuracy | 
| Testing Accuracy | 



---

**🛠️ Technologies Used**

| Category | Tools | Purpose |
|----------|-------|---------|
| Language | **Python** | Core programming |
| Deep Learning | **TensorFlow, Keras** | CNN model creation & training |
| Computation | **NumPy** | Matrix operations & preprocessing |
| Visualization | **Matplotlib, Seaborn** | Accuracy/loss plots + results |
| Image Handling | **OpenCV** | Input preprocessing |
| Deployment | **Streamlit** | Web app for model inference |
| Development | **Jupyter Notebook** | Experimenting & debugging |


