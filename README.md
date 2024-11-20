# **Image Authenticity Detector**

## **Project Overview**

The **Image Authenticity Detector** is a machine learning project designed to classify images as either real or AI-generated. This is done using a **Generative Adversarial Network (GAN)**, which consists of two models: a generator and a discriminator. The **discriminator** is trained on a dataset containing both real and AI-generated images to enhance its ability to distinguish between the two.

The project uses Keras and TensorFlow to build the discriminator model, which classifies images as **real (0)** or **fake (1)**.

## **Data Source**

The dataset used for training the model is the **CIFAKE dataset**, sourced from Kaggle. It contains a collection of 60,000 synthetic images and 60,000 real images from CIFAR-10. The dataset is used to train the model on distinguishing between real and AI-generated images.

**Dataset Link**: [CIFAKE: Real and AI-Generated Synthetic Images](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)

Further details about the dataset can be found in the paper by Bird and Lotfi, 2024: *CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images*.

## **Project Updates**

This project uses the **CCLE GAN model** for the classification task, which is adapted into a binary classifier for detecting AI-generated images.

## **Installation Instructions**

To set up the project, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/Redgerd/Synthetic-Images-Detector
    cd Synthetic-Images-Detector
    ```

2. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Update the README to reflect any new changes or dependencies if necessary.

### **Usage**

Once the dependencies are installed, you can start training the model or use the pre-trained model to classify images.

```python
# Example: Load and use pre-trained model for classification
from keras.models import load_model
model = load_model('path_to_model')

# Assuming 'image' is a pre-processed image to classify
prediction = model.predict(image)
print("Real" if prediction == 0 else "AI-generated")
```

## **Links and References**
- **Papers with Code**: [CIFAKE on Papers with Code](https://paperswithcode.com/dataset/cifake-real-and-ai-generated-synthetic-images)
- **Original Paper**: Bird, J.J. and Lotfi, A., 2024. *CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images*. IEEE Access.

