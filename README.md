# 🌈 Image Colorization using AutoEncoders

This project implements an **image colorization** model using a **Convolutional Autoencoder** enhanced with residual blocks in **TensorFlow/Keras**. The model is designed to take grayscale images as input and generate colorized versions, making it a valuable tool for restoring or enhancing old black-and-white photos.

## 📑 Project Overview

The goal of this project is to develop a deep learning model that can automatically colorize grayscale images using the **CIFAR-10 dataset**. The model leverages the powerful feature extraction capabilities of convolutional neural networks and residual blocks to produce realistic colorization of images.

### Workflow
![Work Flow](https://github.com/user-attachments/assets/febe6942-d960-45d5-aa54-6aebc8c9cab6)


## 📁 Dataset

The **CIFAR-10 dataset** was used for training and testing the model. This dataset consists of 60,000 color images divided into 10 classes, with each image having a resolution of 32x32 pixels. For this project, the images were converted to grayscale for the model's input, and the original color images served as the target output.

### Dataset Details:
- **Training Samples**: 50,000 grayscale-color pairs
- **Testing Samples**: 10,000 grayscale-color pairs
- **Image Dimensions**: `(32, 32)`

## 🧠 Model Architecture

The model architecture is based on a **Convolutional Autoencoder** with residual connections to improve feature extraction and colorization performance.

### Model Components:

1. **Encoder**:
   - Uses convolutional layers with **ReLU activation** for feature extraction.
   - Includes **Batch Normalization** for improved training stability.
   - Downsamples the input image using **MaxPooling** layers.

2. **Residual Block**:
   - Enhances feature extraction using residual connections.
   - Helps the model learn complex patterns and mitigates the vanishing gradient problem.

3. **Decoder**:
   - Upsamples the encoded features back to the original image size.
   - Uses convolutional layers to reconstruct the color channels.
   - Applies a **tanh activation** function in the output layer to generate RGB values.

### Model Summary:
- **Input Shape**: `(32, 32, 1)` (Grayscale image)
- **Output Shape**: `(32, 32, 3)` (Colorized RGB image)
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam

![Model Architecture](https://github.com/user-attachments/assets/dc96f908-bf1b-4f5e-9b84-b0fb7a55fdd9)


## 🔧 Requirements

Ensure you have the following dependencies installed:

- Python 3.x
- TensorFlow 2.x
- NumPy
- Matplotlib (for visualization)

You can install the required packages using:
```bash
pip install tensorflow numpy matplotlib
```

## 🚀 Training the Model

The model was trained using the following setup:

```python
# Compile the model
autoencoder.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
history = autoencoder.fit(
    X_train_gray, X_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test_gray, X_test),
    callbacks=[checkpoint_cb, early_stopping_cb, reduce_lr_cb]
)
```

### Callbacks Used:
- **ModelCheckpoint**: Saves the best model during training based on validation loss.
- **EarlyStopping**: Stops training if the validation loss does not improve for 15 consecutive epochs.
- **ReduceLROnPlateau**: Reduces the learning rate by a factor of 0.5 if validation loss plateaus.

## 📊 Results and Performance

- **Peak Signal-to-Noise Ratio (PSNR)**: Achieved a score of **~22.65**, indicating good image quality.
- **Mean Absolute Error (MAE)**: Low error value, suggesting accurate colorization.
- The model demonstrated stable convergence of training and validation losses.

### Loss Curve
The graph below shows the model's training and validation loss over 100 epochs:

![Loss Curve](https://github.com/user-attachments/assets/bf9ea32a-5056-4754-8760-6d5a3dba6de3)


To visualize the training and validation loss, use the following code:

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
```

## 📂 Usage

To use the trained model for colorizing your own grayscale images:

1. Load your grayscale images and preprocess them to shape `(32, 32, 1)`.
2. Use the `autoencoder.predict()` function to generate colorized outputs.

```python
# Example prediction
colorized_images = autoencoder.predict(grayscale_images)
```

## 🔍 Future Enhancements

- Integrating **Generative Adversarial Networks (GANs)** for more realistic colorization.
- Experimenting with **perceptual loss** using pre-trained networks like VGG to enhance colorization quality.
- Extending the model to support higher-resolution images for improved detail.

## 🤝 Contributing

Contributions are welcome! If you have suggestions for improvements or find any issues, feel free to submit a pull request or report an issue.
