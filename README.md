# Image-Colorization-using-AutoEncoders

This project implements an **image colorization** model using a **Convolutional Autoencoder** with residual blocks in **TensorFlow/Keras**. The model is designed to take grayscale images as input and output colorized versions, making it a great tool for restoring or enhancing old black-and-white photos.

## Model Architecture

The model architecture is based on a **Convolutional Autoencoder** structure, consisting of:

1. **Encoder**:
   - Uses several convolutional layers with ReLU activation.
   - Incorporates **Batch Normalization** for faster convergence and stability.
   - Downsamples the input image using **MaxPooling** layers.

2. **Residual Block**:
   - Enhances feature extraction using a residual connection.
   - Helps in learning complex patterns and mitigates the vanishing gradient problem.

3. **Decoder**:
   - Upsamples the encoded features back to the original image size.
   - Uses convolutional layers to reconstruct the color channels.
   - Outputs an RGB image with values normalized using a **tanh** activation function.

### Model Summary

- **Input**: Grayscale image of shape `(32, 32, 1)`
- **Output**: Colorized image of shape `(32, 32, 3)`
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam

## Requirements

- Python 3.x
- TensorFlow 2.x
- NumPy
- Matplotlib (for visualization)

## Training

The model was trained using a dataset of grayscale images (`X_train_gray`) and their corresponding color images (`X_train`).

```python
# Training the model
history = autoencoder.fit(
    X_train_gray, X_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test_gray, X_test),
    callbacks=[checkpoint_cb, early_stopping_cb, reduce_lr_cb]
)
```

### Callbacks Used
- **ModelCheckpoint**: Saves the best model during training.
- **EarlyStopping**: Stops training if the validation loss does not improve for a specified number of epochs.
- **ReduceLROnPlateau**: Reduces the learning rate if validation loss plateaus.

## Results

- Achieved a Peak Signal-to-Noise Ratio (PSNR) of **~22.65**, which shows a good balance between the original and colorized images.
- The training and validation loss history is available for visualization to assess model performance over epochs.

## Usage

To run the model for your own grayscale images:
1. Load your image data.
2. Preprocess it to match the input shape `(32, 32, 1)`.
3. Use the `autoencoder.predict()` method to get colorized images.

```python
# Example prediction
colorized_images = autoencoder.predict(grayscale_images)
```

## How to Plot Training and Validation Loss

```python
import matplotlib.pyplot as plt

# Plot training & validation loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()
```

## License
This project is open-source and available for educational purposes.
