# Video Classification with Transfer Learning

This project aims to build a video classifier that can distinguish between 5 different sport activities using transfer learning with pre-trained models. The main approach is to train a Convolutional Neural Network (CNN) for image classification and turn it into a video classifier using moving averages.

## Dataset

The dataset consists of images and videos for five classes of sport activities: baseball, basketball, fencing, tennis, and volleyball. Images are stored in separate folders under the `Sport Images` folder, and videos are stored in the `Sport Videos` folder.

## Requirements

* Python 3.7+
* Keras
* TensorFlow
* OpenCV
* NumPy
* Pandas
* scikit-learn

## Steps

1. Data Exploration and Pre-processing
    - Divide the 3280 images into training (70%), validation (15%), and test sets (15%).
    - Zero-pad or resize the images to have the same size.

2. Transfer Learning for Image Classification
    - Use pre-trained models ResNet50, EfficientNetB0, and VGG16.
    - Perform empirical regularization and image augmentation (crop, zoom, rotate, flip, contrast, and translate).
    - Use ReLU activation functions in the last layer and a softmax layer, along with batch normalization and a dropout rate of 20% as well as ADAM optimizer.
    - Train the networks for at least 50 epochs (preferably 100 epochs) and perform early stopping using the validation set.
    - Plot the training and validation errors vs. epochs.
    - Report the Confusion Matrix, Precision, Recall, Accuracy, and F1 score for the model on both training and test sets.

3. Video Classification Using Moving Averages
    - Reuse the validation and test data to train the network without seriously overfitting it.
    - Apply at least 100 equally spaced frames of each video to the model to obtain probability predictions from the softmax layer.
    - Calculate the average probability vector for each video and select the class with the maximum probability.
    - Report the Confusion Matrix, Precision, Recall, Accuracy, and F1 score for the model on the test data (681 videos).

## License

This project is licensed under the MIT License.
