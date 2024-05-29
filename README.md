This project is a pipeline for creating UNet CNN models for pixel level segmentations of images for the diagnosis of affected cell tissue by ultrasound pulses.
Run model_trainer on training images. Run model_performance on model and test images to evaluate model performance graphically. model_performance creates logistic regression model based on output mappings.

Areas of Improvement:
Data normalization on training/testing images for better model generalization on test images.
Z-Domain transformations on the output image to create more robust logistic regression model.
