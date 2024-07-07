# Face Recognition and Similar Image Search

## Overview
This project uses deep learning techniques to perform facial recognition and find similar images based on facial features. It leverages the MTCNN for face detection and VGGFace for feature extraction. The project also includes a supervised hashing mechanism to generate binary codes for the facial features, which helps in efficiently finding similar images.

## Requirements
To run this project, you need the following libraries:
- os
- numpy
- logging
- Pillow
- matplotlib
- scipy
- keras_vggface
- scikit-learn
- mtcnn
- tensorflow (required by keras_vggface)