# Classification of Repeat Features using Deep Learning

This project aims to classify repeat features in DNA sequences using both supervised and unsupervised deep learning models. The code provided uses toy models and simplifies the problem for faster execution.

## Brief Description

The project uses DNA sequences of dog genomes to classify repeat features. The sequences are first padded and one-hot encoded. Then, two deep learning models are implemented:

1. An unsupervised model based on an LSTM autoencoder, which learns the latent representation of the input sequences and clusters them using K-means. The performance is evaluated using silhouette scores.
2. A supervised model based on a bidirectional LSTM, which learns to classify the sequences into different repeat feature classes. The model's performance is evaluated using accuracy.

## Model Architectures

### Unsupervised Model

The unsupervised model uses an LSTM autoencoder with the following architecture:

- Input Layer
- LSTM (64 units)
- RepeatVector
- LSTM (5 units, return_sequences=True)

The autoencoder is trained on the input sequences, and the encoder part is then used to obtain the latent representations. These representations are clustered using K-means, and the clustering performance is evaluated using silhouette scores.

### Supervised Model

The supervised model uses a bidirectional LSTM with the following architecture:

- Bidirectional LSTM (64 units, return_sequences=True)
- Bidirectional LSTM (64 units)
- Dense (34 units, ReLU activation)
- Dropout (0.5)
- Dense (7 units, Softmax activation)

The model is trained on one-hot encoded input sequences and their corresponding class labels. The performance is evaluated using accuracy.

Please note that these models are toy models and may not provide the best performance for classifying repeat features in DNA sequences. They are intended for demonstration purposes only.

## Dependencies

- pandas
- numpy
- scikit-learn
- TensorFlow
- Keras

## Files

- `unsupervised_model.py`: This file contains the code for the unsupervised model based on the LSTM autoencoder.
- `supervised_model.py`: This file contains the code for the supervised model based on the bidirectional LSTM.

## Usage

1. Download the `dog.txt` file containing the DNA sequences and place it in the same directory as the Python scripts.
2. Run `unsupervised_model.py` to execute the unsupervised model.
3. Run `supervised_model.py` to execute the supervised model.

