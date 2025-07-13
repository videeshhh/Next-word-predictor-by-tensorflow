#  Next Word Prediction using TensorFlow

A simple next word prediction model built using TensorFlow and trained on a small custom dataset due to hardware limitations. The model demonstrates the basic principles of language modeling and text prediction.

---

## Project Overview

This project implements a **next word prediction** model using a recurrent neural network (RNN) built with TensorFlow and Keras. The model is trained to predict the most probable next word given a sequence of words.

---

##  Technologies Used

- Python 3.11
- TensorFlow 2.5.0
- NumPy
- Keras
- Natural Language Processing (NLP)
- Jupyter Notebook (for development) in V.S. code

---

## Dataset

- A **small, custom dataset** was used, containing basic English text of sherlock holmes.
- The limited dataset size was due to **hardware constraints**, which affected model accuracy and generalization capability.
- The dataset was preprocessed with:
  - Tokenization
  - Lowercasing
  - Sequence padding

---

##  Model Architecture

- Embedding layer to convert words into dense vectors
- LSTM layer to learn sequential dependencies
- Dense (fully connected) output layer with softmax activation

![image_alt](https://github.com/videeshhh/Next-word-predictor-by-tensorflow/blob/main/Screenshot%202025-07-13%20151119.png?raw=true)
![image_alt](https://github.com/videeshhh/Next-word-predictor-by-tensorflow/blob/main/Screenshot%202025-07-13%20151129.png?raw=true)
