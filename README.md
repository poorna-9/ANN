# 🧠 NeuroLite

A lightweight, modular, and fully customizable Artificial Neural Network (ANN) framework built from scratch using NumPy — no TensorFlow or PyTorch involved!

## 🚀 Features

- Build deep neural networks using simple class-based API.
- Add **any number of layers** dynamically — similar to TensorFlow/Keras `Sequential`.
- Fully supports:
  - Multiple **activation functions** (ReLU, Sigmoid, Tanh, etc.)
  - Common **loss functions** (MSE, Cross-Entropy, etc.)
  - **Backpropagation**
  - **Batch training**
- Trained and tested on **MNIST** dataset with successful results.
- 100% implemented using **NumPy only**.
- Designed with **modularity** in mind — easy to extend or debug.


```python
from neuro_lite import NeuralNetwork, Dense, ReLU, Softmax, CrossEntropyLoss

model = NeuralNetwork()
model.add(Dense(784, 128))
model.add(ReLU())
model.add(Dense(128, 64))
model.add(ReLU())
model.add(Dense(64, 10))
model.add(Softmax())

model.compile(loss=CrossEntropyLoss(), learning_rate=0.01)
model.fit(X_train, y_train, epochs=10, batch_size=32)
model.evaluate(X_test, y_test)

