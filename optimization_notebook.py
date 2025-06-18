import numpy as np
import matplotlib.pyplot as plt


# Initialize weights and biases for a simple one-hidden-layer network
# Weights are initialized with small random values to break symmetry

class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

        # --- Helper Function - Plot Activations --- #

def plot_activation(activation: callable, title: str):
  X = np.linspace(-5, 5, 100)
  Y = activation(X)

  plt.figure(figsize=(6, 4))
  plt.plot(X, Y)
  plt.title(title)
  plt.xlabel('Input')
  plt.ylabel('Output')
  plt.grid(True)
  plt.show()