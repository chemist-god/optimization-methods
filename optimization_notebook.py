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

                                         # --- Activation Functions --- #


# --- ReLU Activation Function --- #
def relu(x):
    return np.maximum(0, x)

# plotting ReLU
plot_activation(relu, "ReLU Activation Function")


# --- LeakyReLU Activation Function --- #
def leaky_relu(x):
    return np.maximum(0.1*x, x)

# plotting ReLU
plot_activation(leaky_relu, "ReLU Activation Function")


# --- Derivative ReLU Activation Function --- #
def relu_derivative(x):
    return np.where(x > 0, 1, 0)

plot_activation(relu_derivative, "Derivative ReLU Activation Function")


# --- Tanh Activation Function --- #
def tanh(x):
    return np.tanh(x)

# plotting tanh
plot_activation(tanh, "Tanh Activation Function")


# --- Derivative Tanh Activation Function --- #
def tanh_derivative(x):
    return 1 - np.tanh(x)**2

# plotting tanh
plot_activation(tanh_derivative, "Derivative Tanh Activation Function")