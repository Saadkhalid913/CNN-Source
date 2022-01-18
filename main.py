import numpy as np
from tensorflow.keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from scipy.signal import convolve2d, correlate2d

def sigmoid(x, derivative = False):
    if derivative:
      return sigmoid(x) * (1 - sigmoid(x))
    return 1 / (1 + np.exp(-x))
def MSELoss(y_pred, y_truth, derivative = False):
    '''
        Mean Squared Error function with derivative
    '''
    if derivative:
        return -1 * (y_truth - y_pred)
    return np.power((y_truth - y_pred), 2) / 2

class Dense():
    def __init__(self, input_neurons: int, output_neurons: int, activation):
        # Create weights of shape (num_inputs, num_outputs)
        self.Weights = np.random.randn(input_neurons, output_neurons)

        # Create bias of shape (1, num_outputs)
        self.Biases = np.random.randn(1, output_neurons)

        # Define Activation Functions 
        self.Activation = activation

        self.input_neruons = input_neurons

        self.output_neurons = output_neurons

        self.Old_A0 = None
        self.Old_Z1 = None

    def Forward(self, x: np.array):

        # Check if dot product between inputs 
        # and weights is defined 
        assert x.shape[1] == self.Weights.shape[0]

        # Unactivated Dot Product 
        Z1 = np.dot(x, self.Weights)

        self.Old_A0 = x
        self.Old_Z1 = Z1

        # Activated Dot Product 
        A1 = self.Activation(Z1)

        return Z1, A1


    def Backprop(self, Z1: np.array, A0: np.array, gradient: np.array, learning_rate = 0.0, next_weights=None):
        # Z1: Unactivated outputs for this layer 
        # A0: Activated outputs of previous layer 
        # Gradient: Gradient with respect to the weights 

        ## Assert that the gradient
        ## has the same number of neurons 
        ## as the outputs 
        assert gradient.shape[1] == self.Weights.shape[1]

        ## Copy the weights to pass to previous 
        ## Layers gradient 
        OldWeights = self.Weights.copy()

        ## if there are no forward weights, 
        ## we take the gradient only 
        if not next_weights:
            delta = self.Activation(Z1, derivative = True) *  gradient # (samples, output_weights) 
        else:
            # (samples, output_weights) * (output_weights, next_layer_weights) DOT (samples, output_weights)
            # The total error is defined as element-wise multiplication of 
            # derivative of activation with respect to its input 
            # and the gradient of each respective weight 
            delta = self.Activation(Z1, derivative = True) *  np.dot(next_weights, gradient) 
        
        # weight is updated with respect 
        # to the error times each input 
        # value of the neuron in the previous 
        # layer 
        WeightUpdate = np.dot(A0.T, delta)  
        BiasUpdate = np.sum(delta, axis = 0, keepdims=True)

        # we update both of the weights
        self.Weights += -learning_rate * WeightUpdate
        self.Biases += -learning_rate * BiasUpdate
        
        # we return the gradient 
        # with respect to the current weights 
        return np.dot(delta, OldWeights.T)



class ConvLayer():
  def __init__(self, input_shape: tuple, output_depth: int, kernel_size: int):
    input_depth, input_rows, input_cols = input_shape 
    self.Z1 = None 
    self.input_depth = input_depth
    self.output_depth = output_depth
    self.output_shape = (output_depth, input_rows - kernel_size + 1 , input_cols - kernel_size + 1)
    kernels_shape = (output_depth, input_depth, kernel_size, kernel_size)
    bias_shape = self.output_shape
    self.kernels = np.random.rand(*kernels_shape)
    self.biases = np.random.rand(*bias_shape)

  def forward(self, input):
    self.Z1 = input 
    output = self.biases.copy()

    for i in range(self.output_depth):
      for j in range(self.input_depth):
        output[i] += correlate2d(input[j], self.kernels[i][j], mode="valid")

    return output

  def backprop(self, gradient, learning_rate = 0.01):
    kernel_gradient = np.zeros(self.kernels.shape)
    input_gradient = np.zeros(self.Z1.shape)

    for i in range(self.output_depth):
      for j in range(self.input_depth):
        kernel_gradient[i][j] += convolve2d(self.Z1[j], gradient[i], mode="valid")
        input_gradient[j] += convolve2d(gradient[i], self.kernels[i][j], mode="full")

    self.kernels += -1 * learning_rate * kernel_gradient
    self.biases += -1 * learning_rate * gradient

    return input_gradient

(x_train, y_train), (x_test, y_test) = mnist.load_data()



def preprocess_data(x, y, max):
    x = x[ : max]
    y = y[ : max]
    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype("float32") / 255
    y = to_categorical(y)
    y = y.reshape(len(y), 1, 10)

    return x, y

x_train, y_train = preprocess_data(x_train, y_train, 1100)
x_test, y_test = preprocess_data(x_test, y_test, 100)

# initialize layers 

C1 = ConvLayer((1,28,28), 7, 5)
D1 = Dense(7 * 24 * 24, 100, sigmoid)
D2 = Dense(100, 10, sigmoid)


epochs = 100
for epoch in range(epochs):
  total_correct = 0
  for index, sample in enumerate(x_train):
    ConvOutput = C1.forward(sample)
    ActivatedConvOutput = sigmoid(ConvOutput)

    Flattened_Output = ActivatedConvOutput.reshape(1, -1)

    Z0, A0 = D1.Forward(Flattened_Output)
    Z1, A1 = D2.Forward(A0)

    if (np.argmax(A1) == np.argmax(y_train[index])):
      total_correct +=1 

    loss_gradient = MSELoss(A1, y_train[index], derivative = True)

    grad = D2.Backprop(Z1, A0, loss_gradient, learning_rate = 0.1)
    grad = D1.Backprop(Z0, Flattened_Output, grad, learning_rate = 0.1)

    grad = grad.reshape(C1.output_shape)

    C1.backprop(grad, learning_rate = 0.1)
  
  print(f"Epoch #{epoch + 1}, Accuracy: {total_correct * 100 / len(x_train)}%")

correct = 0
for index, sample in enumerate(x_test):
  ConvOutput = C1.forward(sample)
  sigmoid_output = sigmoid(ConvOutput)
  sigmoid_output = sigmoid_output.reshape(1, -1)
  dummy, A0 = D1.Forward(sigmoid_output)
  dummy, A1 = D2.Forward(A0)

  if np.argmax(A1, axis =1 ) == np.argmax(y_test[index], axis = 1):
    correct +=1 

  
print( correct * 100 / len(y_test), "%")