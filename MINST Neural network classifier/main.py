import numpy as np
from keras.datasets import mnist
from matplotlib import pyplot

#loading the dataset
(train_X, train_y), (test_X, test_y) = mnist.load_data()

#printing the shapes of the vectors 
print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  '  + str(test_X.shape))
print('Y_test:  '  + str(test_y.shape))

#batch_size = 16

def vizualize_data():
    for i in range(9):  
        pyplot.subplot(330 + 1 + i)
        pyplot.imshow(train_X[i], cmap=pyplot.get_cmap('gray'))
        #print(len(train_X[i]))
    pyplot.show()

def one_hot(labels, n_classes):
    return np.eye(n_classes)[labels.reshape(-1)]

class Layer_Dense:
    def __init__(self, n_inputs, n_neuron):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neuron)
        self.biases = np.zeros((1, n_neuron))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
    def forward(self, inputs):
        inputs = 1 / (1 + np.exp(-inputs))
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

class Activation_Sigmoid:
    def forward(self, inputs):
        self.output = 1 / (1 + np.exp(-inputs))

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]

        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods


#Layer1 = Layer_Dense(5, 16)
Layer1 = Layer_Dense(784, 16)
Activation1 = Activation_ReLU()

Layer2 = Layer_Dense(16, 16)
Activation2 = Activation_ReLU()

Layer3 = Layer_Dense(16, 10)
Activation3 = Activation_Softmax()

#one_hot_train_y = one_hot(np.array(train_y), 10)

Loss_function = Loss_CategoricalCrossentropy()

#Layer1.forward(x)
#Layer1.forward(train_X[0].reshape(1, 784))
Layer1.forward(train_X[0].reshape(-1))
Activation1.forward(Layer1.output)

Layer2.forward(Activation1.output)
Activation2.forward(Layer2.output)

Layer3.forward(Activation2.output)
Activation3.forward(Layer3.output)

loss = Loss_function.calculate(Activation3.output, train_y)

print("Loss: ", loss)
print(Activation3.output)
print(np.sum(Activation3.output))
print(Layer3.output)
print(Activation2.output)
print(Layer2.output)
print(Activation1.output)
print(Layer1.output)
#The loss function takes all the truth values, the code is intended for batches of x and y.