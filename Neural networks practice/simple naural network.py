import numpy as np

inputs = [1.0, 2.0, 3.0, 2.5]
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
biases = [2.0, 3.0, 0.5]

outputs = np.dot(weights, inputs) + biases

layer = []
for i in range(len(biases)):
    node_value = biases[i]
    for weight, input in zip(weights[i], inputs):
        node_value += weight*input
    layer.append(node_value)

print(layer)

print(outputs)