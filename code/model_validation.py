"""
    The purpose of the following programs is to use the final model solution to validate the model's performance in distinguishing between healthy and disease samples.

    In order to make it easier for us to visualise it later, we filtered out those connections with a weight of 0 when we saved the parameters of the model during the training phase.
    Note: Since we are using a fully connected network, but the actual regulatory network is sparse, the weights of those connections that do not actually exist in the network are forced to 0 in the training phase.
    Now we want to reassign the parameters of the final solution to the network, we take the following approach:
    load one of the models saved in the training phase as our validation network and read its model We do this by loading a model saved in the training phase as our validation network, reading its model parameters,
    and overriding the non-zero weights of the model with the parameters of the final solution, thus replacing them one by one, with the caveat that all the connections have non-zero weights except for the ones that we forced to be 0. After doing this,
    all the parameters of the network are now updated to the parameters of the final solution.

    Finally, the input values of each real disease sample and healthy sample are fed into the network to obtain the output values of each layer.
    After flattening the network's output values, the Euclidean distance between these output values and the corresponding real sample's node values at each layer is calculated.
    A smaller distance indicates that the sample aligns more closely with the disease pattern described by the model solution.
    For example, if we use the model solution for disease samples, a smaller distance for a real sample after input suggests a higher probability that the real sample is a disease sample.

    model_path: The reference model is used to provide the positions of non-zero weights in the network. We replace the parameters of this reference model one by one with the parameters from the final model solution.
    weight: The weights of all connections in the final model solution are all non-zero.
    bias: The biases of each node in the final model solution.
    input_file: The values of each real sample, including input node values and values of nodes in each layer, are used as follows:
            the input node values are fed into the model to obtain the predicted values of nodes in each layer, while the values of nodes in each layer are used to validate the model's predictive performance.
"""

import torch
import pandas as pd
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

# Define network classes with the same structure as the original model
class NeuralNetwork(nn.Module):
    def __init__(self, num_nodes_per_layer):
        super(NeuralNetwork, self).__init__()
        self.layers = nn.ModuleList([
            nn.Linear(num_nodes_per_layer[i], num_nodes_per_layer[i + 1])
            for i in range(len(num_nodes_per_layer) - 1)
        ])

    def forward(self, x):
        intermediate_outputs = []
        for layer in self.layers[:-1]:
            x = F.leaky_relu(layer(x), negative_slope=0.01)
            intermediate_outputs.append(x)
        x = self.layers[-1](x)
        intermediate_outputs.append(x)
        return x, intermediate_outputs

model_path = "T_RA16_0.pth"
original_model = torch.load(model_path)

# Extracting weights and biases from the original model
reference_weights = []
reference_biases = []
for layer in original_model.layers:
    if isinstance(layer, nn.Linear):
        reference_weights.append(layer.weight.data.clone())  # clone weight
        reference_biases.append(layer.bias.data.clone())     # clone bias

# Loading new non-zero weights and bias data
weight = pd.read_csv('mean_a_closest_100_1_virtual-T-RA16.csv')
bias = pd.read_csv('mean_b_closest_100_1_virtual-T-RA16.csv')

weight = weight.apply(pd.to_numeric, errors='coerce').iloc[:, 0].to_list()
print(len(weight))
print(weight)
bias = bias.apply(pd.to_numeric, errors='coerce').iloc[:, 0].to_list()
print(len(bias))
print(bias)

# Create a new model with the same structure as the original model
num_nodes_per_layer = [layer.weight.size(1) for layer in original_model.layers if isinstance(layer, nn.Linear)]
num_nodes_per_layer.append(original_model.layers[-1].weight.size(0))  # 最后一层输出维度
net = NeuralNetwork(num_nodes_per_layer)

# Fill in the new weight matrix, leaving the zero position of the reference model unchanged
layer_idx = 0
weight_idx = 0
bias_idx = 0
# New weights and bias assignment code section
for layer in net.layers:
    if isinstance(layer, nn.Linear):
        ref_weight_matrix = reference_weights[layer_idx]
        new_weight_matrix = torch.zeros_like(ref_weight_matrix)

        # Fill new weights to non-zero positions
        for i in range(ref_weight_matrix.size(0)):
            for j in range(ref_weight_matrix.size(1)):
                if ref_weight_matrix[i, j] != 0:
                    new_weight_matrix[i, j] = weight[weight_idx]
                    weight_idx += 1

        # Update weights
        with torch.no_grad():
            layer.weight.copy_(new_weight_matrix)

        # Update bias, populate bias directly to each layer in sequence
        new_bias = torch.zeros_like(reference_biases[layer_idx])
        for i in range(new_bias.size(0)):
            new_bias[i] = bias[bias_idx]
            bias_idx += 1

        # Disable requires_grad temporary assignment
        layer.bias.requires_grad_(False)
        layer.bias.copy_(new_bias)
        layer.bias.requires_grad_(True)  # Restore requires_grad

        layer_idx += 1

# Verify that weights and biases are loaded successfully
for i, layer in enumerate(net.layers):
    if isinstance(layer, nn.Linear):
        print(f"Layer {i} weights after update:", layer.weight)
        print(f"Layer {i} biases after update:", layer.bias)

# Testing the effectiveness of model validation
for i in range(1, 8):
    input_file = pd.read_excel('T-real_samples.xlsx')
    input_data = pd.to_numeric(input_file.iloc[:12, i], errors='coerce').values
    input_data = torch.tensor(input_data, dtype=torch.float32)
    # print("input_data:", input_data)
    output, intermediate_outputs = net(input_data)

    # flatten out intermediate_outputs
    intermediate_outputs_flat = torch.cat([out.flatten() for out in intermediate_outputs]).detach().numpy()
    real_node_values_flat = input_file.iloc[12:, i].values
    # print(real_node_values_flat)

    # Check that intermediate_outputs_flat and real_node_values_flat are the same length.
    if len(intermediate_outputs_flat) != len(real_node_values_flat):
        raise ValueError("The lengths of intermediate_outputs_flat and real_node_values_flat do not match.")

    # Calculate the Euclidean distance
    euclidean_distance = np.linalg.norm(intermediate_outputs_flat - real_node_values_flat)
    print(euclidean_distance)
