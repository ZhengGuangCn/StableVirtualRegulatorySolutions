"""
The following programs are used to train the model, obtain the model parameters, training loss, and the model's outputs.

First, a fully connected neural network is constructed based on the reconstructed XML file. Since the training network we use is fully connected,
but the actual regulatory network is sparse, it is necessary to initialize the network connection weights before starting training by setting the weights of nonexistent connections to 0.
The training data used by the network is the augmented data generated in the data_extend.py file.
The targets_data contains the node values of each layer of real samples. By augmenting the input of these real samples and using them for network training, the outputs of each layer in the network are forced to approximate targets_data.
This approach enables the network to learn the implicit information in the regulatory pathways.
In every training iteration, the connection weights must be constrained to ensure they adhere to the regulatory network structure. Otherwise, the network would train itself into a fully connected structure.
After training each model, the input data of real samples in targets_data is fed into the network to obtain the network's predicted values. Finally, each trained model is saved for subsequent analysis.

File Introduction:
xml_path: location of the refactored xml file,The new XML file contains information about the layers, including the layer identifiers, layer numbers, and details about each layer's nodes.
            For the nodes, it includes their names, IDs, and types. Additionally, it provides information about the connections, specifying the source and target nodes for each connection, as well as the type of connection.
data_path: The training data used by the network is the augmented data generated in the data_extend.py file, which includes both the network's input and output.
targets_data: The targets_data contains the node values of each layer of real samples. By augmenting the input of these real samples and using them for network training, the outputs of each layer in the network are forced to approximate targets_data.
            This approach enables the network to learn the implicit information in the regulatory pathways.

"""
import torch
import csv
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class NeuralNetwork(nn.Module):
    def __init__(self, num_nodes_per_layer):
        super(NeuralNetwork, self).__init__()
        self.layers = nn.ModuleList([
            nn.Linear(num_nodes_per_layer[i], num_nodes_per_layer[i + 1])
            for i in range(len(num_nodes_per_layer) - 1)
        ])

    def forward(self, x):
        intermediate_outputs = []  # used to store the output of each layer
        for layer in self.layers[:-1]:
            x = F.leaky_relu(layer(x),negative_slope= 0.01)    # Using leaky_relu as an activation function
            intermediate_outputs.append(x)
        x = self.layers[-1](x)
        intermediate_outputs.append(x)
        return x, intermediate_outputs  # returns the final output and the output of all intermediate layers

def record_layer_outputs(outputs, filename): # record the output of each layer into filename
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        row = []
        for i, output in enumerate(outputs):
            if output.dim() > 1:  # check if the output is multidimensional, usually 2D
                row.extend(output.flatten().tolist())  # spreading the tensor and adding it to the row
            else:
                row.extend(output.tolist())
        writer.writerow(row)
        writer.writerow([])

def print_weights_and_biases(model, filename): # Record the non-zero weights of the connections and the biases of the nodes for each trained network model into a file.
    # Extract model weights and biases
    weights = []
    biases = []

    for layer in model.layers:
        layer_weights = layer.weight.data.cpu().numpy().flatten()
        layer_biases = layer.bias.data.cpu().numpy().flatten()

        # Filtering non-zero weights
        weights.extend([w for w in layer_weights if w != 0])
        biases.extend(layer_biases)

    # If the file already exists, read the existing data
    if os.path.exists(filename):
        existing_data = pd.read_csv(filename)
        # Adding a new column
        existing_data[f"Non_Zero_Weights_{len(existing_data.columns)//2 + 1}"] = pd.Series(weights).dropna().reset_index(drop=True)
        existing_data[f"Biases_{len(existing_data.columns)//2 + 1}"] = pd.Series(biases).dropna().reset_index(drop=True)
    else:
        existing_data = pd.DataFrame({
            "Non_Zero_Non_One_Weights_1": pd.Series(weights).dropna().reset_index(drop=True),
            "Biases_1": pd.Series(biases).dropna().reset_index(drop=True)
        })

    # write back
    existing_data.to_csv(filename, index=False)

def apply_custom_initialization(model, edges, layers, num_nodes_per_layer):  # Assigning initial values to network weights
    with torch.no_grad():
        tagged_pairs = set()
        from_layer_index_last = 0
        for edge in edges:
            from_node, to_node, subtype = edge['from'], edge['to'], edge['subtype']
            from_layer_index = next((i for i, nodes in enumerate(layers) if from_node in nodes), None)
            to_layer_index = next((i for i, nodes in enumerate(layers) if to_node in nodes), None)
            if from_layer_index_last != from_layer_index:
                for from_index in range(from_layer_nodes_num):
                    for to_index in range(to_layer_nodes_num):
                        if (from_index,to_index) in tagged_pairs:
                            continue
                        else:
                            model.layers[from_layer_index_last].weight[to_index, from_index].fill_(0)
                tagged_pairs.clear()
            if from_layer_index is not None and to_layer_index == from_layer_index + 1:
                from_index = layers[from_layer_index].index(from_node)
                to_index = layers[to_layer_index].index(to_node)
                if subtype == 'equivalence':
                    model.layers[from_layer_index].weight[to_index, from_index].fill_(1)
                    model.layers[from_layer_index].bias[to_index].fill_(0)
                    tagged_pairs.add((from_index,to_index))
                elif subtype == 'binding/association' or subtype =='secretion':
                    weight = model.layers[from_layer_index].weight[to_index, from_index].item()
                    model.layers[from_layer_index].weight[to_index, from_index] = torch.tensor(weight)
                    tagged_pairs.add((from_index, to_index))
                elif subtype == 'activation' or subtype == "indirect activation":
                    weight = model.layers[from_layer_index].weight[to_index, from_index].item()
                    if weight < 0:
                        model.layers[from_layer_index].weight[to_index, from_index] = torch.abs(torch.tensor(weight))
                    tagged_pairs.add((from_index, to_index))
                elif subtype == 'inhibition':
                    weight = model.layers[from_layer_index].weight[to_index, from_index].item()
                    if weight > 0:
                        model.layers[from_layer_index].weight[to_index, from_index] = -torch.abs(torch.tensor(weight))
                    tagged_pairs.add((from_index, to_index))
            from_layer_nodes_num = num_nodes_per_layer[from_layer_index]
            to_layer_nodes_num = num_nodes_per_layer[to_layer_index]
            from_layer_index_last = from_layer_index
            if edge == edges[-1]:
                for from_index in range(from_layer_nodes_num):
                    for to_index in range(to_layer_nodes_num):
                        if (from_index,to_index) in tagged_pairs:
                            continue
                        else:
                            model.layers[from_layer_index_last].weight[to_index, from_index].fill_(0)
                tagged_pairs.clear()

def enforce_constraints(model, edges, layers,num_nodes_per_layer):  # Adding strong constraints on network weights and node bias
    with torch.no_grad():
        tagged_pairs = set()
        from_layer_index_last = 0
        for edge in edges:
            from_node, to_node, subtype = edge['from'], edge['to'], edge['subtype']
            from_layer_index = next((i for i, nodes in enumerate(layers) if from_node in nodes), None)
            to_layer_index = next((i for i, nodes in enumerate(layers) if to_node in nodes), None)
            if from_layer_index_last != from_layer_index:
                for from_index in range(from_layer_nodes_num):
                    for to_index in range(to_layer_nodes_num):
                        if (from_index,to_index) in tagged_pairs:
                            continue
                        else:
                            model.layers[from_layer_index_last].weight[to_index, from_index].fill_(0)
                tagged_pairs.clear()
            if from_layer_index is not None and to_layer_index == from_layer_index + 1:
                from_index = layers[from_layer_index].index(from_node)
                to_index = layers[to_layer_index].index(to_node)
                if subtype == 'equivalence':
                    # Set weights and biases to be constant values
                    model.layers[from_layer_index].weight[to_index, from_index].fill_(1)
                    model.layers[from_layer_index].bias[to_index].fill_(0)
                    tagged_pairs.add((from_index, to_index))
                elif subtype == 'binding/association' or subtype =='secretion':
                    weight = model.layers[from_layer_index].weight[to_index, from_index].item()
                    model.layers[from_layer_index].weight[to_index, from_index] = torch.tensor(weight)
                    tagged_pairs.add((from_index, to_index))
                elif subtype == 'activation' or subtype == "indirect activation":
                    weight = model.layers[from_layer_index].weight[to_index, from_index].item()
                    if weight < 0:
                        model.layers[from_layer_index].weight[to_index, from_index] = torch.abs(torch.tensor(weight))
                    tagged_pairs.add((from_index, to_index))
                elif subtype == 'inhibition':
                    weight = model.layers[from_layer_index].weight[to_index, from_index].item()
                    if weight > 0:
                        model.layers[from_layer_index].weight[to_index, from_index] = -torch.abs(torch.tensor(weight))
                    tagged_pairs.add((from_index, to_index))
            from_layer_nodes_num = num_nodes_per_layer[from_layer_index]
            to_layer_nodes_num = num_nodes_per_layer[to_layer_index]
            from_layer_index_last = from_layer_index
            if edge == edges[-1]:
                for from_index in range(from_layer_nodes_num):
                    for to_index in range(to_layer_nodes_num):
                        if (from_index,to_index) in tagged_pairs:
                            continue
                        else:
                            model.layers[from_layer_index_last].weight[to_index, from_index].fill_(0)
                tagged_pairs.clear()

def load_network_structure(xml_path):  # Read the network structure stored in the xml file
    tree = ET.parse(xml_path)
    root = tree.getroot()
    layers = []
    edges = []
    for layer in root.find("layers"):
        layers.append([entry.get('id') for entry in layer])
    for edge in root.find('edges'):
        edges.append({
            'from': edge.get('from'),
            'to': edge.get('to'),
            'subtype': edge.get('subtype')
        })
    return layers, edges

def make_weights_positive(model):
    with torch.no_grad():
        for layer in model.layers:
            # Adjust only negative values in the weight matrix
            negative_weights = layer.weight < 0
            layer.weight[negative_weights] = layer.weight[negative_weights].abs()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Network structure initialisation and loading
xml_path = '../data/hsa04660-T-cell.xml'   # location of the refactored xml file
layer_nodes, edge_info = load_network_structure(xml_path)
num_nodes_per_layer = [len(layer) for layer in layer_nodes]

net = NeuralNetwork(num_nodes_per_layer).to(device)
apply_custom_initialization(net, edge_info, layer_nodes, num_nodes_per_layer)

# load train data
data_path = '../data/T-RA16-train.csv'
data = pd.read_csv(data_path)
inputs = data.iloc[:, :12].values   # input data
outputs = data.iloc[:, 12:].values  # output data

# divide the data set
inputs_train, inputs_test, outputs_train, outputs_test = train_test_split(inputs, outputs, test_size=0.2)

# Convert to Tensor and move data to GPU
inputs_train = torch.tensor(inputs_train, dtype=torch.float32).to(device)
outputs_train = torch.tensor(outputs_train, dtype=torch.float32).to(device)
inputs_test = torch.tensor(inputs_test, dtype=torch.float32).to(device)
outputs_test = torch.tensor(outputs_test, dtype=torch.float32).to(device)

optimizer = optim.Adam(net.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Read target data
targets_data = pd.read_excel('../data/T-real_samples.xlsx')
# Here is the real data that the network nodes need to approximate during the training phase
target_values = targets_data.iloc[12:, 2].values
target_values = pd.to_numeric(target_values, errors='coerce')  # Convert non-numeric values to NaN
print(target_values)
print(len(target_values))

# Remove the number of nodes in the input layer
num_nodes_per_layers = num_nodes_per_layer[1:]

# Targets sliced by layer
start_idx = 0
targets_per_layer = []
for num_nodes in num_nodes_per_layers:
    end_idx = start_idx + num_nodes
    layer_targets = target_values[start_idx:end_idx]
    # Convert the sliced target value to a tensor and move it to the GPU
    targets_per_layer.append(torch.tensor(layer_targets, dtype=torch.float32).to(device))
    start_idx = end_idx

# targets checking
'''for i, layer_targets in enumerate(targets_per_layer):
    print(f"Layer {i+1} targets: {layer_targets}")'''

def train_model():
    for repeat in range(500):   # This is how many models to train
        print(f"Training repetition {repeat + 1}")
        net = NeuralNetwork(num_nodes_per_layer).to(device)  # Move new network instances to the GPU as well
        optimizer = optim.Adam(net.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        loss_history1 = []
        for epoch in range(1200):  # Number of training rounds per model
            optimizer.zero_grad()
            final_output, intermediate_outputs = net(inputs_train)  # Get the final output and the output of all intermediate layers
            loss = 0  # Loss of final output to target output
            # Calculation of losses per layer
            layer_losses = []
            for output, target in zip(intermediate_outputs, targets_per_layer):
            # Let the output values of the nodes at each level of the network be sufficiently close to the target values of the nodes at each level of the network
                layer_loss = criterion(output, target.expand_as(output))
                layer_losses.append(layer_loss.item())
                loss += layer_loss  # Accumulation of losses per layer

            if epoch % 100 == 99:
                print(f'Epoch {epoch + 1}, Loss: {loss.item()}')
                print("Layer losses: ", layer_losses)

            loss.backward()  # backward propagation
            optimizer.step()  # Updating parameters
            loss_history1.append(loss.item())
            enforce_constraints(net, edge_info, layer_nodes, num_nodes_per_layer)  # Enforcement of constraints

        # make_weights_positive(net)
        filename = 'final_weights.csv'
        print_weights_and_biases(net, filename)
        x0 = targets_data.iloc[:12, 2].values  # Network inputs used to validate as well as the effect of the completed trained model
        x0 = torch.tensor(x0, dtype=torch.float32).to(device)
        print(x0)
        _, outputs = net(x0.unsqueeze(0))
        record_layer_outputs(outputs, 'layer_outputs.csv')   # Saves the output values of each layer of the network

        # Save each model's loss to a file
        with open('layer_losses.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([f'{loss.item()}'])

        plt.figure()
        plt.plot(range(len(loss_history1)), loss_history1)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()

        # Save each network model
        torch.save(net, "T-RA16_train_{}.pth".format(repeat))
        print("Model saved")

torch.backends.cudnn.benchmark = True
train_model()
