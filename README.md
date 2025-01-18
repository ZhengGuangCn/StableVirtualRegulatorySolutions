Introduction:

This study aims to simulate the regulatory networks within organisms using neural networks to further uncover biological regulatory associations. 
We designed a method to solve the regulatory network of KEGG-related pathways associated with rheumatoid arthritis based on fully connected neural networks. 
Rheumatoid arthritis-related pathways from the KEGG database were transformed into neural networks for training. 
Given the sparsity of connections in the pathway networks, we introduced strong constraints to allow the fully connected networks to better fit the pathway networks. 
Finally, we employed a virtual mean solution in the trained model to validate the method. The results demonstrate that our approach achieves excellent predictive performance.


Dependencies:

python 3.8;
numpy 1.20.1;
scikit-learn 0.24.1;
scipy 1.6.2;
torch 1.10.2;
R 4.2.3.


result:

These are the model training results for six KEGG pathways associated with rheumatoid arthritis. The folder annotations provide explanations for each folder.


data:

The folder stores 20 real sample datasets for each of the six KEGG pathways, along with XML files containing the network structure.


code:

The experimental code for this study, with relatively detailed comments included in each Python file


How to use:

1. data_extend.py:
Since the number of real samples is too small, the data needs to be expanded by adding a Gaussian noise of 0.1 to the real samples and expanding to 10,000 samples
data: The real sample input values and output values used for data augmentation.


2. model_train.py :
The following programs are used to train the model, obtain the model parameters, training loss, and the model's outputs.

First, a fully connected neural network is constructed based on the reconstructed XML file. Since the training network we use is fully connected,
but the actual regulatory network is sparse, it is necessary to initialize the network connection weights before starting training by setting the weights of nonexistent connections to 0. 
The training data used by the network is the augmented data generated in the data_extend.py file.
The targets_data contains the node values of each layer of real samples. By augmenting the input of these real samples and using them for network training, the outputs of each layer in the network are forced to approximate targets_data.
This approach enables the network to learn the implicit information in the regulatory pathways.
In every training iteration, the connection weights must be constrained to ensure they adhere to the regulatory network structure. Otherwise, the network would train itself into a fully connected structure.
After training each model, the input data of real samples in targets_data is fed into the network to obtain the network's predicted values. Finally, each trained model is saved for subsequent analysis.

xml_path: location of the refactored xml file,The new XML file contains information about the layers, including the layer identifiers, layer numbers, and details about each layer's nodes. 
For the nodes, it includes their names, IDs, and types. Additionally, it provides information about the connections, specifying the source and target nodes for each connection, as well as the type of connection.
data_path: The training data used by the network is the augmented data generated in the data_extend.py file, which includes both the network's input and output.
targets_data: The targets_data contains the node values of each layer of real samples. By augmenting the input of these real samples and using them for network training, the outputs of each layer in the network are forced to approximate targets_data. 
This approach enables the network to learn the implicit information in the regulatory pathways.


3. model_validation.py:
The purpose of the following programs is to use the final model solution to validate the model's performance in distinguishing between healthy and disease samples.

To facilitate our subsequent visualization, during the training phase, we filtered out connections with weights equal to 0 when saving the model parameters. 
Note that although we are using a fully connected network, the actual regulatory network is sparse, so during training, 
we force the weights of connections that do not exist in the network to be set to 0. Now, to reassign the final solution parameters to the network, 
we adopted the following approach: we load a model saved during the training phase as our validation network, read its model parameters, 
and then overwrite the non-zero weight values with the parameters from the final solution. This way, we perform a one-to-one replacement, 
with the premise that all connections except for those with weights forced to 0 will have non-zero weights.
 After doing this, all the parameters of the current network have been updated to the final solution's parameters.

Finally, the input values of each real disease sample and healthy sample are fed into the network to obtain the output values of each layer. 
After flattening the network's output values, the Euclidean distance between these output values and the corresponding real sample's node values at each layer is calculated.
A smaller distance indicates that the sample aligns more closely with the disease pattern described by the model solution. 
For example, if we use the model solution for disease samples, a smaller distance for a real sample after input suggests a higher probability that the real sample is a disease sample.

model_path: The reference model is used to provide the positions of non-zero weights in the network. We replace the parameters of this reference model one by one with the parameters from the final model solution.
weight: The weights of all connections in the final model solution are all non-zero.
bias: The biases of each node in the final model solution.
input_file: The values of each real sample, including input node values and values of nodes in each layer, are used as follows: the input node values are fed into the model to obtain the predicted values of nodes in each layer, while the values of nodes in each layer are used to validate the model's predictive performance.


4. mean_solution.py:
The following code selects the models with the top-ranked losses, calculates the average of their parameters as the mean solution, and writes it to a new file.

First, read the files that record connection weights, node biases, output results, and loss values. The values in these files correspond to each other in the order of the trained models. 
Then, sort the loss values in ascending order and retrieve the corresponding indices to filter the connection weights, node biases, and output results in the same order. 
Subsequently, compute the mean of the parameters from the top N models with the smallest loss values as a relatively stable virtual solution and write it to a new file.

df: File storing non-zero weights and biases
df_outputs: A file used to store layer outputs
df_loss: A file used to store the loss values of each model


5. visualization_highlight_connections.py:
The following code visualizes the N connections with the largest regulatory differences

csvreader3: In csvreader3, the data records the diseased model solution and the healthy model solution.
        It calculates the absolute value of the relative difference, ABS((diseased model weight - healthy model weight) / healthy model weight).
        This set of values represents the regulatory difference for each connection between the diseased and healthy models.
        The larger the value, the greater the difference in that connection between the diseased and healthy models.
csvreader4: In csvreader4, the data records the layer outputs of the diseased model and the healthy model.
        It calculates the absolute value of the relative difference, ABS((diseased model layer output - healthy model layer output) / healthy model layer output).
        This set of values represents the difference for each node between the diseased and healthy models.
        The larger the value, the greater the difference in that node between the diseased and healthy models.
tree: Structures used to store network diagrams, includes various attribute values for nodes and edges


6. visualization_network.py:
The purpose of this program is to Visualizing the complex regulation network with detailed information

The specific process is roughly divided into the following five steps:
    1. Get the data of the network graph: ① use pd.read_csv() to read the data of the nodes and edges in the network graph,
and then save it to list; ② use ET.parse() to parse the xml file, and then get the names and types of the nodes, the start and 
end points of the edges and the types, and then save it to dictionary and list, respectively.
    2. Assign values to the relevant attributes of nodes and edges in the network graph: use the nx.MultiDiGraph() method to 
create the network graph, and then assign values to various attributes of nodes and edges in G.nodes() and G.edges().① For nodes 
the main attributes of prediction and node type are assigned, where in node type, gene is specified as rectangle, compound as 
oval, and cell as circle. In the node color, through the mcolors.LinearSegmentedColormap() method to map the node's prediction 
value to its color display, the color used here is blue, the larger the prediction value, the darker the blue; ② For the edge is 
mainly weight, edge style and color assignment, which provides: inhibition type of edge for the red T-arrow, activation type of 
edge for the green ordinary arrow, binding/association type of edge for the black ordinary arrow, equality type of edge for the 
gray straight line, secretion type of edge for the green ordinary arrow(secretion is activation in another sense, so it is also 
indicated here by a plain green arrow), in addition to the For edges with the INDIRECT keyword in the type description, 
the dotted line is used.
    3. Custom functions for drawing edges and nodes: ① for nodes: use patches.Rectangle(), patches.Circle() and patches.Ellipse() to 
draw three different types of nodes; ② for edges: use ax.plot() to draw T-arrows, ax.annotate() to draw normal arrows and lines. 
Use the ax.text() method to display non-1 prediction values to the corresponding edges.
    4. draw network diagram: Use the two functions customized in step 3 for drawing nodes and edges to draw the pathway network 
graph, use ax.set_xlim(), ax.set_ylim() method to adjust the position of the pathway network diagram to make the overall look more beautiful.
    5. Drawing notes: here for the drawing of notes in order to explain the various types of edges, the use of lines.Line2D () to draw 
a variety of arrows on behalf of the edge, the use of fig.add_axes () method to adjust the notes to the appropriate location so that 
the overall look of a more coordinated.

csv_file1: Used to store the node's prediction value
csv_file2: Used to store the weights of the edges
tree: Structures used to store network diagrams, includes various attribute values for nodes and edges


