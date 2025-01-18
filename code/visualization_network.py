'''
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

File Introduction：
csv_file1: Used to store the node's prediction value
csv_file2: Used to store the weights of the edges
tree: Structures used to store network diagrams, includes various attribute values for nodes and edges
'''

import xml.etree.ElementTree as ET
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as patches
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.lines as lines

csv_file1 = r'../data/mean_a_closest_100_1_virtual-T-RA16.csv'
csv_file2 = r'../data/mean_a_closest_100_1_virtual-T-RA16.csv'
csv1 = pd.read_csv(csv_file1, header=None)
csv2 = pd.read_csv(csv_file2)

# Get the prediction value of the node
entry_prediction = []
draw_prediction = []
for i in range(0, 193):
    tempo = csv1.iloc[0][i]
    tempo = round(tempo, 2)  # Retain two decimal places
    entry_prediction.append(tempo)
    draw_prediction.append('\n' + '(' + str(tempo) + ')')
print(entry_prediction)

# Get the weight of the edge
edge_weight = []
for i in range(0, 240):  # -1
    tempo = csv2.iloc[i][4]
    tempo = round(tempo, 3)  # Retain three decimal places
    edge_weight.append(tempo)
print(edge_weight)

# Parsing XML files
tree = ET.parse(r'../data/hsa04660-T-cell-draw.xml')
root = tree.getroot()

# Create an empty multimap
G = nx.MultiDiGraph()
node_data = {}

# Get nodes from XML files
for elem in root:
    if elem.tag == 'layer':
        layer_id = elem.attrib['id']
        layer_name = elem.attrib['name']
        node_data[layer_id] = {'layer_name': layer_name}

        # Loop over, get node from “graphics name”.
        for entry in elem.findall('entry'):
            entry_id = entry.attrib['id']
            for graph in entry.findall('graphics'):
                name_value = graph.attrib.get('name', '')
                entry_name = name_value.split(',')[0].strip()
                entry_type = graph.attrib['type']

            # Add node
            G.add_node(entry_id, layer_id=layer_id, entry_id=entry_id, entry_name=entry_name, entry_type=entry_type)
            node_data[entry_id] = {'entry_name': entry_name, 'entry_type': entry_type}

# Get the sides
alledges = []
i = 0
for elem in root.findall('edge'):
    entry1 = elem.attrib['from']
    for node in G.nodes():
        if node == entry1:
            entry_1 = node
    entry2 = elem.attrib['to']
    for node in G.nodes():
        if node == entry2:
            entry_2 = node
    relation_type = elem.attrib['subtype']
    edge_tuple = (entry_1, entry_2, relation_type, edge_weight[i])
    i += 1
    alledges.append(edge_tuple)
    G.add_edge(entry_1, entry_2, relation_type=relation_type)

# Assign values to the weight attributes of edges
count = 0
visited_edges1 = set()
for source, target, key, data in G.edges(data=True, keys=True):
    edge_pair = (source, target)
    if edge_pair in visited_edges1:
        continue
    visited_edges1.add(edge_pair)
    for entry1, entry2, relation_type, weight in alledges:
        if source == entry1 and target == entry2:
            G.edges[(source, target, key)]['weight'] = weight
    count = count + 1

for source, target, key, data in G.edges(data=True, keys=True):
    weight = G.edges[(source, target, key)].get('weight', None)

# Assign values to other attributes of the edge
i = 0
visited_edges = set()
for source, target, key, data in G.edges(data=True, keys=True):
    edge_pair = (source, target)
    if edge_pair in visited_edges:  # Skip if this pair of edges has already been processed
        continue
    visited_edges.add(edge_pair)
    relation_type = data.get('relation_type')
    weight = G.edges[(source, target, key)]['weight']

    # Set lines solid or dashed based on the type of edge in the XML file
    if relation_type == 'indirect activation':
        linestyle = 'dashed'
    else:
        linestyle = 'solid'

    # Set other attributes based on the relationship type of the edge
    if relation_type == 'activation':
        color = (0, 100 / 255, 0)  # Dark green
        size = 0.5
        arrowstyle = '->'
    elif relation_type == 'binding/association':
        color = (0, 0, 0)  # black
        size = 0.5
        arrowstyle = '->'
    elif relation_type == 'equivalence':
        color = (190 / 255, 190 / 255, 190 / 255)  # Gray
        size = 0.5
        arrowstyle = '-'
        linestyle = 'solid'
    elif relation_type == 'indirect activation':
        color = (0, 100 / 255, 0)  # Dark green
        size = 0.5
        arrowstyle = '->'
    elif relation_type == 'inhibition':
        color = (1, 0, 0)  # red
        size = 0.5
        arrowstyle = '->'

        # Update the edge attributes in graph G
    G.edges[(source, target, key)]['color'] = color
    G.edges[(source, target, key)]['size'] = size
    G.edges[(source, target, key)]['arrowstyle'] = arrowstyle
    G.edges[(source, target, key)]['style'] = linestyle
    G.edges[(source, target, key)]['arrow_color'] = color
    G.edges[(source, target, key)]['edge_weight'] = weight
    G.edges[(source, target, key)]['lineSize'] = 1
    i += 1

# Define how the diagram is laid out
pos = nx.multipartite_layout(G, subset_key="layer_id", scale=1.5, align="vertical")

# Adjust the x-coordinates of the nodes to expand the left and right boundaries so that the nodes are more spread out on the x-axis
for node, (x, y) in pos.items():
    pos[node] = (x * 2.0, y)

# Custom node drawing functions
def draw_custom_nodes(ax, pos, nodes, node_width, node_height, node_colors, alphas, node_shapes):
    for node in nodes:
        x, y = pos[node]
        color = node_colors[node]
        alpha = alphas[node]
        shape = node_shapes[node]

        # Rectangular nodes
        if shape == 's':
            rect = patches.Rectangle((x - node_width / 2, y - node_height / 2),
                                     node_width, node_height,
                                     edgecolor='black', facecolor=color, alpha=alpha, lw=0)
            ax.add_patch(rect)

        # Circular nodes
        elif shape == 'o':
            circle_radius = node_height * 0.9
            circle = patches.Circle((x, y), radius=circle_radius,
                                    edgecolor='black', facecolor=color, alpha=alpha, lw=0)
            ax.add_patch(circle)


# Node Properties
node_width = 0.21
node_height = 0.09
node_colors = {}
alphas = {}
node_shapes = {}
color_alphas = {}
com = 0

# Get the minimum and maximum of all prediction values for color mapping
min_pred = min(entry_prediction)
max_pred = max(entry_prediction)


# Maps prediction values to color shades, larger prediction values correspond to darker blues
def map_prediction_to_color(prediction_value, min_val, max_val):
    norm = mcolors.Normalize(vmin=min_val - 0.06, vmax=max_val + 0.07)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'custom_blue',
        [
            (0.9, 0.95, 1.0),  # Light blue
            (0.6, 0.7, 0.8),  # Medium blue

        ]
    )
    return cmap(norm(prediction_value))

# Assign values to various attributes of a node
for node in G.nodes():
    prediction_value = entry_prediction[com]
    color = map_prediction_to_color(prediction_value, min_pred, max_pred)
    node_colors[node] = color
    alphas[node] = 1
    node_shapes[node] = 'o'
    com += 1

# Assign values to node shape attributes by node type
for node, data in G.nodes(data=True):
    entry_type = data.get('entry_type')
    if entry_type == 'cell':
        node_shape = 'o'
    elif entry_type == 'rectangle' or entry_type == 'gene':
        node_shape = 's'
    elif entry_type == 'lon':
        node_shape = '8'
    node_shapes[node] = node_shape

# Resize the canvas to fit the distribution of nodes and edges
fig, ax = plt.subplots(figsize=(23, 14))

# Draw nodes using custom functions
draw_custom_nodes(ax, pos, G.nodes(), node_width, node_height, node_colors, alphas, node_shapes)

# Label drawing
node_labels = {node: node_data[node]['entry_name'] + biass for node, biass in zip(G.nodes(), draw_prediction)}
nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=6, font_weight='bold', ax=ax)

# Define functions to adjust the position of arrows
def adjust_arrow_position(start, end, node_width, node_height):
    x0, y0 = start
    x1, y1 = end
    dx, dy = x1 - x0, y1 - y0

    if dx > 0:
        x0 += (5 * node_width) / 9
        x1 -= (5 * node_width) / 9
    else:
        x0 -= (5 * node_width) / 9
        x1 += (5 * node_width) / 9

    return (x0, y0), (x1, y1)

# Custom functions for drawing edges
def draw_edges(ax, pos, edges, node_width, node_height):
    for source, target, data in edges:
        start_pos = pos[source]
        end_pos = pos[target]
        arrow_start, arrow_end = adjust_arrow_position(start_pos, end_pos, node_width, node_height)

        # T-arrow drawing
        if data['relation_type'] == 'inhibition':
            ax.plot([arrow_start[0], arrow_end[0]], [arrow_start[1], arrow_end[1]], color=data['color'],
                    linestyle=data['style'], lw=float(data['lineSize']))

            # Compute the direction vector perpendicular to the edge
            dx, dy = arrow_end[0] - arrow_start[0], arrow_end[1] - arrow_start[1]
            norm = (dx ** 2 + dy ** 2) ** 0.5
            dx, dy = dx / norm, dy / norm  # 单位化

            # Draw T-arrows perpendicular to the edge to reduce the offset
            perp_dx, perp_dy = -dy, dx
            T_end1 = (arrow_end[0] + perp_dx * 0.01, arrow_end[1] + perp_dy * 0.01)
            T_end2 = (arrow_end[0] - perp_dx * 0.01, arrow_end[1] - perp_dy * 0.01)
            ax.plot([T_end1[0], T_end2[0]], [T_end1[1], T_end2[1]], color=data['color'], lw=float(data['lineSize']))

        # Straight line (without arrows) drawing
        elif data['relation_type'] == 'equivalence':
            ax.annotate("",
                        xy=arrow_end, xytext=arrow_start,
                        arrowprops=dict(arrowstyle='-', color=data['color'], lw=float(data['lineSize']),
                                        linestyle=data['style'],
                                        mutation_scale=5))
        # Plain arrow drawing
        else:
            ax.annotate("",
                        xy=arrow_end, xytext=arrow_start,
                        arrowprops=dict(arrowstyle='->', color=data['color'], lw=float(data['lineSize']),
                                        linestyle=data['style'],
                                        mutation_scale=5))

            # Display weights with values other than 1 on their sides
        if data['edge_weight'] != 1:
            weight_label = f"{data['edge_weight']:.3f}"  # Formatted weights displayed (3 decimal places)
            print(weight_label)
            # Get the angle of the arrow
            angle = np.degrees(np.arctan2(arrow_end[1] - arrow_start[1], arrow_end[0] - arrow_start[0]))

            # Fine-tune the angle so that the text is displayed in roughly the same direction as the arrow (as far to the top left of the line as possible)
            if angle < 0:
                angle = angle + 10
            if angle > 0:
                angle = angle - 8

                # Automatically angle text based on arrow direction
            ax.text((start_pos[0] + end_pos[0]) / 2, (start_pos[1] + end_pos[1]) / 2 + 0.015, weight_label,
                    horizontalalignment='center', verticalalignment='center', fontsize=3, color=data['color'],
                    fontweight='bold',
                    rotation_mode='anchor', rotation=angle)  # 粗体

# Use custom functions to draw edges
draw_edges(ax, pos, G.edges(data=True), node_width, node_height)

# Adjust network boundaries
ax.set_xlim(min(pos[node][0] for node in G.nodes()) - 0.5, max(pos[node][0] for node in G.nodes()) + 0.5)
ax.set_ylim(min(pos[node][1] for node in G.nodes()) - 0.2, max(pos[node][1] for node in G.nodes()) + 0.5)

# Title
plt.title("T-cell(HE)  Network")

# Arrays describing types of arrows
descriptions = [
    'Activation ',
    'Indirect Activation',
    'Equivalence ',
    'Binding/Association',
    'Inhibition'
]

# Adjust the position and size of the note in the main image
inset_ax = fig.add_axes([0.107, 0.72, 0.13, 0.15])

# Define the parameters of the T-arrow
x_start = 1  # x-coordinate of the left center of the arrow
width = 0.2  # Width of horizontal line of T (half of vertical horizontal line)
height = 0.5  # Height of vertical line of T
bar_height = 0.2  # Height of horizontal horizontal line of T
y_spacing = 0.6  # Vertical spacing between arrows
num_arrows = 5  # number of arrows

# Cyclic drawing of arrows to diagram notes
for i in range(num_arrows):
    y_start = 0.95 + i * y_spacing  # adjust the starting y-coordinate of each arrow
    if i < 4:  # The first four are normal arrows
        if i == 0:
            arrow_body = lines.Line2D(
                [x_start - 0.8 + 0.08 + 0.62, x_start + height - bar_height - 0.5 + 0.08 + 0.62],
                # x-coordinates of start and end points
                [y_start, y_start],  # y-coordinates of start and end points
                linestyle='-', color='green', linewidth=2
            )
            arrow_head = patches.Polygon(
                [
                    (x_start + height - 0.7 + 0.08 + 0.62, y_start - 0.15),  # Left bottom point
                    (x_start + height - 0.7 + 0.08 + 0.62, y_start + 0.15),  # Vertex point
                    (x_start + height + bar_height - 0.72 + 0.08 + 0.62, y_start)  # Bottom right
                ],
                closed=True, color='green'
            )
        elif i == 1:
            arrow_body = lines.Line2D(
                [x_start - 0.8 + 0.08 + 0.62, x_start + height - bar_height - 0.5 + 0.08 + 0.62],
                [y_start, y_start],
                linestyle='--', color='green', linewidth=2
            )
            arrow_head = patches.Polygon(
                [
                    (x_start + height - 0.7 + 0.08 + 0.62, y_start - 0.15),
                    (x_start + height - 0.7 + 0.08 + 0.62, y_start + 0.15),
                    (x_start + height + bar_height - 0.72 + 0.08 + 0.62, y_start)
                ],
                closed=True, color='green'
            )
        elif i == 2:
            arrow_body = lines.Line2D(
                [x_start - 0.8 + 0.08 + 0.62, x_start + height - bar_height - 0.5 + 0.08 + 0.75],
                [y_start, y_start],
                linestyle='-', color='grey', linewidth=2
            )


        elif i == 3:
            arrow_body = lines.Line2D(
                [x_start - 0.8 + 0.08 + 0.62, x_start + height - bar_height - 0.5 + 0.08 + 0.62],
                [y_start, y_start],
                linestyle='-', color='black', linewidth=2
            )
            arrow_head = patches.Polygon(
                [
                    (x_start + height - 0.7 + 0.08 + 0.62, y_start - 0.15),
                    (x_start + height - 0.7 + 0.08 + 0.62, y_start + 0.15),
                    (x_start + height + bar_height - 0.72 + 0.08 + 0.62, y_start)
                ],
                closed=True, color='black'
            )
        inset_ax.add_line(arrow_body)  # Add arrow head
        inset_ax.add_patch(arrow_head)  # Add arrow body

    else:  # T-arrow
        # Drawing the arrow body
        if i % 2 == 0:
            arrow_body = lines.Line2D(
                [x_start - 0.8 + 0.08 + 0.62, x_start + height - bar_height - 0.5 + 0.16 + 0.08 + 0.62],
                # x-coordinates of start and end points
                [y_start, y_start],  # y-coordinates of start and end points
                linestyle='-', color='red', linewidth=2
            )
        inset_ax.add_line(arrow_body)

        # Define the vertices at the top of the T-arrow (in clockwise order)
        vertices = [
            (x_start + height - 0.67 + 0.16 + 0.08 + 0.62, y_start + width + 0.01),  # 右上角
            (x_start + height - 0.67 + 0.16 + 0.08 + 0.62, y_start - width - 0.01),  # 右下角
            (x_start + height - 0.7 + 0.16 + 0.08 + 0.62, y_start - width - 0.01),  # 左下角
            (x_start + height - 0.7 + 0.16 + 0.08 + 0.62, y_start + width + 0.01)  # 左上角
        ]

        # Create polygons to represent the top of a T-arrow
        t_arrow_head = patches.Polygon(vertices, closed=True, color='red')

        # Add arrow tops to subgraphs
        inset_ax.add_patch(t_arrow_head)

    # Add a text description
    text_x = x_start + height + 0.33  # Set the x-coordinate of the text
    text_y = y_start  # Set the y-coordinate of the text
    inset_ax.text(
        text_x, text_y,
        descriptions[i],
        fontsize=10, color='black',
        verticalalignment='center',
        horizontalalignment='left'
    )

# Hide subgraph axes
inset_ax.set_xlim(0, 3.5)
inset_ax.set_ylim(0, num_arrows * y_spacing + 1)
inset_ax.axis('off')  # Hide subgraph axes

# Add subimage borders to the main image
bbox = inset_ax.get_position()
border = patches.Rectangle(
    (bbox.x0 + 0.03, bbox.y0 + 0.018), bbox.width - 0.03, bbox.height - 0.025,
    linewidth=1, edgecolor='black', facecolor='none',
    transform=fig.transFigure  # Use the coordinates of the main image
)
fig.add_artist(border)

# Save Picture
plt.savefig('T-cell(HE).png', format='png', dpi=300)
