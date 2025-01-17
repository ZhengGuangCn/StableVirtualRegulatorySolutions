"""
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
"""
import xml.etree.ElementTree as ET
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrow

csvreader3 = pd.read_excel('subgraph-final-RA-HE-mean-100.xlsx')
csvreader4 = pd.read_csv('mean_o-RA-HE-mean-100.csv',header=None)
csvreader4 = pd.DataFrame(csvreader4)
csvreader4 = csvreader4.T

weight = []
for i in range(0, 240):
    weight.append(abs(float(csvreader3.iloc[i][2])))
print(weight)

prediction = []
for i in range(0, 193):
    prediction.append(abs(float(csvreader4.iloc[i][2])))
print(prediction)

# Parse the XML file
tree = ET.parse(r'hsa04660-T-cell-draw.xml')
# Note that the XML file here is different from the one used during training.
# The XML file used for training has an additional layers element, and the layer structure is at the next level under layers
root = tree.getroot()

# Create a multigraph with directed edges
G = nx.MultiDiGraph()
node_data = {}
edge_weights = {}

# Traverse the XML hierarchy and add nodes and edges
for elem in root:
    if elem.tag == 'layer':
        layer_id = float(elem.attrib['id'])
        layer_name = elem.attrib['name']
        node_data[layer_id] = {'layer_name': layer_name}

        for entry in elem.findall('entry'):
            entry_id = entry.attrib['id']
            for graph in entry.findall('graphics'):
                name_value = graph.attrib.get('name', '')
                # Extract the first element (split by commas)
                entry_name = name_value.split(',')[0].strip()
                entry_type = graph.attrib['type']
                G.add_node(entry_id, layer_id=layer_id, entry_id=entry_id, entry_name=entry_name, entry_type=entry_type)
                node_data[entry_id] = {'entry_name': entry_name, 'entry_type': entry_type}

# Traverse the edges and record the edge and its corresponding weight
i = 0
for elem in root.findall('edge'):
    entry1 = elem.attrib['from']
    entry2 = elem.attrib['to']
    relation_type = elem.attrib['subtype']
    weight_value = weight[i]
    edge_weights[(entry1, entry2)] = weight_value
    G.add_edge(entry1, entry2, relation_type=relation_type)
    i += 1

factor = 0.023  # This factor is used to control what percentage of the edges are highlighted.
factor1 = float(factor * 100)

sorted_edges = sorted(edge_weights.items(), key=lambda item: item[1], reverse=True)
top_30_percent_edges = set(edge for edge, weight in sorted_edges[:int(len(sorted_edges) * factor)])
print(top_30_percent_edges)

# Calculate the color of edges and nodes
edge_colors = {}
node_colors = {}

# Get the minimum and maximum values of all prediction values for color mapping
min_pred = min(prediction)
max_pred = max(prediction)

# Map the prediction values to color intensity, with larger prediction values corresponding to deeper colors
def map_prediction_to_color(prediction_value, min_val, max_val):
    norm = mcolors.Normalize(vmin = min_val - 0.06, vmax = max_val + 0.07)
    cmap = plt.cm.Blues
    return cmap(norm(prediction_value))

for edge, weight in edge_weights.items():
    if edge in top_30_percent_edges:
        edge_colors[edge] = 'green'  # Highlight edge color
    else:
        edge_colors[edge] = 'lightgray'  # Fade the edge color

for u,v, data in G.edges(data=True):
    if data.get('relation_type') == 'inhibition' and (u,v) in top_30_percent_edges:
        edge_colors[(u, v)] = 'red'  # Set the edges with relation_type as 'inhibition' to red

# Determine the highlighted nodes
highlighted_nodes = set(node for edge in top_30_percent_edges for node in edge)
# Define the height and width of rectangular nodes
node_width = 0.12
node_height = 0.07
# node_width = 0.18
# node_height = 0.1

node_shapes = {
    node: 'c' if data.get('entry_type') == 'cell' else  # Draw cell types as circles
          'h' if data.get('entry_type') == 'compound' else   # Draw compound types as ellipses
          's'
    for node, data in G.nodes(data=True)
}
alphas = {node: 1 for node in G.nodes()}

for node in G.nodes():
    if node in highlighted_nodes:
        # Get the index of the node in the predictions
        node_index = list(G.nodes()).index(node)
        prediction_value = prediction[node_index]
        node_colors[node] = map_prediction_to_color(prediction_value, min_pred, max_pred)
    else:
        node_colors[node] = '#e0e0e0'  # Fade the node color
        alphas[node] = 0.1  # Make non-highlighted nodes more transparent

# Layout and plotting
pos = nx.multipartite_layout(G, subset_key="layer_id", scale=1.5, align="vertical")

# Manually adjust the x-coordinates of the nodes, expand the left and right boundaries, and make the nodes more spread out along the x-axis
for node, (x, y) in pos.items():
    pos[node] = (x * 1.9, y)

def draw_custom_nodes(ax, pos, nodes, node_width, node_height, node_colors, alphas, node_shapes):
    for node in nodes:
        x, y = pos[node]
        color = node_colors[node]
        alpha = alphas.get(node, 1)
        shape = node_shapes.get(node, 'o')

        if shape == 's':
            rect = patches.Rectangle((x - node_width / 2, y - node_height / 2),
                                     node_width, node_height,
                                     edgecolor='none', facecolor=color, alpha=alpha, lw=1.5)
            ax.add_patch(rect)
        elif shape == 'c':
            circle_radius = node_height * 0.8
            circle = patches.Circle((x, y), radius=circle_radius,
                                    edgecolor='none', facecolor=color, alpha=alpha, lw=1.5)
            ax.add_patch(circle)
        elif shape == 'h':
            ellipse = patches.Ellipse((x, y), width=node_width, height=node_height,
                                      edgecolor='none', facecolor=color, alpha=alpha, lw=1.5)
            ax.add_patch(ellipse)


fig, ax = plt.subplots(figsize=(32, 24))  # Adjust the canvas size to fit the distribution of nodes and edges

draw_custom_nodes(ax, pos, G.nodes(), node_width, node_height, node_colors, alphas, node_shapes)

node_labels = {
    node: 'Synovial\nmacrophage' if node_data[node].get('entry_name') == 'Synovial macrophage'
    else 'Synovial\nfibroblast' if node_data[node].get('entry_name') == 'Synovial fibroblast'
    else node_data[node].get('entry_name')
    for node in G.nodes()
}

label_colors = {node: 'black' if node in highlighted_nodes else '#a9a9a9' for node in G.nodes()}  # The label color of non-highlighted nodes is dark gray


# The label colors for highlighted and non-highlighted nodes
highlighted_label_color = 'black'
non_highlighted_label_color = '#a9a9a9'

# Get labels for highlighted and non-highlighted nodes
highlighted_labels = {node: node_labels[node] for node in highlighted_nodes}
non_highlighted_labels = {node: node_labels[node] for node in G.nodes() if node not in highlighted_nodes}

# Draw labels for highlighted and non-highlighted nodes respectively
nx.draw_networkx_labels(
    G, pos, labels=highlighted_labels, font_size=10, font_weight='bold',
    font_color=highlighted_label_color, ax=ax
)

nx.draw_networkx_labels(
    G, pos, labels=non_highlighted_labels, font_size=10, font_weight='bold',
    font_color=non_highlighted_label_color, ax=ax
)


# Define a function to adjust the position of the arrow
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

# Drawing edges and using T-arrows
def draw_edges(ax, pos, edges, edge_colors, node_width, node_height, top_30_percent_edges):
    for source, target, data in edges:
        start_pos = pos[source]
        end_pos = pos[target]
        arrow_start, arrow_end = adjust_arrow_position(start_pos, end_pos, node_width, node_height)

        # Select the arrow type according to the type of edge
        if data['relation_type'] == 'inhibition':
            # Drawing T-arrows
            ax.plot([arrow_start[0], arrow_end[0]], [arrow_start[1], arrow_end[1]], color=edge_colors[(source, target)], lw=1)

            # Calculate the direction vector perpendicular to the edge
            dx, dy = arrow_end[0] - arrow_start[0], arrow_end[1] - arrow_start[1]
            norm = (dx ** 2 + dy ** 2) ** 0.5
            dx, dy = dx / norm, dy / norm

            # Draw T-arrows perpendicular to the edge to reduce the offset
            perp_dx, perp_dy = -dy, dx
            T_end1 = (arrow_end[0] + perp_dx * 0.01, arrow_end[1] + perp_dy * 0.01)
            T_end2 = (arrow_end[0] - perp_dx * 0.01, arrow_end[1] - perp_dy * 0.01)
            ax.plot([T_end1[0], T_end2[0]], [T_end1[1], T_end2[1]], color=edge_colors[(source, target)], lw=1)
        else:
            # Drawing normal arrows
            ax.annotate("",
                        xy=arrow_end, xytext=arrow_start,
                        arrowprops=dict(arrowstyle='->', color=edge_colors[(source, target)], lw=1, mutation_scale=5))

draw_edges(ax, pos, G.edges(data=True), edge_colors, node_width, node_height, top_30_percent_edges)

ax.set_xlim(min(pos[node][0] for node in G.nodes()) - 0.45, max(pos[node][0] for node in G.nodes()) + 0.45)
ax.set_ylim(min(pos[node][1] for node in G.nodes()) - 0.3, max(pos[node][1] for node in G.nodes()) + 0.3)

# The following sections are drawn for legend
descriptions = ['Activation ', 'Inhibition']

inset_ax = fig.add_axes([0.107, 0.72, 0.13, 0.15])  # [x, y, width, height] Scale relative to the main image
x_start = 1
width = 0.2
height = 0.5
bar_height = 0.2
y_spacing = 0.3
num_arrows = 2

for i in range(num_arrows):
    y_start = 0.95 + i * y_spacing+0.05

    if i == 0:
        arrow_body = lines.Line2D(
            [x_start - 0.8 + 0.08 + 0.62, x_start + height - bar_height - 0.5 + 0.08 + 0.62],
            [y_start, y_start],
            linestyle='-', color='green', linewidth=2
        )
        arrow_head = patches.Polygon(
            [
                (x_start + height - 0.7 + 0.08 + 0.62, y_start-0.05 ),
                (x_start + height - 0.7 + 0.08 + 0.62, y_start + 0.05),
                (x_start + height + bar_height - 0.72 + 0.08 + 0.62, y_start)
            ],
            closed=True, color='green'
        )
        inset_ax.add_patch(arrow_head)
        inset_ax.add_line(arrow_body)

    elif i == 1:
        arrow_body = lines.Line2D(
            [x_start - 0.8 + 0.08 + 0.62, x_start + height - bar_height - 0.5 + 0.16 + 0.08 + 0.62],
            [y_start, y_start],
            linestyle='-', color='red', linewidth=2
            )

        vertices = [
            (x_start + height - 0.67 + 0.16 + 0.08 + 0.62, y_start + width - 0.3),
            (x_start + height - 0.67 + 0.16 + 0.08 + 0.62, y_start - width + 0.3),
            (x_start + height - 0.7 + 0.16 + 0.08 + 0.62, y_start - width + 0.3),
            (x_start + height - 0.7 + 0.16 + 0.08 + 0.62, y_start + width - 0.3)
        ]

        t_arrow_head = patches.Polygon(vertices, closed=True, color='red')
        inset_ax.add_line(arrow_body)
        inset_ax.add_patch(t_arrow_head)

    text_x = x_start + height + 0.33
    text_y = y_start
    inset_ax.text(
        text_x, text_y,
        descriptions[i],
        fontsize=10, color='black',
        verticalalignment='center',
        horizontalalignment='left'
    )

# Setting the Submap Display Range
inset_ax.set_xlim(0, 3.5)
inset_ax.set_ylim(0, num_arrows * y_spacing + 1)
inset_ax.axis('off')

# Add subimage border to main image
bbox = inset_ax.get_position()
border = patches.Rectangle(
    (bbox.x0 + 0.03, bbox.y0 + 0.076), bbox.width - 0.065, bbox.height - 0.085,
    linewidth=1, edgecolor='black', facecolor='none',
    transform=fig.transFigure  # Use the coordinates of the main image
)
fig.add_artist(border)
plt.savefig(f'Top {len(top_30_percent_edges)} varied regulations of Th17 cell differentiation process between RA patients and healthy controls-blue.png', dpi=300)  # 保存图形，提高分辨率
plt.show()
