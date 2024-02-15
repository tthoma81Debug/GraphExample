# NOTE: You will need to run pip install networkx and maybe pip install matplotlib to resolve dependencies
# Importing required libraries for graph operations and visualization
import heapq
import matplotlib.pyplot as plt
import networkx as nx

# Example graph represented as an adjacency list, where each node is connected to others with a distance
graph = {
    'A': [('B', 1), ('C', 3)],
    'B': [('A', 1), ('D', 5), ('E', 1)],
    'C': [('A', 3), ('F', 5)],
    'D': [('B', 5)],
    'E': [('B', 1), ('F', 1)],
    'F': [('C', 5), ('E', 1)]
}

# Positions for each node when we visualize the graph
positions = {
    'A': (0, 1),
    'B': (1, 2),
    'C': (1, 0),
    'D': (2, 3),
    'E': (2, 1),
    'F': (3, 0)
}

# Define nodes representing special locations on the graph
nodes_to_avoid = {'D'}  # Node 'D' represents an ice cream shop (to avoid)
nodes_to_include = {'E'}  # Node 'E' represents a small park (to include in the route)

# Custom A* search function
def a_star_search(graph, start, end):
    # Initialize the open set with the starting node and zero cost
    open_set = [(0, start, [])]
    heapq.heapify(open_set)  # Convert list to a priority queue
    closed_set = set()  # Initialize an empty set to keep track of visited nodes

    while open_set:
        # Pop the node with the lowest score (cost + heuristic) from the queue
        current_score, current, path = heapq.heappop(open_set)

        # If the current node is the goal, return the path and score
        if current == end:
            return path + [current], current_score

        # Add current node to the set of visited nodes
        closed_set.add(current)

        # Loop through the neighboring nodes of the current node
        for node, weight in graph[current]:
            # Skip this neighbor if it has already been visited
            if node in closed_set:
                continue

            # Add or remove weight depending on whether the node should be avoided or favored
            if node in nodes_to_avoid:
                weight += 10  # Add weight for nodes to avoid
            elif node in nodes_to_include:
                weight -= 10  # Subtract weight for nodes that are desirable

            # Calculate the new cumulative score for the neighboring node
            new_score = current_score + weight

            # Insert the neighbor in the open set with the new score
            heapq.heappush(open_set, (new_score, node, path + [current]))

    # If no path is found, we return None and infinite cost
    return None, float('inf')

# Example usage of the A* search function with specified start and end nodes
start_node = 'A'
end_node = 'F'
path, _ = a_star_search(graph, start_node, end_node)

# Visualization setup using networkx and matplotlib 
G = nx.DiGraph()  # Create a new directed graph
# Populate the graph with nodes and weighted edges from our adjacency list representation
for node, edges in graph.items():
    for edge in edges:
        G.add_edge(node, edge[0], weight=edge[1])

# Draw the full graph using matplotlib
plt.figure(figsize=(10, 5))  # Define the figure size
# Draw the nodes, edges, and labels using previously defined positions and styles
nx.draw(G, with_labels=True, pos=positions, node_color='skyblue', node_size=1500, 
        edge_color='grey', width=2, font_size=15, font_weight='bold')

# If a path was found, highlight it in the visualization
if path:
    # Calculate the edges of the path (sequence of nodes)
    path_edges = list(zip(path, path[1:]))
    # Highlight nodes of the path
    nx.draw_networkx_nodes(G, pos=positions, nodelist=path, node_color='lightgreen')
    # Highlight edges of the path
    nx.draw_networkx_edges(G, pos=positions, edgelist=path_edges, edge_color='green', width=3)

# Display the graph with highlighted path (if found)
plt.show()

# Print the resulting path from start to end node
print("Found path:")
print(" -> ".join(path))


'''
Explanation

Here's what happens step by step:

1. Initialization: The priority queue is initialized with a single entry for the starting node, with a path cost of 0 and an empty path, since we haven't moved anywhere yet.

2. Expansion: In each iteration, A* selects and removes the node with the lowest "f score" from the priority queue. The 'f score' is a combination of the 'g score' (the cost from the start node to the current node) and the 'h score' (the estimated cost from the current node to the goal, as determined by the heuristic function).

3. Update: For the selected node, A* examines each of its neighbors. If a neighbor can be reached by a lower-cost path than previously known, or if it has not been seen before, a new entry for that neighbor is created or updated with the total cost (from the start node through the current node to the neighbor) and the current path leading to it. These neighbor entries are added to the priority queue.

4. Loop: Steps 2 and 3 are repeated, with nodes being pulled from the priority queue, expanded, and their neighbors potentially being added to the queue, until the goal node is reached or the priority queue is emptied (in which case there is no path).

A* works best when we have a good predefined hueristic (which we have in the form of weights for avoiding and including nodes )

'''