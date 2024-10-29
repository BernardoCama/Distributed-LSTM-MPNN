import torch
from collections import Counter, defaultdict
import numpy as np
from torch_geometric.utils import to_networkx
from sklearn.preprocessing import LabelEncoder
import networkx as nx

def nested_dict():
    return defaultdict(nested_dict)

def assign_edges(sample, output, thr):
    # Dictionary V1-node1-V2-node2 
    assignment = nested_dict()
    node_vehicle = sample.node_vehicle.numpy().astype(int)
    
    i = 0
    for edge in sample.edge_index.numpy().transpose()[:,:2]:
        s = edge[0]
        d = edge[1]
        assignment[node_vehicle[s]][s][node_vehicle[d]][d] = output[i].tolist()[0]
        i+=1
        
    for v1 in assignment.keys():
        for node1 in assignment[v1].keys():
            for v2 in assignment[v1][node1].keys():
                #assignment[v1][node1][v2].keys(), assignment[v1][node1][v2].values()
                max_key = max(assignment[v1][node1][v2], key=assignment[v1][node1][v2].get)
                if assignment[v1][node1][v2][max_key] > thr:
                    assignment[v1][node1][v2] = dict.fromkeys(assignment[v1][node1][v2].keys(), 0)
                    assignment[v1][node1][v2][max_key] = 1
                else:
                    assignment[v1][node1][v2] = dict.fromkeys(assignment[v1][node1][v2].keys(), 0)

    i = 0
    for edge in sample.edge_index.numpy().transpose()[:,:2]:
        s = edge[0]
        d = edge[1]
        output[i] = assignment[node_vehicle[s]][s][node_vehicle[d]][d]
        i+=1
        
    return assignment,  torch.reshape(output, (-1,))


def apply_heuristics (sample, output, output_fractionary):
    
    try:
        output = output.numpy()
    except:
        pass
    edges = sample.edge_index.t()
    edgelist_ones = edges[np.where(output==1)[0]].numpy()
    edgelist_zeros = edges[np.where(output==0)[0]].numpy()
    edges = edges.numpy()

    # Recover correspondence node-vehicle
    node_vehicle = sample.node_vehicle
    # Encode node_vehicle to be used as color mapping
    encoder = LabelEncoder().fit(node_vehicle)
    node_vehicle = encoder.transform(node_vehicle)
    # Recover number of nodes
    number_nodes = sample.num_nodes    


    # Create graph
    G = to_networkx(sample, node_attrs=None, edge_attrs=None, to_undirected=False, remove_self_loops=False)

    # Consider only edges classified as 1
    G.remove_edges_from(edgelist_zeros)

    # Compute connected components
    G_undirected = G.to_undirected(reciprocal=True)
    connected_components = nx.connected_components(G_undirected)
    connected_components = sorted(connected_components)
    G_components = [G_undirected.subgraph(c).copy() for c in connected_components]

    # Recover correspondence node-vehicle
    connected_components_vehicles = []
    for el in connected_components:
        el = sorted(el)
        # el, np.array([node_vehicle[node] for node in el])
        connected_components_vehicles.append(np.array([node_vehicle[node] for node in el]))

    # Apply euristic
    i = 0
    for el in connected_components_vehicles:
        # If two or more boxes in same vehicle are classified as same box
        duplicates = [item for item, count in Counter(el).items() if count > 1]

        for duplicate in duplicates:

            # el
            # connected_components[i], G_components[i].edges

            index_duplicate = np.where(el==duplicate)[0]

            source = list(sorted(connected_components[i]))[index_duplicate[0]]
            target = list(sorted(connected_components[i]))[index_duplicate[1]]
            
            try:
                path = nx.shortest_path(G_components[i], source=source, target=target)
                path = list(zip(path,(path[1:]+path[:1])))[:-1]

                # Remove for each path connecting two unfeasible nodes
                connected_edges_prediction = []
                indexes = []
                for edge in path:
                    index_in = np.where((edges[:, 0] == edge[0]) & (edges[:, 1] == edge[1]))[0]
                    index_out = np.where((edges[:, 0] == edge[1]) & (edges[:, 1] == edge[0]))[0]
                    indexes.append([index_in, index_out])
                    # output_fractionary[index_in]
                    # output_fractionary[index_out]
                    connected_edges_prediction.append(output_fractionary[index_in])
                # connected_edges_prediction
                # indexes
                index_edges_misclassified = connected_edges_prediction.index(min(connected_edges_prediction))

                # Classify as not associated
                # indexes[index_edges_misclassified]
                output[indexes[index_edges_misclassified][0]] = 0
                output[indexes[index_edges_misclassified][1]] = 0

                G_components[i].remove_edge(path[index_edges_misclassified][0],path[index_edges_misclassified][1])
                
            except:
                pass
        i += 1

























































































































































































































































































































































































































































































































































