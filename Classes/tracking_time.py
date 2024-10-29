import numpy as np
from torch_geometric.utils import to_networkx
from sklearn.preprocessing import LabelEncoder
import networkx as nx
import torch
import re

def new_target_pos(sample, labels):
    # Given the association, compute the targets new position    
    
    edges = sample.edge_index.t()
        
    # Time instant
    instant = sample.time_instant
    # Recover correspondence node-vehicle
    node_vehicle = sample.node_vehicle
    # Encode node_vehicle to be used as color mapping
    encoder = LabelEncoder().fit(node_vehicle)
    node_vehicle = encoder.transform(node_vehicle)
    # Recover number of nodes
    number_nodes = sample.num_nodes
    # Node attributes
    x = sample.x.reshape(-1, 3, 8).numpy()
    centroids = np.mean(x, axis=2)
    
    # Recover name for the track
    track_id = sample.track_id
    # Recover name of features
    feature_name = sample.feature_name
    
        
    edgelist_ones = edges[np.where(labels==1)[0]].numpy() # does not preserve order
    edgelist_zeros = edges[np.where(labels==0)[0]].numpy() 
    
      
        
    # Create graph
    G = to_networkx(sample, node_attrs=None, edge_attrs=None, to_undirected=False, remove_self_loops=False) 
    # Consider only edges classified as 1
    G.remove_edges_from([tuple(x) for x in edgelist_zeros]) 
    # Compute connected components
    G_undirected = G.to_undirected(reciprocal=True)
    connected_components = nx.connected_components(G_undirected)
    connected_components = sorted(connected_components)
    G_components = [G_undirected.subgraph(c).copy() for c in connected_components]

        
    
    node_vehicle = encoder.inverse_transform(node_vehicle).astype(int)
    
    # Recover correspondence node-vehicle and compute graph vehicle-features
    G_vehicle_features = nx.Graph()
    connected_components_vehicles = []
    # Which vehicles are in each connected component
    connected_components_unique_vehicles = []
    # new name of each target
    name_features = []
    # name for the tracking
    track_id_new = []
    # mean position of target
    x_out = {}
    i = 0
    for el in connected_components:
        el = sorted(el)
        connected_components_vehicles.append(np.array([node_vehicle[int(node)] for node in el])) 
        connected_components_unique_vehicles.append(np.unique(np.array([node_vehicle[int(node)] for node in el])))
        track_id_new.append(f'ped_pred_{i}_t_{instant.tolist()[0]}')
        x_out[track_id_new[i]] = np.mean([x[int(node)] for node in el], 0)
        
        # Choose as name to track, the majority vote name
        names_in_connected_component = [feature_name[node] for node in el]
        name_features.append(max(set(names_in_connected_component), key = names_in_connected_component.count))
        
        i+=1
    return x_out, track_id_new, name_features


# Add fake measurements to the sample
def add_fictitious_vehicle (sample, x_new, track_id_new, feature_name_new, colors, vehicles_pos_xy = None):
    # sample: sample of next instant in which we want to add the fake measurements
    # x_new: shape (old_connected_components/pedestrians, 3, 8)
    # track_id_new: list of name of old measurements ["ped_pred_0_t_784", ... ]
    # feature_name_new: list of names of features ['ped_3','ped_38', ...] computed by majority voting
    # vehicles_pos_xy: Add the fakle vehicle position to the vehicles_pos_xy
    
    
    x = sample.x
    edge_index = sample.edge_index
    edge_attr = sample.edge_attr
    edge_labels = sample.edge_labels
    instant = sample.time_instant.tolist()[0]
    node_vehicle = sample.node_vehicle
    node_id = sample.node_id # id (int) of each node according to the order of x
    feature_name = sample.feature_name
    track_id = sample.track_id
    
    feature = len(np.unique(node_id))
    
    for node in range(len(x_new)):
        
        box = x_new[node]
        
        track_id_box = track_id_new[node]
        feature_name_box = feature_name_new[node]
        
        centroid = np.mean(box, 1)

        add_source_node = 0
        
        for node2 in range(len(node_id)):
            
            box2 = x[node2].view(3, 8).numpy()

            centroid2 = np.mean(box2, 1)

            if np.linalg.norm(centroid-centroid2) < 10:  # m
                
                # If it is the first time I add the source node
                if not add_source_node:
                    
                    # Add node attribute
                    x = torch.cat([x,  torch.tensor(np.expand_dims(box.flatten(), axis=0), dtype=torch.float)], dim=0)
                    
                    node_id = torch.cat([node_id, torch.tensor(feature).unsqueeze(dim=0)])
                    
                    # The fake vehicle has id 1.000 + instant
                    instant_fake_vehicle = re.findall('\d+', track_id_box)[1]
                    node_vehicle = torch.cat([node_vehicle, torch.tensor(1000 + int(instant_fake_vehicle)).unsqueeze(dim=0)])
                    
                    # Add fake vehicle position
                    if vehicles_pos_xy is not None:
                        vehicles_pos_xy[1000 + int(instant_fake_vehicle)][int(instant)] = np.array([100+10*0, 210])
                    
                    # Add fake vehicle color
                    colors[1000 + int(instant_fake_vehicle)] = 'orange'
                    
                    #feature_name.append(track_id_box)
                    feature_name = (*feature_name, feature_name_box)
                    
                    #track_id.append(track_id_box)
                    track_id = (*track_id, track_id_box)
     
                edge_index = torch.cat([edge_index, torch.tensor([feature, node_id[node2]]).unsqueeze(dim=1)], dim=1)
                
                attr =   [box[0,0]-box2[0,0],
                          box[1,0]-box2[1,0], 
                          box[2,0]-box2[2,0],
                          box[0,7]-box2[0,7],
                          box[1,7]-box2[1,7],
                          box[2,7]-box2[2,7]]
                edge_attr = torch.cat([edge_attr, torch.tensor(attr).unsqueeze(dim=0)], dim=0)
                
                edge_labels = torch.cat([edge_labels, torch.tensor(1 if feature_name[node2] == feature_name_new[node] else 0).unsqueeze(dim=0)])
                    

                add_source_node = 1
                
        if add_source_node:
            
            feature += 1
            
    
    sample.x = x.type(torch.float)
    sample.edge_index =edge_index.type(torch.long)
    sample.edge_attr = edge_attr.type(torch.float)
    sample.edge_labels = edge_labels.type(torch.long)
    sample.node_vehicle = node_vehicle
    sample.node_id = node_id
    sample.feature_name = feature_name
    sample.track_id = track_id
  
    return sample, colors
















































































































































































































































































































































































































































































































































