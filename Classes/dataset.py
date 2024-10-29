import torch
import numpy as np
import scipy
from tqdm import tqdm
from copy import copy
from torch_geometric.data import InMemoryDataset
from collections import OrderedDict
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data
from datetime import datetime
import os
import math

class GraphDataset(InMemoryDataset):
    def __init__(self, root, train_params = None, dataset_params = None, model_params = None, transform=None, pre_transform=None, dataset_instance = None):
        
        self.train_params = train_params
        self.dataset_params = dataset_params
        self.model_params = model_params
        self.dataset_instance = dataset_instance

        for key, value in train_params.items():
            setattr(self, key, value)
        for key, value in dataset_params.items():
            setattr(self, key, value)
        for key, value in model_params.items():
            setattr(self, key, value)
        self.dir_dataset = train_params['dir_dataset']
        self.name_processed_dataset = train_params['dataset_name']

        if self.reprocess_graph_dataset:
            try:
                os.remove(self.dir_dataset + self.name_processed_dataset + '.dataset') 
            except:
                pass
                
        super(GraphDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []
    # Where to put output and its name
    @property
    def processed_file_names(self):
        if self.name_processed_dataset is None:
            return [self.dir_dataset + 'GraphDataset.dataset']
        else:
            return [self.dir_dataset + f'{self.name_processed_dataset}.dataset']

    def download(self):
        pass

    def min_max_scaler(self, X, X_min, X_max, num_features, feature_min = 0, feature_max = 1):
        if num_features == 1:
            X_std = (X - X_min) / (X_max - X_min)
            X_scaled = X_std * (feature_max - feature_min) + feature_min
        else:
            X_std = (X[:num_features] - X_min[:num_features]) / (X_max[:num_features] - X_min[:num_features])
            X_scaled = X_std[:num_features] * (feature_max - feature_min) + feature_min
            X_scaled = np.nan_to_num(X_scaled)
        return X_scaled

    def process(self):

        data_list = []

        for instance in tqdm(range(self.instances)):

            if self.dataset_instance is None:
                self.dataset_params['seed'] = int(datetime.now().timestamp()*10000)%(2**32)

                # random velocities/accelerations
                if self.model_params['number_state_variables'] >= 4:
                    self.dataset_params['mean_velocity_x'] = np.random.uniform(self.dataset_params['limit_vx'][0],self.dataset_params['limit_vx'][1],(1,))[0]
                    self.dataset_params['mean_velocity_y'] = np.random.uniform(self.dataset_params['limit_vy'][0],self.dataset_params['limit_vy'][1],(1,))[0]
                    if self.model_params['number_state_variables'] >= 6:
                        self.dataset_params['mean_acceleration_x'] = np.random.uniform(self.dataset_params['limit_ax'][0],self.dataset_params['limit_ax'][1],(1,))[0] 
                        self.dataset_params['mean_acceleration_y'] = np.random.uniform(self.dataset_params['limit_ay'][0],self.dataset_params['limit_ay'][1],(1,))[0] 

                # random number of agents (NO)
                # self.dataset_params['num_agents'] = int(np.random.uniform(3,self.dataset_params['limit_num_agents'],(1,))[0])
                dataset_instance = ArtificialDataset(self.dataset_params)
                dataset_instance.compute_whole_dataset()
            else: 
                dataset_instance = self.dataset_instance
            
            #####
            mean_x, std_x = dataset_instance.compute_mean_std_x()
            std_x[std_x==0] = 1 # avoid dividing by zero
            min_x, max_x = dataset_instance.compute_min_max_x()
            limit_features_x_min = np.array([self.limit_x[0], self.limit_y[0], self.limit_vx[0], self.limit_vy[0], self.limit_ax[0], self.limit_ay[0]])[:self.number_state_variables]
            limit_features_x_max = np.array([self.limit_x[1], self.limit_y[1], self.limit_vx[1], self.limit_vy[1], self.limit_ax[1], self.limit_ay[1]])[:self.number_state_variables]
            #####
            mean_gnss, std_gnss = dataset_instance.compute_mean_std_gnss()
            std_gnss[std_gnss==0] = 1 # avoid dividing by zero
            min_gnss, max_gnss = dataset_instance.compute_min_max_gnss()
            limit_features_gnss_min = np.array([self.limit_x[0], self.limit_y[0], self.limit_vx[0], self.limit_vy[0], self.limit_ax[0], self.limit_ay[0]])[:self.NUM_MEAS_GNSS]
            limit_features_gnss_max = np.array([self.limit_x[1], self.limit_y[1], self.limit_vx[1], self.limit_vy[1], self.limit_ax[1], self.limit_ay[1]])[:self.NUM_MEAS_GNSS]
            #####
            mean_z, std_z = dataset_instance.compute_mean_std_meas()
            std_z[std_z==0] = 1 # avoid dividing by zero
            mean_z = mean_z.tolist()[0]
            std_z = std_z.tolist()[0]
            min_z, max_z = dataset_instance.compute_min_max_meas()
            min_z = min_z.tolist()[0]
            max_z = max_z.tolist()[0]
            limit_features_z_min = 0
            limit_features_z_max = np.sqrt((self.limit_x[1] - self.limit_x[0])**2 + (self.limit_y[1] - self.limit_y[0])**2).tolist()


            mean_std_min_max_variables = {'mean_x':np.float32(mean_x),
                                  'std_x':np.float32(std_x),
                                  'min_x':np.float32(min_x),
                                  'max_x':np.float32(max_x),
                                  'limit_features_x_min':np.float32(limit_features_x_min),
                                  'limit_features_x_max':np.float32(limit_features_x_max),
                                  'mean_gnss':np.float32(mean_gnss),
                                  'std_gnss':np.float32(std_gnss),
                                  'min_gnss':np.float32(min_gnss),
                                  'max_gnss':np.float32(max_gnss),
                                  'limit_features_gnss_min':np.float32(limit_features_gnss_min),
                                  'limit_features_gnss_max':np.float32(limit_features_gnss_max),
                                  'mean_z':np.float32(mean_z),
                                  'std_z':np.float32(std_z),
                                  'min_z':np.float32(min_z),
                                  'max_z':np.float32(max_z),
                                  'limit_features_z_min':np.float32(limit_features_z_min),
                                  'limit_features_z_max':np.float32(limit_features_z_max),
                                  }


            for instant in (range(self.number_instants-1)):
            
                edge_index = []                 # indexes of edges    (#2 x #2EDGES)
                # edge_labels = []              # labels of edges
                edge_attr = []                  # attributes of edges (#2EDGES x #EDGEFEATURES)
                edge_real_distance = []         # real distance between the agents of the edge
                source_nodes = []
                dest_nodes = []
                x = {}                          # node latent features, prior beliefs (#NODES x #NODEFEATURES)
                x_not_normalized = {} 
                node_attr = {}                  # GNSS measurements
                node_attr_not_normalized = {}
                node_id = {}                    # id at the node
                feature_name = {}               # name of the feature
                track_id = {}                   # identificative of the tracking
                # node_labels = {}
                node_next_x = {}                # GT

                node = 0

                for agent in range(self.num_agents):

                    feature = 0

                    for agent2 in range(agent+1, self.num_agents):

                        feature2 = 0
                            
                        if dataset_instance.connectivity_matrix[instant][agent][agent2]:  # m
                        
                            add_source_node = 1
                            
                            # x[agent] = [(dataset_instance.x[agent][instant][:self.number_state_variables]-mean_x[:self.number_state_variables])/std_x[:self.number_state_variables], agent]
                            # x[agent2] = [(dataset_instance.x[agent2][instant][:self.number_state_variables]-mean_x[:self.number_state_variables])/std_x[:self.number_state_variables], agent2]
                            x[agent] = [self.min_max_scaler(dataset_instance.x[agent][instant][:self.number_state_variables], limit_features_x_min, limit_features_x_max, self.number_state_variables), agent]
                            x[agent2] = [self.min_max_scaler(dataset_instance.x[agent2][instant][:self.number_state_variables], limit_features_x_min, limit_features_x_max, self.number_state_variables), agent2]

                            x_not_normalized[agent] = [dataset_instance.x[agent][instant][:self.number_state_variables], agent]
                            x_not_normalized[agent2] = [dataset_instance.x[agent2][instant][:self.number_state_variables], agent2]

                            # node_attr[agent] = [(dataset_instance.gnss[agent][instant][:self.NUM_MEAS_GNSS]-mean_gnss[:self.NUM_MEAS_GNSS])/std_gnss[:self.NUM_MEAS_GNSS], agent]
                            # node_attr[agent2] = [(dataset_instance.gnss[agent2][instant][:self.NUM_MEAS_GNSS]-mean_gnss[:self.NUM_MEAS_GNSS])/std_gnss[:self.NUM_MEAS_GNSS], agent2]
                            node_attr[agent] = [self.min_max_scaler(dataset_instance.gnss[agent][instant][:self.NUM_MEAS_GNSS], limit_features_gnss_min, limit_features_gnss_max, self.NUM_MEAS_GNSS), agent]
                            node_attr[agent2] = [self.min_max_scaler(dataset_instance.gnss[agent2][instant][:self.NUM_MEAS_GNSS], limit_features_gnss_min, limit_features_gnss_max, self.NUM_MEAS_GNSS), agent2]

                            node_attr_not_normalized[agent] = [(dataset_instance.gnss[agent][instant][:self.NUM_MEAS_GNSS]), agent]
                            node_attr_not_normalized[agent2] = [(dataset_instance.gnss[agent2][instant][:self.NUM_MEAS_GNSS]), agent2]

                            node_next_x[agent] = [dataset_instance.x[agent][instant+1][:self.number_state_variables], agent]
                            node_next_x[agent2] = [dataset_instance.x[agent2][instant+1][:self.number_state_variables], agent2]

                            # node_labels[agent] = agent
                            # node_labels[agent2] = agent2

                            node_id[agent] = agent
                            node_id[agent2] = agent2 
                            
                            # feature_name[node] = name
                            # feature_name[node2] = name2   
                            
                            # track_id[node] = name
                            # track_id[node2] = name2      
                            
                            source_nodes.append(agent)

                            dest_nodes.append(agent2)

                            # edge_labels.append(1 if (name == name2) and (('FP' not in name) and ('FP' not in name2)) else 0)

                            # edge_attr.append([(dataset_instance.measurements[instant][agent][agent2]-mean_z)/std_z])
                            edge_attr.append([self.min_max_scaler(dataset_instance.measurements[instant][agent][agent2], limit_features_z_min, limit_features_z_max, 1)])

                            edge_real_distance.append([dataset_instance.mutual_distances[instant][agent][agent2]])

                        feature2 += 1                            
                        
                    node += 1
                    
                    feature += 1

                # Node features
                x = OrderedDict(sorted(x.items()))
                # Node agent (correspondence node_ith - agent, order of nodes is the same of x)
                node_agent = torch.tensor([v[1] for k, v in x.items()], dtype=torch.float)  
                x = torch.tensor([v[0] for k, v in x.items()], dtype=torch.float)
                # x not normalized
                x_not_normalized = OrderedDict(sorted(x_not_normalized.items()))
                x_not_normalized = torch.tensor([v[0] for k, v in x_not_normalized.items()], dtype=torch.float)
                # gnss meas
                node_attr = OrderedDict(sorted(node_attr.items()))
                node_attr = torch.tensor([v[0] for k, v in node_attr.items()], dtype=torch.float)
                # gnss meas not normalized
                node_attr_not_normalized = OrderedDict(sorted(node_attr_not_normalized.items()))
                node_attr_not_normalized = torch.tensor([v[0] for k, v in node_attr_not_normalized.items()], dtype=torch.float)
                # Agent next position
                node_next_x = OrderedDict(sorted(node_next_x.items()))
                node_next_x = torch.tensor([v[0] for k, v in node_next_x.items()], dtype=torch.float)
                # Node labels
                # node_labels = OrderedDict(sorted(node_labels.items()))
                # node_labels = torch.tensor([v for k, v in node_labels.items()], dtype=torch.long)
                # Edge indexes
                encoder = LabelEncoder().fit(source_nodes + dest_nodes)
                edge_index = torch.tensor([encoder.transform(source_nodes).tolist(), encoder.transform(dest_nodes).tolist()], dtype=torch.long)
                # Edge features
                edge_attr = torch.tensor(edge_attr, dtype=torch.float)
                # Edge real distance
                edge_real_distance = torch.tensor(edge_real_distance, dtype=torch.float)
                # Edge labels
                # edge_labels = torch.tensor(edge_labels, dtype=torch.long)
                # Time instant
                time_instant = torch.tensor(instant, dtype=torch.long)
                # Id of each node
                node_id = torch.tensor(encoder.transform([v for k, v in node_id.items()]), dtype=torch.float)
                # Name of each node
                # feature_name = OrderedDict(sorted(feature_name.items()))
                # feature_name = tuple([v for k, v in feature_name.items()])
                # Id for tracking data association for each node
                # track_id = OrderedDict(sorted(track_id.items()))
                # track_id = tuple([v for k, v in track_id.items()])

                data = Data (x = x,
                            edge_attr = torch.cat((edge_attr, edge_attr), dim = 0),
                            edge_index = torch.cat((edge_index, torch.stack((edge_index[1], edge_index[0]))), dim=1))
                # data.node_labels = node_labels
                # data.edge_labels = torch.cat((edge_labels, edge_labels), dim = 0)
                data.x_not_normalized = x_not_normalized
                data.node_attr = node_attr
                data.node_attr_not_normalized = node_attr_not_normalized
                data.node_next_x = node_next_x
                data.edge_real_distance = edge_real_distance
                data.time_instant = time_instant
                data.node_agent = node_agent
                data.node_id = node_id
                # data.feature_name = feature_name
                # data.track_id = track_id
                data.mean_std_min_max_variables = mean_std_min_max_variables

                data_list.append(data)
            
        print(len(data_list))
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class ArtificialDataset():
    def __init__(self, params):

        self.params = params

        self.load_dataset_bool = params['load_dataset']
        self.dataset_file = params['dataset_file']

        # Reproducibility
        # self.deterministic = params['deterministic']                   # 0 do not use noises
        self.seed = params['seed']
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.limit_behavior = params['limit_behavior']                  # 'reflection', 'continue', 'none'
        self.setting_trajectories = params['setting_trajectories']      # 'star', 'not_defined'
        self.limit_x = params['limit_x']                                # [-100,100] [m]
        self.limit_y = params['limit_y']                                # [-100,100] [m]
        self.limit_vx = params['limit_vx']                              # [-10,10] [m/s]
        self.limit_vy = params['limit_vy']                              # [-10,10] [m/s]
        self.limit_ax = params['limit_ax']                              # [-10,10] [m/s^2]
        self.limit_ay = params['limit_ay']                              # [-10,10] [m/s^2]
        self.num_agents = params['num_agents']                          # 10
        self.number_instants = params['number_instants']                # 100

        # physical parameters
        self.T_between_timestep = params['T_between_timestep']           # 1 [s]
        self.comm_distance = params['comm_distance']                    # 100 [m]

        self.noise_type = params['noise_type']                          # 'Gaussian', 'Laplacian'
        # self.motion_model = params['motion_model']                    # 'M1', 'M2', 'M3', 'M4'
        self.std_noise_position = params['std_noise_position']          # [m]
        self.mean_velocity_x = params['mean_velocity_x']                # [m/s]
        self.mean_velocity_y = params['mean_velocity_y']                # [m/s]
        self.std_noise_velocity = params['std_noise_velocity']          # [m/s]
        self.mean_acceleration_x = params['mean_acceleration_x']        # [m/s^2]
        self.mean_acceleration_y = params['mean_acceleration_y']        # [m/s^2]
        self.std_noise_acceleration = params['std_noise_acceleration']  # [m/s^2] 

        self.std_noise_measurements = params['std_noise_measurements']  # [m] 
        self.std_noise_gnss_position = params['std_noise_gnss_position']# [m] 
        self.std_noise_gnss_velocity = params['std_noise_gnss_velocity']# [m/s] 
        self.std_noise_gnss_acceleration= params['std_noise_gnss_acceleration']# [m/s^2] 

        # timestep
        self.n = 0

        # noise
        if self.noise_type == 'Gaussian':
            self.noise_class = np.random.normal
        elif self.noise_type == 'Laplacian':
            self.noise_class = np.random.laplace
        else:
            self.noise_class = np.random.normal


        if self.load_dataset_bool:
            self.load_dataset()
        else:
            # compute positions, velocities and accelerations of each agent
            self.define_initial_x()
            try:
                self.mutual_distances = [scipy.spatial.distance.cdist(np.array([self.positions[agent] for agent in range(self.num_agents)]).squeeze(),np.array([self.positions[agent] for agent in range(self.num_agents)]).squeeze()).tolist()]
            except:
                self.mutual_distances = [0]
            
            self.connectivity_matrix = [(np.array(self.mutual_distances)<self.comm_distance).squeeze()*1 - np.eye(self.params['num_agents'])]
            self.x = {agent:[[self.positions[agent][self.n][0], 
                            self.positions[agent][self.n][1], 
                            self.velocities[agent][self.n][0], 
                            self.velocities[agent][self.n][1], 
                            self.accelerations[agent][self.n][0], 
                            self.accelerations[agent][self.n][1]]] for agent in range(self.num_agents)}
            
            self.measurements = [self.mutual_distances[0]+self.noise_class(0, self.std_noise_measurements, (self.num_agents,self.num_agents))]
            self.gnss = {agent:[[self.positions[agent][0][0] + self.noise_class(0, self.std_noise_gnss_position, (1,)).item(), 
                                 self.positions[agent][0][1] + self.noise_class(0, self.std_noise_gnss_position, (1,)).item(), 
                                 self.velocities[agent][0][0] + self.noise_class(0, self.std_noise_gnss_velocity, (1,)).item(), 
                                 self.velocities[agent][0][1] + self.noise_class(0, self.std_noise_gnss_velocity, (1,)).item(), 
                                 self.accelerations[agent][0][0] + self.noise_class(0, self.std_noise_gnss_acceleration, (1,)).item(), 
                                 self.accelerations[agent][0][1] + self.noise_class(0, self.std_noise_gnss_acceleration, (1,)).item()]] for agent in range(self.num_agents)}
        
        # x_n = F x_n-1 + W w_n-1
        # x: pos, vel, acc. w: pos_noise, vel_noise, acc_noise
        self.F = np.array([[1, 0, self.T_between_timestep, 0                      , (self.T_between_timestep**2)/2, 0                             ],
                           [0, 1, 0                      , self.T_between_timestep, 0                             , (self.T_between_timestep**2)/2],
                           [0, 0, 1                      , 0                      , self.T_between_timestep       , 0                             ],
                           [0, 0, 0                      , 1                      , 0                             , self.T_between_timestep       ],
                           [0, 0, 0                      , 0                      , 1                             , 0                             ],
                           [0, 0, 0                      , 0                      , 0                             , 1                             ]])
        self.W = copy(self.F)

        
    def compute_next_step(self): 

        new_positions_agents_list = []
        for agent in range(self.num_agents):

            noise_position, noise_velocities, noise_accelerations = self.add_noise_deterministic_components(agent)
            new_x = self.F@np.array(self.x[agent][-1]) + self.W@np.concatenate((noise_position, noise_velocities, noise_accelerations))
            new_x_list = new_x.tolist()

            # check if agent is outside the area
            if self.limit_behavior == 'reflection':
                if new_x_list[0] < self.limit_x[0]:
                    while new_x_list[0] < self.limit_x[0]:
                        new_x_list[0] = self.limit_x[0] + abs(self.limit_x[0]-new_x_list[0])
                    new_x_list[2] = - new_x_list[2]
                    new_x_list[4] = - new_x_list[4]
                elif new_x_list[0] > self.limit_x[1]:
                    while new_x_list[0] > self.limit_x[1]:
                        new_x_list[0] = self.limit_x[1] - abs(self.limit_x[1]-new_x_list[0])
                    new_x_list[2] = - new_x_list[2]
                    new_x_list[4] = - new_x_list[4]
                if new_x_list[1] < self.limit_y[0]:
                    while new_x_list[1] < self.limit_y[0]:
                        new_x_list[1] = self.limit_y[0] + abs(self.limit_y[0]-new_x_list[1])
                    new_x_list[3] = - new_x_list[3]
                    new_x_list[5] = - new_x_list[5]
                elif new_x_list[1] > self.limit_y[1]:
                    while new_x_list[1] > self.limit_y[1]:
                        new_x_list[1] = self.limit_y[1] - abs(self.limit_y[1]-new_x_list[1])
                    new_x_list[3] = - new_x_list[3]
                    new_x_list[5] = - new_x_list[5]
            elif self.limit_behavior == 'continue':
                if new_x_list[0] < self.limit_x[0]:
                    while new_x_list[0] < self.limit_x[0]:
                        new_x_list[0] = self.limit_x[1] - abs(self.limit_x[0]-new_x_list[0])
                if new_x_list[0] > self.limit_x[1]:
                    while new_x_list[0] > self.limit_x[1]:
                        new_x_list[0] = self.limit_x[0] + abs(self.limit_x[1]-new_x_list[0])
                if new_x_list[1] < self.limit_y[0]:
                    while new_x_list[1] < self.limit_y[0]:
                        new_x_list[1] = self.limit_y[1] - abs(self.limit_y[0]-new_x_list[1])
                if new_x_list[1] > self.limit_y[1]:
                    while new_x_list[1] > self.limit_y[1]:
                        new_x_list[1] = self.limit_y[0] + abs(self.limit_y[1]-new_x_list[1])

            # limit velocity
            if new_x_list[2] < self.limit_vx[0]:
                new_x_list[2] = self.limit_vx[0]
            if new_x_list[2] > self.limit_vx[1]:
                new_x_list[2] = self.limit_vx[1]
        
            if new_x_list[3] < self.limit_vy[0]:
                new_x_list[3] = self.limit_vy[0]
            if new_x_list[3] > self.limit_vy[1]:
                new_x_list[3] = self.limit_vy[1]

            # limit acceleration
            if new_x_list[4] < self.limit_ax[0]:
                new_x_list[4] = self.limit_ax[0]
            if new_x_list[4] > self.limit_ax[1]:
                new_x_list[4] = self.limit_ax[1]
        
            if new_x_list[4] < self.limit_ay[0]:
                new_x_list[4] = self.limit_ay[0]
            if new_x_list[4] > self.limit_ay[1]:
                new_x_list[4] = self.limit_ay[1]

            new_positions_agents_list.append(new_x_list[0:2])
            self.x[agent].append(new_x_list)
            self.positions[agent].append(new_x_list[0:2])
            self.velocities[agent].append(new_x_list[2:4])
            self.accelerations[agent].append(new_x_list[4:6])

            self.gnss[agent].append([self.positions[agent][-1][0] + self.noise_class(0, self.std_noise_gnss_position, (1,)).item(), 
                                    self.positions[agent][-1][1] + self.noise_class(0, self.std_noise_gnss_position, (1,)).item(), 
                                    self.velocities[agent][-1][0] + self.noise_class(0, self.std_noise_gnss_velocity, (1,)).item(), 
                                    self.velocities[agent][-1][1] + self.noise_class(0, self.std_noise_gnss_velocity, (1,)).item(), 
                                    self.accelerations[agent][-1][0] + self.noise_class(0, self.std_noise_gnss_acceleration, (1,)).item(), 
                                    self.accelerations[agent][-1][1] + self.noise_class(0, self.std_noise_gnss_acceleration, (1,)).item()])

        new_positions_agents = np.array(new_positions_agents_list)

        # compute real distances
        new_mutual_distances = scipy.spatial.distance.cdist(new_positions_agents,new_positions_agents).tolist()
        try:
            self.mutual_distances.append(new_mutual_distances)
        except:
            self.mutual_distances.append(0)

        # compute connectivity matrix
        self.connectivity_matrix.append((np.array(new_mutual_distances)<self.comm_distance).squeeze()*1 - np.eye(self.params['num_agents']))

        # compute new measurements
        self.measurements.append(new_mutual_distances+self.noise_class(0, self.std_noise_measurements, (self.num_agents,self.num_agents)))

        # Update timestep
        self.n = self.n + 1

    def compute_whole_dataset(self): 
        for n in range(self.n, self.number_instants):
            self.compute_next_step()

    def define_initial_x(self):

        # random initial position, velocity defined, acceleration defined 
        if self.setting_trajectories == 'not_defined':
            self.positions = {agent:[[np.random.uniform(self.limit_x[0],self.limit_x[1]), np.random.uniform(self.limit_y[0],self.limit_y[1])]] for agent in range(self.num_agents)}
            self.velocities = {agent:[[self.mean_velocity_x, self.mean_velocity_y]] for agent in range(self.num_agents)}
            self.accelerations = {agent:[[self.mean_acceleration_x, self.mean_acceleration_y]] for agent in range(self.num_agents)}
        elif self.setting_trajectories == 'star' or self.setting_trajectories == 'spiral':
            angle_directions = np.arange(0,360, 360/self.num_agents) * math.pi/180
            self.positions = {agent:[[np.random.uniform(0,0), np.random.uniform(0,0)]] for agent in range(self.num_agents)}
            self.velocities = {agent:[[abs(self.mean_velocity_x)*np.cos(angle_directions[agent]), abs(self.mean_velocity_x)*np.sin(angle_directions[agent])]] for agent in range(self.num_agents)}
            self.accelerations = {agent:[[abs(self.mean_acceleration_x)*np.cos(angle_directions[agent]), abs(self.mean_acceleration_x)*np.sin(angle_directions[agent])]] for agent in range(self.num_agents)}      

    def add_noise_deterministic_components(self, agent):
        # random component
        noise_position = self.noise_class(0, self.std_noise_position, 2) 
        noise_velocities = self.noise_class(0, self.std_noise_velocity, 2) 
        noise_accelerations = self.noise_class(0, self.std_noise_acceleration, 2) 

        # deterministic component
        if self.setting_trajectories == 'spiral' and self.n >3: # self.n >20:

            # perfect spiral
            # angle = np.arctan2(self.positions[agent][-1][1], self.positions[agent][-1][0])
            # angle_deg = angle*180/np.pi
            # distance_from_center = np.linalg.norm([self.positions[agent][-1][0],self.positions[agent][-1][1]])
            # mod_vel = np.max((np.linalg.norm([self.mean_velocity_x, self.mean_velocity_y]), 1))
            # orthogonal_vel = np.array([-np.sin(angle), np.cos(angle)])* mod_vel #*(1-distance_from_center/self.limit_x[0])
            # # remove velocity
            # orthogonal_vel -= np.array([self.velocities[agent][-1][0],self.velocities[agent][-1][1]])
            # noise_velocities += orthogonal_vel

            # golder spiral
            angle = np.arctan2(self.positions[agent][-1][1], self.positions[agent][-1][0])
            angle_deg = angle*180/np.pi
            distance_from_center = np.linalg.norm([self.positions[agent][-1][0],self.positions[agent][-1][1]])
            mod_vel = np.max((np.linalg.norm([self.mean_velocity_x, self.mean_velocity_y]), 1))
            orthogonal_vel = np.array([-np.sin(angle), np.cos(angle)])* mod_vel #*(1-distance_from_center/self.limit_x[0])

            if self.n + agent > 10 + 16:
                # remove velocity
                orthogonal_vel -= np.array([self.velocities[agent][-1][0],self.velocities[agent][-1][1]])
            noise_velocities += orthogonal_vel

        return noise_position, noise_velocities, noise_accelerations

    def compute_mean_std_x(self):
        dataset = np.array([self.x[agent] for agent in range(self.num_agents)]).reshape([-1, 6])
        return np.mean(dataset, 0), np.std(dataset, 0)
    def compute_min_max_x(self):
        dataset = np.array([self.x[agent] for agent in range(self.num_agents)]).reshape([-1, 6])
        return np.min(dataset, 0), np.max(dataset, 0)
    
    def compute_mean_std_meas(self):
        # remove diagonal values
        mask = np.ones(self.measurements[0].shape, dtype=bool)
        np.fill_diagonal(mask, False)
        dataset = np.array([self.measurements[n][mask].flatten() for n in range(self.number_instants)]).reshape([-1, 1])
        return np.mean(dataset, 0), np.std(dataset, 0)
    def compute_min_max_meas(self):
        # remove diagonal values
        mask = np.ones(self.measurements[0].shape, dtype=bool)
        np.fill_diagonal(mask, False)
        dataset = np.array([self.measurements[n][mask].flatten() for n in range(self.number_instants)]).reshape([-1, 1])
        return np.min(dataset, 0), np.max(dataset, 0)
    
    def compute_mean_std_gnss(self):
        dataset = np.array([self.gnss[agent] for agent in range(self.num_agents)]).reshape([-1, 6])
        return np.mean(dataset, 0), np.std(dataset, 0)
    def compute_min_max_gnss(self):
        dataset = np.array([self.gnss[agent] for agent in range(self.num_agents)]).reshape([-1, 6])
        return np.min(dataset, 0), np.max(dataset, 0)
    
    def save_dataset(self):
        np.save(self.dataset_file + '.npy', {'x':self.x, 
                                    'positions':self.positions,
                                    'velocities':self.velocities,
                                    'accelerations':self.accelerations,
                                    'mutual_distances':self.mutual_distances,
                                    'connectivity_matrix':self.connectivity_matrix,
                                    'measurements':self.measurements, 
                                    'gnss': self.gnss,
                                    'time_n': self.n,
                                    'params':self.params}, 
                                    allow_pickle = True)

    def load_dataset(self):
        ris = np.load(self.dataset_file + '.npy', allow_pickle = True).tolist()
        self.x = ris['x']
        self.positions = ris['positions']
        self.velocities = ris['velocities']
        self.accelerations = ris['accelerations']
        self.mutual_distances = ris['mutual_distances']
        self.connectivity_matrix = ris['connectivity_matrix']
        self.measurements = ris['measurements']
        self.gnss = ris['gnss']
        self.n = ris['time_n']
        self.params = ris['params']




















































































































































































































































































































































































































































































































































