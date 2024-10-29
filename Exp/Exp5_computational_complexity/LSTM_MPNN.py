import numpy as np
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")
# TO TEST CODE PERFORMANCES
# Create a profiler object
# profiler = cProfile.Profile()
import torch
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams.update({'font.size': 22})

# Reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# Directories
CLASSES_DIR = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]
DIR_EXPERIMENT = os.path.dirname(os.path.abspath(__file__))
NAME_EXPERIMENT = DIR_EXPERIMENT.split(os.sep)[-1]
cwd = DIR_EXPERIMENT

sys.path.append(os.path.dirname(CLASSES_DIR))

# Classes and Functions
from Classes.dataset import GraphDataset, ArtificialDataset
from Classes.utils import mkdir
from Classes.model import BP, NetBP_LSTM
from Classes.others import create_exp_folder




# Dataset ##############################################################################################################
# import dataset
dir_dataset = cwd + f'/../../../Dataset/{NAME_EXPERIMENT}/'
mkdir(dir_dataset)
dataset_name = 'dataset_test'
dataset_file = dir_dataset + dataset_name

# METHOD USED
# name_method = 'BP_Kalman_no_coop'
# name_method = 'BP_particle_no_coop'
# name_method = 'BP_particle_coop'
name_method = 'LSTM_MPNN'
mean_time = {}

Dimensions = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
for dim_ in Dimensions:

    print(dim_)

    MC_iterations = 5
    time_per_step_MC = []
   
    for mc in range(MC_iterations):

        print(mc)

        dataset_params = {'load_dataset':0,
                        'reprocess_graph_dataset':0,
                        'dataset_file':dataset_file, 
                        'seed':seed,
                        'limit_behavior':'none',              # 'reflection', 'continue', 'none'
                        'setting_trajectories': 'spiral',       # 'star', 'spiral', 'not_defined'
                        'limit_x':[-100, 100],
                        'limit_y':[-100, 100],
                        'limit_vx':[-10, 10],
                        'limit_vy':[-10, 10],
                        'limit_ax':[-10, 10],
                        'limit_ay':[-10, 10],
                        'limit_num_agents': 10,
                        'num_agents': 16, 
                        'number_instants':100,
                        'T_between_timestep':1, 
                        'comm_distance':1000,
                        'noise_type':'Gaussian',
                        'std_noise_position':0,
                        'mean_velocity_x':1,
                        'mean_velocity_y':-1,
                        'std_noise_velocity':0,
                        'mean_acceleration_x':0,
                        'mean_acceleration_y':0,
                        'std_noise_acceleration':0,

                        'std_noise_measurements':5, # inter-distances [m]
                        'std_noise_gnss_position':5,
                        'std_noise_gnss_velocity':1,
                        'std_noise_gnss_acceleration':0,
        }


        training_validation_dataset_params = {
                        'dataset_name':dataset_name, 
                        'dir_dataset':dir_dataset,
                        'percentage_training_instances': 0.7,
                        'instances':1,                          # useful to increase
        }


        dataset_params['seed'] = int(mc)
            
        # create dataset
        dataset_instance = ArtificialDataset(dataset_params)
        dataset_instance.compute_whole_dataset()
        dataset_instance.save_dataset()
        dataset_instance.load_dataset()
        dataset_instance.n = 0

        # Algorithm
        model_params = {'centralized': 1,                                           # 0 for Distributed training
                        'distributed_strategy': 1,                                  # 1, 2, 3
            
                        'log_MPNN_LSTM':0,
                        'log_BP':0,
                        'batch_size_length_seq':dataset_params['number_instants']-1,   # batch size is the length of the sequence for the training of lstm
                        'batch_size_real':10,                                           # number of instances to be trained with simultaneously 
                        
                        'NUM_MEAS': 1, 
                        'NUM_MEAS_GNSS':4,
                        'T_message_steps':10,
                        'num_regress_steps':10-1,

                        'LSTM_num_layers':1,
                        'LSTM_bidirectional': 1,
                        'num_node_latent_feat': dim_,
                        'mu_lstm':1,
                        'mu_mpnn':1,
                        
                        'Particle_filter':0,
                        'number_state_variables':4,
                        'num_particles': 1000}

        # transform in graph
        dataset_graph = GraphDataset(root = dir_dataset, 
                                    train_params = training_validation_dataset_params, 
                                    dataset_params = dataset_params, 
                                    model_params = model_params,
                                    dataset_instance=dataset_instance)
        # each batch is a dataset instance, each dataset instance last for dataset_params['number_instants']-1 instants
        batch_size = model_params['batch_size_length_seq'] 
        dataset_graph_loader = DataLoader(dataset_graph, batch_size=batch_size, shuffle=0)
        iterator_ = iter(dataset_graph_loader)
        data = next(iterator_)  # batch of batch_size elements

        # Algorithms ##############################################################################################################
        # BP
        motion_model = [dataset_params['std_noise_position']**2, 
                        dataset_params['std_noise_position']**2,
                        dataset_params['std_noise_velocity']**2 + (0.01)**2, 
                        dataset_params['std_noise_velocity']**2 + (0.01)**2, 
                        dataset_params['std_noise_acceleration']**2, 
                        dataset_params['std_noise_acceleration']**2]
        motion_model = np.diag(motion_model)
        Q = dataset_instance.W@motion_model@dataset_instance.W.transpose()
        Q = Q[:model_params['number_state_variables'], :model_params['number_state_variables']]
        BP_instance = BP(dataset_instance, dataset_params, model_params, dataset_instance.F[:model_params['number_state_variables'], :model_params['number_state_variables']], Q, motion_model)

        # BP-MPNN-LSTM
        train_params = {'num_epochs': 30}
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        model = NetBP_LSTM(model_params, training_validation_dataset_params, dataset_params).to(device)
        model.device = device
        LR = 0.001
        EPOCHS = 288 # 288
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        # prediction
        model.eval()
        with torch.no_grad():
            data = data.to(device)   
            start = time.time()     
            prediction_NetBP_LSTM = model(data, testing = 1)
            end = time.time()
            print('time train: {}'.format(end-start))

        time_per_step = {f'time_{name_method}': [end-start]}
        
        time_per_step_MC.append(time_per_step)

    for k in time_per_step.keys():
        mean_time[f'{k}_{dim_}'] = [time_per_step_MC[mc][k] for mc in range(MC_iterations)]
Result_name = f'Results_{name_method}'
DIR_EXPERIMENT = create_exp_folder(cwd, DIR_EXPERIMENT=DIR_EXPERIMENT + f'/{Result_name}')
np.save(DIR_EXPERIMENT + f'/mean_time_{name_method}.npy', mean_time, allow_pickle = True) 

print(mean_time)































































































































































































































































































































