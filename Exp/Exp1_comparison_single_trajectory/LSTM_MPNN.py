import numpy as np
import os
import sys
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
# import Classes.model
from Classes.dataset import GraphDataset, ArtificialDataset
from Classes.utils import mkdir
from Classes.model import BP, NetBP_LSTM
from Classes.others import create_exp_folder, save_predictions




# Dataset ##############################################################################################################
# import dataset
dir_dataset = cwd + f'/../../../Dataset/{NAME_EXPERIMENT}/'
mkdir(dir_dataset)
dataset_name = 'dataset_test'
dataset_file = dir_dataset + dataset_name

dataset_params = {'load_dataset':0,
                  'reprocess_graph_dataset':1,
                  'dataset_file':dataset_file, 
                  'seed':seed,
                  'limit_behavior':'none',              # 'reflection', 'continue', 'none'
                  'setting_trajectories': 'star',       # 'star', 'spiral', 'not_defined'
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
                'instances':1,                          
}


MC_iterations = 10
RMSE_per_step_MC = []
x_per_step_MC = []

for mc in range(MC_iterations):
        
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
                    'num_node_latent_feat': 16,
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
    model.load_state_dict(torch.load(DIR_EXPERIMENT + f'/Models/model_{EPOCHS}_lr_{LR}', map_location=torch.device(device)))

    # prediction
    model.eval()
    with torch.no_grad():
        data = data.to(device)        
        prediction_NetBP_LSTM = model(data, testing = 1)

    # METHOD USED
    # name_method = 'BP_Kalman_no_coop'
    # name_method = 'BP_particle_no_coop'
    # name_method = 'BP_particle_coop'
    name_method = 'LSTM_MPNN'

    RMSE_per_step = {f'pos_{name_method}': [],
                    f'vel_{name_method}': [],
                    f'acc_{name_method}': []}
    x_per_step = {f'pos_GT': [[] for agent in range(dataset_params['num_agents'])],
                f'vel_GT': [[] for agent in range(dataset_params['num_agents'])],
                f'acc_GT': [[] for agent in range(dataset_params['num_agents'])],
                f'pos_{name_method}': [[] for agent in range(dataset_params['num_agents'])],
                f'vel_{name_method}': [[] for agent in range(dataset_params['num_agents'])],
                f'acc_{name_method}': [[] for agent in range(dataset_params['num_agents'])]}

    for n in range(dataset_params['number_instants']):

        # Real positions
        for agent in range(dataset_params['num_agents']):
            if agent == 1:
                print('Real x: n {} a {} {}'.format(n, agent, dataset_instance.x[agent][n]))
        
        if n == 0 and model_params['log_BP']:
            for agent in range(dataset_params['num_agents']):
                if agent == 0:
                    if not model_params['Particle_filter']:
                        print('Prior mean x: n {} a {} {}'.format(n, agent, BP_instance.beliefs[agent][0].squeeze().tolist()))
                        print('Prior var x: n {} a {}\n{}'.format(n, agent, BP_instance.beliefs[agent][1]))
                    else:
                        prod_particles_weights = np.repeat(BP_instance.beliefs[agent][1].transpose(), BP_instance.model_params['number_state_variables'], 1).transpose()*BP_instance.beliefs[agent][0]
                        x_state_mean = np.sum(prod_particles_weights,1)/np.sum(BP_instance.beliefs[agent][1])
                        mean = x_state_mean[:model_params['number_state_variables']]
                        print('Prior mean x: n {} a {} {}'.format(n, agent, mean.squeeze().tolist()))

        save_predictions(dataset_params, model_params, dataset_instance, name_method,
                                                            prediction_NetBP_LSTM = prediction_NetBP_LSTM, 
                                                            time_n=n, 
                                                            RMSE_per_step = RMSE_per_step, 
                                                            x_per_step = x_per_step,
                                                            DIR_EXPERIMENT = DIR_EXPERIMENT)
    
    RMSE_per_step_MC.append(RMSE_per_step)
    x_per_step_MC.append(x_per_step)

for k in RMSE_per_step.keys():
    RMSE_per_step[k] = np.mean([RMSE_per_step_MC[mc][k] for mc in range(MC_iterations)], 0)

Result_name = f'Results_{name_method}'
DIR_EXPERIMENT = create_exp_folder(cwd, DIR_EXPERIMENT=DIR_EXPERIMENT + f'/{Result_name}')
np.save(DIR_EXPERIMENT + f'/rmse_x_predicted_{name_method}.npy', [RMSE_per_step, x_per_step], allow_pickle = True) 





























































































































































































































































































































