import numpy as np
import os
import sys
import random
import warnings
warnings.filterwarnings("ignore")
# TO TEST CODE PERFORMANCES
# Create a profiler object
# profiler = cProfile.Profile()
import torch
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams.update({'font.size': 22})
from matplotlib.gridspec import GridSpec

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
from Classes.dataset import ArtificialDataset
from Classes.utils import mkdir
from Classes.others import create_exp_folder


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
                'instances':1,                          # useful to increase
}

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


# METHOD USED
mean_RMSE = {}
color_indexes_per_method = {'GT':-1, 'BP_Kalman_no_coop':0,'BP_particle_no_coop':1, 'BP_particle_coop':2, 'LSTM_MPNN':3}
line_style_indexes_per_method = {'GT':'-', 'BP_Kalman_no_coop':'-.','BP_particle_no_coop':':', 'BP_particle_coop':'--', 'LSTM_MPNN':(0,(3, 1, 1, 1))}
METHODS = ['BP_particle_coop', 'LSTM_MPNN']

fig = plt.figure(figsize=(10,10), constrained_layout=True)
gs = GridSpec(1, 1, figure=fig)
ax_error = [fig.add_subplot(gs[0, var]) for var in range(1)]
plt.rcParams.update({'font.size': 22}) 
prop_cycle = plt.rcParams['axes.prop_cycle']
if dataset_params['num_agents'] > 100:
    colors = [plt.cm.Spectral(random.random()) for _ in range(len(METHODS+1))]
else:
    colors = prop_cycle.by_key()['color']
colors_default = prop_cycle.by_key()['color']

NOISES = [0, 0.01, 0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

for i, name_method in enumerate(METHODS):
    Result_name = f'Results_{name_method}'
    DIR_EXPERIMENT_method = create_exp_folder(cwd, DIR_EXPERIMENT=DIR_EXPERIMENT + f'/{Result_name}')
    ris = np.load(DIR_EXPERIMENT_method + f'/mean_RMSE_{name_method}.npy', allow_pickle=True).tolist()
    mean_RMSE = {**mean_RMSE, **ris}

mean_RMSE_pos = []
for noise_idx, noise in enumerate(NOISES):
    mean_RMSE_pos.append([])
    for i, name_method in enumerate(METHODS):
        mean_RMSE_pos[noise_idx].append(mean_RMSE[f'pos_{name_method}_{noise}'])
mean_RMSE_pos = np.array(mean_RMSE_pos)
if len(ax_error) >= 2:
    mean_RMSE_vel = []
    for noise_idx, noise in enumerate(NOISES):
        mean_RMSE_vel.append([])
        for i, name_method in enumerate(METHODS):
            mean_RMSE_vel[noise_idx].append(mean_RMSE[f'vel_{name_method}_{noise}'])
    mean_RMSE_vel = np.array(mean_RMSE_vel)
    if len(ax_error) >= 3:
        mean_RMSE_acc = []
        for noise_idx, noise in enumerate(NOISES):
            mean_RMSE_acc.append([])
            for i, name_method in enumerate(METHODS):
                mean_RMSE_acc[noise_idx].append(mean_RMSE[f'acc_{name_method}_{noise}'])
        mean_RMSE_acc = np.array(mean_RMSE_acc)

for i, name_method in enumerate(METHODS):
    ax_error[0].plot(NOISES, mean_RMSE_pos[:,i], linestyle=line_style_indexes_per_method[name_method], color=colors[color_indexes_per_method[name_method]])
    ax_error[0].legend([k for k in METHODS], loc='best', shadow=False)
    ax_error[0].set(xlabel='Noise')
    ax_error[0].set(ylabel='RMSE pos [m]')
    ax_error[0].set_xscale('log')
    ax_error[0].grid(True)
        
plt.savefig(DIR_EXPERIMENT + '/results.pdf', bbox_inches='tight')
plt.savefig(DIR_EXPERIMENT + '/results.eps', format='eps', bbox_inches='tight')
























































































































































































































































































































