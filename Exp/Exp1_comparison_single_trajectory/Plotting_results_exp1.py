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
from statsmodels.distributions.empirical_distribution import ECDF
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
# import Classes.model
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
                'instances':1,                          
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
RMSE_per_step = {}
x_per_step = {}
METHODS = ['BP_Kalman_no_coop','BP_particle_no_coop', 'BP_particle_coop', 'LSTM_MPNN']

fig = plt.figure(figsize=(25,10), constrained_layout=True)
gs = GridSpec(2, int(model_params['number_state_variables']/2), figure=fig)
ax_scenario = fig.add_subplot(gs[:, 0])
ax_error = [fig.add_subplot(gs[var, 1]) for var in range(int(model_params['number_state_variables']/2))]
plt.rcParams.update({'font.size': 22}) 
prop_cycle = plt.rcParams['axes.prop_cycle']
if dataset_params['num_agents'] > 100:
    colors = [plt.cm.Spectral(random.random()) for _ in range(len(METHODS+1))]
else:
    colors = prop_cycle.by_key()['color']
colors_default = prop_cycle.by_key()['color']
color_indexes_per_method = {'GT':-1, 'BP_Kalman_no_coop':0,'BP_particle_no_coop':1, 'BP_particle_coop':2, 'LSTM_MPNN':3}
line_style_indexes_per_method = {'GT':'-', 'BP_Kalman_no_coop':'-.','BP_particle_no_coop':':', 'BP_particle_coop':'--', 'LSTM_MPNN':(0,(3, 1, 1, 1))}

for i, name_method in enumerate(METHODS):
    Result_name = f'Results_{name_method}'
    DIR_EXPERIMENT_method = create_exp_folder(cwd, DIR_EXPERIMENT=DIR_EXPERIMENT + f'/{Result_name}')
    ris = np.load(DIR_EXPERIMENT_method + f'/rmse_x_predicted_{name_method}.npy', allow_pickle=True).tolist()
    RMSE_per_step = {**RMSE_per_step, **ris[0]}
    x_per_step = {**x_per_step, **ris[1]}

    # plot GT
    if i == 0:
        pos_GT = np.array(x_per_step['pos_GT'])
        for agent in range(dataset_params['num_agents']):
            ax_scenario.plot(pos_GT[agent,:,0], pos_GT[agent,:,1], linestyle=line_style_indexes_per_method['GT'], color=colors[color_indexes_per_method['GT']])

    # plot method
    pos_method = np.array(x_per_step[f'pos_{name_method}'])
    for agent in range(dataset_params['num_agents']):
        ax_scenario.plot(pos_method[agent,:,0], pos_method[agent,:,1], linestyle=line_style_indexes_per_method[name_method], color=colors[color_indexes_per_method[name_method]])

    RMSE_pos_per_step_method = RMSE_per_step[f'pos_{name_method}'] 
    ax_error[0].plot(range(len(RMSE_pos_per_step_method)), RMSE_pos_per_step_method, linestyle=line_style_indexes_per_method[name_method], color=colors[color_indexes_per_method[name_method]])
    print('{}: Mean RMSE pos [m]: {:.3f}'.format(name_method, np.mean(RMSE_pos_per_step_method)))
    if len(ax_error) >= 2:
        RMSE_vel_per_step_method = RMSE_per_step[f'vel_{name_method}'] 
        ax_error[1].plot(range(len(RMSE_vel_per_step_method)), RMSE_vel_per_step_method, linestyle=line_style_indexes_per_method[name_method], color=colors[color_indexes_per_method[name_method]])
        print('{}: Mean RMSE vel [m]: {:.3f}'.format(name_method, np.mean(RMSE_vel_per_step_method)))
        if len(ax_error) >= 3:
            RMSE_acc_per_step_method = RMSE_per_step[f'acc_{name_method}'] 
            ax_error[2].plot(range(len(RMSE_acc_per_step_method)), RMSE_acc_per_step_method, linestyle=line_style_indexes_per_method[name_method], color=colors[color_indexes_per_method[name_method]])
            print('{}: Mean RMSE acc [m]: {:.3f}'.format(name_method, np.mean(RMSE_acc_per_step_method)))

ax_scenario.grid()
ax_scenario.axis('equal')
ax_scenario.set(xlim=(dataset_instance.params['limit_x'][0],dataset_instance.params['limit_x'][1]), ylim=(dataset_instance.params['limit_y'][0],dataset_instance.params['limit_y'][1]))
ax_scenario.set(xlabel='X')
ax_scenario.set(ylabel='Y')

ax_error[0].legend([k for k in METHODS], loc='best', shadow=False)
ax_error[0].set(xlabel='Time n')
ax_error[0].set(ylabel='RMSE pos [m]')
ax_error[0].grid()
if len(ax_error) >= 2:
    ax_error[1].legend([k for k in METHODS], loc='best', shadow=False)
    ax_error[1].set(xlabel='Time n')
    ax_error[1].set(ylabel='RMSE vel [m/s]')
    ax_error[1].grid()
    if len(ax_error) >= 3:
        ax_error[2].legend([k for k in METHODS], loc='best', shadow=False)
        ax_error[2].set(xlabel='Time n')
        ax_error[2].set(ylabel='RMSE acc [m/s^2]')
        
plt.savefig(DIR_EXPERIMENT + '/results.pdf', bbox_inches='tight')
plt.savefig(DIR_EXPERIMENT + '/results.eps', format='eps', bbox_inches='tight')



# Plot the empirical CDFs
fig = plt.figure(figsize=(25,10), constrained_layout=True)
gs = GridSpec(1, int(model_params['number_state_variables']/2), figure=fig)
ax_cdf = [fig.add_subplot(gs[0, var]) for var in range(int(model_params['number_state_variables']/2))]
plt.rcParams.update({'font.size': 22}) 


# CDF
for i, name_method in enumerate(METHODS):

    RMSE_pos_per_step_method = RMSE_per_step[f'pos_{name_method}'] 
    cdf_pos = ECDF(RMSE_pos_per_step_method)
    ax_cdf[0].plot(cdf_pos.x, cdf_pos.y, color=colors[color_indexes_per_method[name_method]])
    ax_cdf[0].grid()
    if len(ax_error) >= 2:
        RMSE_vel_per_step_method = RMSE_per_step[f'vel_{name_method}'] 
        cdf_vel = ECDF(RMSE_vel_per_step_method)
        ax_cdf[1].plot(cdf_vel.x, cdf_vel.y, color=colors[color_indexes_per_method[name_method]])
        ax_cdf[1].grid()
        if len(ax_error) >= 3:
            RMSE_acc_per_step_method = RMSE_per_step[f'acc_{name_method}'] 
            cdf_acc = ECDF(RMSE_acc_per_step_method)
            ax_cdf[2].plot(cdf_acc.x, cdf_acc.y, color=colors[color_indexes_per_method[name_method]])
            ax_cdf[2].grid()

ax_cdf[0].legend([k for k in METHODS], loc='best', shadow=False)
if len(ax_cdf) >= 2:
    ax_cdf[1].legend([k for k in METHODS], loc='best', shadow=False)
    if len(ax_cdf) >= 3:
        ax_cdf[2].legend([k for k in METHODS], loc='best', shadow=False)

plt.savefig(DIR_EXPERIMENT + '/cdf_results.pdf', bbox_inches='tight')
plt.savefig(DIR_EXPERIMENT + '/cdf_results.eps', format='eps', bbox_inches='tight')

























































































































































































































































































































