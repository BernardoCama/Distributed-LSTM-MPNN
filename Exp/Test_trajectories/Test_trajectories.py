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
from Classes.dataset import GraphDataset, ArtificialDataset
from Classes.utils import mkdir
from Classes.model import BP, NetBP_LSTM
from Classes.plotting import plot_step_artificial_dataset



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
                
                'Particle_filter':1,
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
# inter-agent meas
inter_agent_meas_model = dataset_params['std_noise_measurements']**2 + 15**2
# gnss meas
num_measurements = min(model_params['NUM_MEAS_GNSS'], model_params['number_state_variables'])
diag_var_gnss = np.array([dataset_params['std_noise_gnss_position']**2 + 15**2, 
                        dataset_params['std_noise_gnss_position']**2 + 15**2,
                        dataset_params['std_noise_gnss_velocity']**2 + 2**2,
                        dataset_params['std_noise_gnss_velocity']**2 + 2**2,
                        dataset_params['std_noise_gnss_acceleration']**2,
                        dataset_params['std_noise_gnss_acceleration']**2][:num_measurements]).astype(float)
diag_sigma_gnss = np.sqrt(diag_var_gnss)
Cov_gnss = np.diag(diag_var_gnss)
BP_instance = BP(dataset_instance, dataset_params, model_params, dataset_instance.F[:model_params['number_state_variables'], :model_params['number_state_variables']], Q, motion_model, inter_agent_meas_model = inter_agent_meas_model, gnss_meas_model = diag_var_gnss)

# BP-MPNN-LSTM
train_params = {'num_epochs': 30}
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
model = NetBP_LSTM(model_params, training_validation_dataset_params, dataset_params).to(device)
model.device = device
LR = 0.001
EPOCHS = 288
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
model.load_state_dict(torch.load(DIR_EXPERIMENT + f'/Results/model_{EPOCHS}_lr_{LR}', map_location=torch.device(device)))

# prediction
model.eval()
with torch.no_grad():
    data = data.to(device)        
    prediction_NetBP_LSTM = model(data, testing = 1)

plot_only_trajectory = 1
if plot_only_trajectory:
    fig = plt.figure(figsize=(10,10), constrained_layout=True)
    gs = GridSpec(1, 1, figure=fig)
    ax_scenario = fig.add_subplot(gs[0, 0])
else:
    fig = plt.figure(figsize=(25,10), constrained_layout=True)
    gs = GridSpec(2, int(model_params['number_state_variables']/2), figure=fig)
    ax_scenario = fig.add_subplot(gs[:, 0])
    ax_error = [fig.add_subplot(gs[var, 1]) for var in range(int(model_params['number_state_variables']/2))]

RMSE_per_step = {'pos_BP': [],
                'vel_BP': [],
                'acc_BP': [],
                'pos_NetBP_LSTM': [],
                'vel_NetBP_LSTM': [],
                'acc_NetBP_LSTM': []}
x_per_step = {'pos_BP': [[] for agent in range(dataset_params['num_agents'])],
                'vel_BP': [[] for agent in range(dataset_params['num_agents'])],
                'acc_BP': [[] for agent in range(dataset_params['num_agents'])],
                'pos_NetBP_LSTM': [[] for agent in range(dataset_params['num_agents'])],
                'vel_NetBP_LSTM': [[] for agent in range(dataset_params['num_agents'])],
                'acc_NetBP_LSTM': [[] for agent in range(dataset_params['num_agents'])]}
plt.ion()
plt.show()

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

    BP_instance.prediction(n)
    BP_instance.update(dataset_instance, n)
    BP_instance.estimate_position(n)
    ax_scenario, ax_error = plot_step_artificial_dataset(dataset_params, model_params, dataset_instance, 
                                                         ax_scenario, 
                                                         ax_error = ax_error if not plot_only_trajectory else None, 
                                                         BP_instance = BP_instance if not plot_only_trajectory else None, 
                                                         prediction_NetBP_LSTM = prediction_NetBP_LSTM if not plot_only_trajectory else None, 
                                                         time_n=n, 
                                                         plot_particles = 0,
                                                         plot_GNSS = 0,
                                                         set_x_y_lim = 1 if not plot_only_trajectory else None,
                                                         plotting = 1,
                                                         RMSE_per_step = RMSE_per_step, 
                                                         x_per_step = x_per_step,
                                                         DIR_EXPERIMENT = DIR_EXPERIMENT)
        
    if n != dataset_params['number_instants'] - 1:
        ax_scenario.clear()
    else:
        time.sleep(0.01)

plt.savefig(DIR_EXPERIMENT + '/test_trajectory.pdf', bbox_inches='tight')
plt.savefig(DIR_EXPERIMENT + '/test_trajectory.eps', format='eps', bbox_inches='tight')

plt.show(block=True)































































































































































































































































































































