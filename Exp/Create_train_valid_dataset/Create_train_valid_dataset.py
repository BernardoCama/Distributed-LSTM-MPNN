import numpy as np
import os
import sys
import time
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import torch
from torch_geometric.loader import DataLoader
from copy import deepcopy
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
from Classes.dataset import GraphDataset
from Classes.utils import mkdir
from Classes.model import NetBP_LSTM, NetBP_LSTM_single
from Classes.train_validate_fun import *
from Classes.others import create_exp_folder
from Classes.plotting import plot_loss_acc, plot_validated_graph_dataset


# Dataset ##############################################################################################################
# import dataset
dir_dataset = cwd + f'/../../../Dataset/{NAME_EXPERIMENT}/'
mkdir(dir_dataset)
dataset_name = 'dataset_2'
dataset_file = dir_dataset + dataset_name

training_validation_dataset_params = {
                'dataset_name':dataset_name, 
                'dir_dataset':dir_dataset,
                'percentage_training_instances': 0.7,
                'instances':1000,                          # useful to increase, 10000 server
}
dataset_params = {'load_dataset':0,
                  'reprocess_graph_dataset':0,
                  'dataset_file':dataset_file, 
                  'seed':seed,
                  'limit_behavior':'none',        # 'reflection', 'continue', 'none'
                  'setting_trajectories': 'not_defined',       # 'star', 'spiral', 'not_defined'
                  'limit_x':[-100, 100],
                  'limit_y':[-100, 100],
                  'limit_vx':[-10, 10],
                  'limit_vy':[-10, 10],
                  'limit_ax':[-10, 10],
                  'limit_ay':[-10, 10],
                  'limit_num_agents': 100,
                  'num_agents': 4, 
                  'number_instants':10,
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
model_params = {'centralized': 1,                                           # 0 for Distributed training
                'distributed_strategy': 3,                                  # 1, 2, 3
    
                'log_MPNN_LSTM':0,
                'log_BP':0,
                'batch_size_length_seq':dataset_params['number_instants']-1,   # batch size is the length of the sequence for the training of lstm
                'batch_size_real':32,                                        # number of instances to be trained with simultaneously 
                
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
                'num_particles': 100}


# Create graph dataset
dataset = GraphDataset(root = dir_dataset, train_params = training_validation_dataset_params, dataset_params = dataset_params, model_params = model_params)

shuffle_dataset = 0
train_dataset = dataset[:int(training_validation_dataset_params['instances']*training_validation_dataset_params['percentage_training_instances']*(dataset_params['number_instants']-1))]
val_dataset = dataset[int(training_validation_dataset_params['instances']*training_validation_dataset_params['percentage_training_instances']*(dataset_params['number_instants']-1)):]

# each batch is a dataset instance, each dataset instance last for dataset_params['number_instants']-1 instants
batch_size = model_params['batch_size_length_seq']
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_dataset)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle_dataset)

iterator_ = iter(train_loader)
data = next(iterator_)  # batch of batch_size elements


# plot training dataset
# plot_graph_dataset(dataset_params, model_params, train_loader, DIR_EXPERIMENT)


# # Models and Parameters ##############################################################################################################
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

LR = 0.001
if model_params['centralized']:
    model = NetBP_LSTM(model_params, training_validation_dataset_params, dataset_params).to(device)
    model.device = device
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
else:
    model = []
    optimizer = []
    for agent in range(dataset_params['num_agents']):

        dataset_params_copy = deepcopy(dataset_params)
        dataset_params_copy['num_agents'] = 1
        dataset_params_copy['agent_id'] = agent
        model_params['device'] = device
        model_ = NetBP_LSTM_single(model_params, training_validation_dataset_params, dataset_params_copy).to(device)
        model_.device = device
        model.append(deepcopy(model_))
        optimizer.append(torch.optim.Adam(model[agent].parameters(), lr=LR))


# # Training ##############################################################################################################
training = 1
train_params = {'num_epochs': 300}  
Result_name = 'Results_centralized'
# Result_name = 'Results_distributed_strategy1'
# Result_name = 'Results_distributed_strategy2'
# Result_name = 'Results_distributed_strategy3'
DIR_EXPERIMENT = create_exp_folder(cwd, DIR_EXPERIMENT=DIR_EXPERIMENT + f'/{Result_name}/')

if training:

    # for plot_loss_acc
    fig = plt.figure(figsize=(25,10))
    ax_acc = fig.add_subplot(1, 2, 1)
    ax_loss = fig.add_subplot(1, 2, 2)

    x_acc = []
    y_acc = []

    x_loss = []
    y_loss = []


    metrics_file = {}

    EPOCHS = train_params['num_epochs']
    for epoch in tqdm(range(EPOCHS)):
        
        start = time.time()
        metrics = train_parallel(dataset_params, model_params, device, model, optimizer, train_loader)
        end = time.time()
        print('time train: {}'.format(end-start))
        
        start = time.time()
        val_metrics = evaluate_parallel(dataset_params, model_params, device, model, val_loader)
        end = time.time()
        print('time valid: {}'.format(end-start))

        print('Epoch: {:03d}, \nTraining: {}, \nValidation: {}\n'.format(epoch, metrics, val_metrics))

        metrics_file[epoch] = {**metrics, **val_metrics}
        
        plot_loss_acc(epoch, metrics, val_metrics, x_loss, x_acc, y_loss, y_acc, ax_acc, ax_loss, DIR_EXPERIMENT, ylog=1)

        if model_params['centralized']:
            torch.save(model.state_dict(), DIR_EXPERIMENT + f'/model_{epoch}_lr_{LR}')
        else:
            torch.save(model[0].state_dict(), DIR_EXPERIMENT + f'/model_{epoch}_lr_{LR}')

        with open(DIR_EXPERIMENT + '/training_epochs.txt', 'w') as file:
            file.write(json.dumps(metrics_file))
        np.save(DIR_EXPERIMENT + f'/training_epochs.npy', metrics_file, allow_pickle = True)

    # Print model's state_dict
    print("Model's state_dict:")
    if model_params['centralized']:
        for param_tensor in model.state_dict():
            print(param_tensor, "\t", model.state_dict()[param_tensor].size())

        torch.save(model.state_dict(), DIR_EXPERIMENT + f'/model_{EPOCHS}_lr_{LR}')
    else:
        for param_tensor in model[0].state_dict():
            print(param_tensor, "\t", model[0].state_dict()[param_tensor].size())

        torch.save(model[0].state_dict(), DIR_EXPERIMENT + f'/model_{EPOCHS}_lr_{LR}')


# # Validation post training ##############################################################################################################
validation = 0
if validation:
    epoch = 270
    model.load_state_dict(torch.load(DIR_EXPERIMENT + f'/model_{epoch}_lr_{LR}', map_location=torch.device(device)))

    model.eval()
    with torch.no_grad():
        for instance,data in enumerate(val_loader):

            data = data.to(device)        
            
            output = model(data, testing = 1)

            plot_validated_graph_dataset(dataset_params, model_params, output, data)


























































































































































































































































































































