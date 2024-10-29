import numpy as np
import os
import sys
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
from Classes.model import BP
from Classes.others import create_exp_folder, save_predictions




# Dataset ##############################################################################################################
# import dataset
dir_dataset = cwd + f'/../../../Dataset/{NAME_EXPERIMENT}/'
mkdir(dir_dataset)
dataset_name = 'dataset_test'
dataset_file = dir_dataset + dataset_name

# METHOD USED
# name_method = 'BP_Kalman_no_coop'
# name_method = 'BP_particle_no_coop'
name_method = 'BP_particle_coop'
# name_method = 'LSTM_MPNN'
mean_RMSE = {}

NOISES = [0, 0.01, 0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for noise in NOISES:

    print(noise)

    MC_iterations = 30
    RMSE_per_step_MC = []
    x_per_step_MC = [] 
   
    mc = 0
    iter_ = 0
    while mc < MC_iterations:

        print(mc)

        dataset_params = {'load_dataset':0,
                        'reprocess_graph_dataset':1,
                        'dataset_file':dataset_file, 
                        'seed':int(iter_),
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
                        'std_noise_gnss_position':noise,
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
                        'num_node_latent_feat': 16,
                        'mu_lstm':1,
                        'mu_mpnn':1,
                        
                        'Particle_filter':1,
                        'number_state_variables':4,
                        'num_particles': 1000}

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
        inter_agent_meas_model = dataset_params['std_noise_measurements']**2 + 5**2
        # gnss meas
        num_measurements = min(model_params['NUM_MEAS_GNSS'], model_params['number_state_variables'])
        diag_var_gnss = np.array([dataset_params['std_noise_gnss_position']**2 + 5**2, 
                                dataset_params['std_noise_gnss_position']**2 + 5**2,
                                dataset_params['std_noise_gnss_velocity']**2 + 2**2,
                                dataset_params['std_noise_gnss_velocity']**2 + 2**2,
                                dataset_params['std_noise_gnss_acceleration']**2,
                                dataset_params['std_noise_gnss_acceleration']**2][:num_measurements]).astype(float)
        diag_sigma_gnss = np.sqrt(diag_var_gnss)
        Cov_gnss = np.diag(diag_var_gnss)
        BP_instance = BP(dataset_instance, dataset_params, model_params, dataset_instance.F[:model_params['number_state_variables'], :model_params['number_state_variables']], Q, motion_model, inter_agent_meas_model = inter_agent_meas_model, gnss_meas_model = diag_var_gnss)

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

            BP_instance.prediction(n)
            BP_instance.update(dataset_instance, n)
            BP_instance.estimate_position(n)
            save_predictions(dataset_params, model_params, dataset_instance, name_method,
                                                                BP_instance = BP_instance, 
                                                                time_n=n, 
                                                                RMSE_per_step = RMSE_per_step, 
                                                                x_per_step = x_per_step,
                                                                DIR_EXPERIMENT = DIR_EXPERIMENT)

        mc +=1
        RMSE_per_step_MC.append(RMSE_per_step)
        x_per_step_MC.append(x_per_step)
        iter_ += 1

    for k in RMSE_per_step.keys():
        RMSE_per_step[k] = np.mean([RMSE_per_step_MC[mc][k] for mc in range(MC_iterations)], 0)
        try:
            mean_RMSE[f'{k}_{noise}'] = np.mean(RMSE_per_step[k][-1])
        except:
            mean_RMSE[f'{k}_{noise}'] = np.mean(RMSE_per_step[k])


Result_name = f'Results_{name_method}'
DIR_EXPERIMENT = create_exp_folder(cwd, DIR_EXPERIMENT=DIR_EXPERIMENT + f'/{Result_name}')
np.save(DIR_EXPERIMENT + f'/mean_RMSE_{name_method}.npy', mean_RMSE, allow_pickle = True) 

print(mean_RMSE)



























































































































































































































































































































