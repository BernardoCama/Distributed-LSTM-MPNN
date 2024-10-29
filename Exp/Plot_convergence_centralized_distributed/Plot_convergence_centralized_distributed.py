import numpy as np
import os
import sys
import warnings
warnings.filterwarnings("ignore")
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
ALL_EXP_DIR = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]
DIR_EXPERIMENT = os.path.dirname(os.path.abspath(__file__))
NAME_EXPERIMENT = DIR_EXPERIMENT.split(os.sep)[-1]
cwd = DIR_EXPERIMENT
sys.path.append(os.path.dirname(ALL_EXP_DIR))

# Classes and Functions
from Classes.train_validate_fun import *



# Dataset ##############################################################################################################
# import dataset
dir_dataset = cwd + f'/../../../Dataset/{NAME_EXPERIMENT}/'
dataset_name = 'dataset_2'
dataset_file = dir_dataset + dataset_name

# DIR_RESULTS = DIR_EXPERIMENT+ '/../Create_train_valid_dataset'
DIR_RESULTS = DIR_EXPERIMENT+ '/folder_test_npy'
DIR_MODELS = ALL_EXP_DIR + '/Create_train_valid_dataset'
train_params = {'num_epochs': 300}  

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
LR = 0.001

TYPES = ['Results_centralized',
         'Results_distributed_strategy1',
         'Results_distributed_strategy2',
         'Results_distributed_strategy3']
NAMES_TYPES = ['Centralized',
                'Distributed 1',
                'Distributed 2',
                'Distributed 3']

# Load data
kl_divs = np.load(DIR_EXPERIMENT + f'/kl.npy', allow_pickle=True).tolist()

# Find global min and max KL divergence values
global_min_kl = np.min([np.min(layer_kl_div) for name, layer_kl_div in kl_divs.items() for layer_name, layer_kl_div in layer_kl_div.items()])
global_max_kl = np.max([np.max(layer_kl_div) for name, layer_kl_div in kl_divs.items() for layer_name, layer_kl_div in layer_kl_div.items()])

plt.rcParams.update({'font.size': 14})
fig, ax = plt.subplots(1, len(TYPES[1:]), figsize=(12, 8), sharex=True)
fig.tight_layout()

for i, (name, kl_div) in enumerate(kl_divs.items()):

    min_values = np.min([layer_kl_div for layer_name, layer_kl_div in kl_div.items()], axis=0)
    max_values = np.max([layer_kl_div for layer_name, layer_kl_div in kl_div.items()], axis=0)

    ax[i].fill_between(range(train_params['num_epochs']), min_values, max_values, color='gray', alpha=0.2)

    for layer_name, layer_kl_div in kl_div.items():
        layer_name = 'regressor' if 'regressor' in layer_name else layer_name
        layer_name = 'edge' if 'edge' in layer_name else layer_name
        layer_name = 'node' if 'node' in layer_name else layer_name
        ax[i].plot(layer_kl_div, label=layer_name)

    ax[i].legend(loc = 'upper right', shadow = False)
    # ax[i].set_title(f"{name} Layers KL Divergence")
    ax[i].set_ylabel("KL Divergence")
    # ax[i].set_ylim(global_min_kl, global_max_kl)
    ax[i].set_ylim(global_min_kl, 1)
    ax[i].set_xlabel("Epochs")
    ax[i].grid(True)

plt.savefig(DIR_EXPERIMENT + '/Plot_convergence_centralized_distributed.pdf', bbox_inches='tight')
plt.savefig(DIR_EXPERIMENT + '/Plot_convergence_centralized_distributed.eps', format='eps', bbox_inches='tight')




















































































































































































































































































































