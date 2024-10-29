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
CLASSES_DIR = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]
DIR_EXPERIMENT = os.path.dirname(os.path.abspath(__file__))
NAME_EXPERIMENT = DIR_EXPERIMENT.split(os.sep)[-1]
cwd = DIR_EXPERIMENT
sys.path.append(os.path.dirname(CLASSES_DIR))

# Classes and Functions
from Classes.train_validate_fun import *



# Dataset ##############################################################################################################
# import dataset
dir_dataset = cwd + f'/../../../Dataset/{NAME_EXPERIMENT}/'
dataset_name = 'dataset_2'
dataset_file = dir_dataset + dataset_name

TYPES = ['Results_centralized',
         'Results_distributed_strategy1',
         'Results_distributed_strategy2',
         'Results_distributed_strategy3']
NAMES_TYPES = ['Centralized',
                'Distributed 1',
                'Distributed 2',
                'Distributed 3']

# DIR_RESULTS = DIR_EXPERIMENT+ '/../Create_train_valid_dataset'
DIR_RESULTS = DIR_EXPERIMENT+ '/folder_test_npy'

y_acc = []
keys = []
for i,type_ in enumerate(TYPES):
    y_acc.append([])
    keys.append([])
    metrics_file = np.load(DIR_RESULTS + f'/{type_}/training_epochs.npy', allow_pickle=True).tolist()
    for epoch,v in metrics_file.items():
        y_acc[i].append([v1 for k1, v1 in v.items() if ('loss' not in k1) and ('train' not in k1)])
        if epoch == 1:
            keys[i].append([k1 for k1, v1 in v.items() if ('loss' not in k1) and ('train' not in k1)])
            
# TYPES X EPOCHS X METRICS
y_acc = np.array(y_acc)
for i,type_ in enumerate(TYPES):
    len_seq = len(y_acc[i])
    y_acc[i] = np.array(y_acc[i])

y_acc = np.array(y_acc)

# PLOTTING ROUND-BASED RESULTS
fig = plt.figure(figsize=(25,10))
ax_pos = fig.add_subplot(1, 2, 1)
ax_vel = fig.add_subplot(1, 2, 2)

x_pos = []
y_pos = []
x_vel = []
y_vel = []

for i,type_ in enumerate(TYPES):
    ax_pos.plot(y_acc[i][:,0])

# Format plot
plt.sca(ax_pos)  
plt.xticks(rotation=0, ha='center')
plt.subplots_adjust(bottom=0.30, right = 1)
# plt.ylim(0.7,1)
ax_pos.set(xlabel='Epoch')
ax_pos.set(ylabel='RMSE [m]')
# ax_pos.title.set_text('RMSE pos')
ax_pos.grid()
ax_pos.legend(NAMES_TYPES, loc='best', shadow=False)
ax_pos.set_yscale('log')

for i,type_ in enumerate(TYPES):
    ax_vel.plot(y_acc[i][:,1])

# Format plot
plt.sca(ax_vel)   
plt.xticks(rotation=0, ha='center')
plt.subplots_adjust(bottom=0.30, right = 1)
# plt.ylim(0.7,1)
ax_vel.set(xlabel='Epoch')
ax_vel.set(ylabel='RMSE [m/s]')
# ax_vel.title.set_text('RMSE pos')
ax_vel.grid()
ax_vel.legend(NAMES_TYPES, loc='best', shadow=False)
ax_vel.set_yscale('log')

# plt.show()
plt.savefig(DIR_EXPERIMENT + '/Plot_comparison_centralized_distributed_round.pdf', bbox_inches='tight')
plt.savefig(DIR_EXPERIMENT + '/Plot_comparison_centralized_distributed_round.eps', format='eps', bbox_inches='tight')



# PLOTTING TIME-BASED RESULTS
TIME_EPOCH_TYPES = [
    24,   # centralized
    562, # dis 1
    194,   # dis 2
    80    # dis 3
]

fig = plt.figure(figsize=(25,10))
ax_pos = fig.add_subplot(1, 2, 1)
ax_vel = fig.add_subplot(1, 2, 2)

x_pos = []
y_pos = []
x_vel = []
y_vel = []

for i,type_ in enumerate(TYPES):
    ax_pos.plot(np.arange(len(y_acc[i][:,0]))*TIME_EPOCH_TYPES[i], y_acc[i][:,0])

# Format plot
plt.sca(ax_pos)   
plt.xticks(rotation=0, ha='center')
plt.subplots_adjust(bottom=0.30, right = 1)
# plt.ylim(0.7,1)
ax_pos.set(xlabel='Time [s]')
ax_pos.set(ylabel='RMSE [m]')
ax_pos.grid()
ax_pos.legend(NAMES_TYPES, loc='best', shadow=False)
ax_pos.set_yscale('log')
ax_pos.set_xscale('log')

for i,type_ in enumerate(TYPES):
    ax_vel.plot(np.arange(len(y_acc[i][:,0]))*TIME_EPOCH_TYPES[i], y_acc[i][:,1])

# Format plot
plt.sca(ax_vel)   
plt.xticks(rotation=0, ha='center')
plt.subplots_adjust(bottom=0.30, right = 1)
# plt.ylim(0.7,1)
ax_vel.set(xlabel='Time [s]')
ax_vel.set(ylabel='RMSE [m/s]')
# ax_vel.title.set_text('RMSE pos')
ax_vel.grid()
ax_vel.legend(NAMES_TYPES, loc='best', shadow=False)
ax_vel.set_yscale('log')
ax_vel.set_xscale('log')

# plt.show()
plt.savefig(DIR_EXPERIMENT + '/Plot_comparison_centralized_distributed_seconds.pdf', bbox_inches='tight')
plt.savefig(DIR_EXPERIMENT + '/Plot_comparison_centralized_distributed_seconds.eps', format='eps', bbox_inches='tight')
