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

TYPES = ['Results_consensus',
         'Results_distributed_strategy3']
NAMES_TYPES = ['CFA',
                'D-MPNN']

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
fig = plt.figure(figsize=(12,10))
ax_pos = fig.add_subplot(1, 1, 1)

x_pos = []
y_pos = []
x_vel = []
y_vel = []

for i,type_ in enumerate(TYPES):
    ax_pos.plot(y_acc[i][:,0])

# Connect points at corresponding moments in time
time_epoch_types = [58, 80]  # CFA, D-MPNN

# Calculate time for each epoch
time_cfa = np.arange(0, len(y_acc[i][:,0]) * time_epoch_types[0], time_epoch_types[0])
time_dmpnn = np.arange(0, len(y_acc[i][:,0]) * time_epoch_types[1], time_epoch_types[1])

# Find nearest corresponding points
for epoch in range(0, len(y_acc[i][:,0]), 3):
    time_cfa_epoch = epoch * time_epoch_types[0]
    time_dmpnn_epoch = epoch * time_epoch_types[1]
    
    # Finding the nearest D-MPNN epoch for the current CFA epoch
    nearest_dmpnn_epoch = np.argmin(np.abs(time_dmpnn - time_cfa_epoch))
    
    # Plotting a line connecting the corresponding points
    ax_pos.plot([epoch, nearest_dmpnn_epoch], [y_acc[0][epoch, 0], y_acc[1][nearest_dmpnn_epoch, 0]], 'k--', alpha=0.5)


# Format plot
plt.sca(ax_pos)   # Use the pyplot interface to change just one subplot
plt.xticks(rotation=0, ha='center')
plt.subplots_adjust(bottom=0.30, right = 1)
# plt.ylim(0.7,1)
ax_pos.set(xlabel='Epoch')
ax_pos.set(ylabel='RMSE [m]')
# ax_pos.title.set_text('RMSE pos')
ax_pos.grid()
ax_pos.legend(NAMES_TYPES + ['Time alignment___'], loc='best', shadow=False)
ax_pos.set_yscale('log')


# plt.show()
plt.savefig(DIR_EXPERIMENT + '/Plot_comparison_baselines_round.pdf', bbox_inches='tight')
plt.savefig(DIR_EXPERIMENT + '/Plot_comparison_baselines_round.eps', format='eps', bbox_inches='tight')


























































































































































































































































































































