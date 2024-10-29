import os
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams.update({'font.size': 22})
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
import json


def create_exp_folder(cwd, number_vehicles = None, number_instants = None, NAME_EXPERIMENT = None, DIR_EXPERIMENT = None):
    if NAME_EXPERIMENT is None and DIR_EXPERIMENT is None:
        NAME_EXPERIMENT = f'/Results/normalloss_normalnet_vehicles{number_vehicles}_instants{number_instants}'
        DIR_EXPERIMENT = cwd + NAME_EXPERIMENT

    # Create folder of EXPERIMENT
    if not os.path.exists(DIR_EXPERIMENT):
        os.makedirs(DIR_EXPERIMENT)
    
    return DIR_EXPERIMENT


def compute_confusion_matrix(sample, output, DIR_EXPERIMENT):
    matrix = confusion_matrix(sample.edge_labels, output)
    # _, [TP, FP,  FN, TN] = fast_compute_class_metric(output, sample.edge_labels)
    TP = matrix[0][0]
    FP = matrix[0][1]
    FN = matrix[1][0]
    TN = matrix[1][1]
    print([TP, FP,  FN, TN])

    fig = plt.figure(figsize=(10,8))
    true = np.sum(matrix, axis = 1)
    predicted = np.sum(matrix, axis = 0)
    x_axis_labels=['Not Association\n\n\n{}'.format(predicted[0]),'Association\n\n\n{}'.format(predicted[1])]
    y_axis_labels=['{}\n\n\nNot Association'.format(true[0]),'{}\n\n\nAssociation'.format(true[1])]
    sns.heatmap(matrix, annot=True, fmt="d", xticklabels=x_axis_labels, yticklabels=y_axis_labels);
    plt.xlabel('Predicted');
    plt.ylabel('True');
    plt.savefig(DIR_EXPERIMENT + '/confusion_matrix.png', bbox_inches='tight')

    confusion_matrix_statistics = {}
    confusion_matrix_statistics['precision'] = TP/(TP+FP)
    confusion_matrix_statistics['recall/sensitivity'] = TP/(TP+FN)
    confusion_matrix_statistics['accuracy'] = (TP+TN)/(TP+FP+TN+FN)
    with open(DIR_EXPERIMENT + '/confusion_matrix_statistics.txt', 'w') as file:
         file.write(json.dumps(confusion_matrix_statistics))




def save_predictions(dataset_params, model_params, dataset_instance, name_method,
                    RMSE_per_step = None, 
                    x_per_step = None,
                    BP_instance=None, 
                    prediction_NetBP_LSTM = None,   
                    time_n = None, 
                    DIR_EXPERIMENT=None):

    if time_n is None:
        time_n = dataset_instance.n
    time_n_NetBP_LSTM = time_n - 1 # the first time instant n = 0 is not predicted by the model

    RMSE_pos_BP = []
    RMSE_vel_BP = []
    RMSE_acc_BP = []

    RMSE_pos_NetBP_LSTM = []
    RMSE_vel_NetBP_LSTM  = []
    RMSE_acc_NetBP_LSTM  = []
    
    for agent in range(dataset_instance.params['num_agents']):

        # GT
        x = np.array(dataset_instance.positions[agent][:time_n+1])[:,0]
        y = np.array(dataset_instance.positions[agent][:time_n+1])[:,1]
        mean_pos = np.array(dataset_instance.positions[agent][:time_n+1])[:,:2]
        vx = np.array(dataset_instance.velocities[agent][:time_n+1])[:,0]
        vy = np.array(dataset_instance.velocities[agent][:time_n+1])[:,1]
        mean_vel = np.array(dataset_instance.velocities[agent][:time_n+1])[:,:2]
        ax = np.array(dataset_instance.accelerations[agent][:time_n+1])[:,0]
        ay = np.array(dataset_instance.accelerations[agent][:time_n+1])[:,1]
        mean_acc = np.array(dataset_instance.accelerations[agent][:time_n+1])[:,:2]

        x_per_step[f'pos_GT'][agent].append(mean_pos[-1,:])
        if model_params['number_state_variables'] >= 4:
            x_per_step[f'vel_GT'][agent].append(mean_vel[-1,:])
            if model_params['number_state_variables'] >= 6:
                x_per_step[f'acc_GT'][agent].append(mean_acc[-1,:])

        if prediction_NetBP_LSTM is not None and time_n!=0:
            # corrispondenza agente-nodo

            x_state_mean = prediction_NetBP_LSTM['outputs_dict_MPNN'][time_n_NetBP_LSTM]['x_hat'][-1].detach().cpu().numpy()[agent,:]
            mean_pos = x_state_mean[:2]
            x_per_step[f'pos_{name_method}'][agent].append(mean_pos)
            if model_params['number_state_variables'] >= 4:
                mean_vel = x_state_mean[2:4]
                x_per_step[f'vel_{name_method}'][agent].append(mean_vel)
                if model_params['number_state_variables'] >= 6:
                    mean_acc = x_state_mean[4:6]
                    x_per_step[f'acc_{name_method}'][agent].append(mean_acc)

            # compute RMSE
            RMSE_pos_NetBP_LSTM.append((x[-1]-mean_pos[0])**2 + (y[-1]-mean_pos[1])**2)
            if model_params['number_state_variables'] >= 4:
                RMSE_vel_NetBP_LSTM.append((vx[-1]-mean_vel[0])**2 + (vy[-1]-mean_vel[1])**2)
                if model_params['number_state_variables'] >= 6:
                    RMSE_acc_NetBP_LSTM.append((ax[-1]-mean_acc[0])**2 + (ay[-1]-mean_acc[1])**2)

        if BP_instance is not None:

            # BP
            beliefs = BP_instance.beliefs
            predicted_x = BP_instance.predicted_x
            predicted_C = BP_instance.predicted_C
            k_ellipse = 2

            if model_params['Particle_filter']:

                x_state_mean = np.array(predicted_x[agent][time_n])
                mean_pos = x_state_mean[:2]

                C = predicted_C[agent][time_n]
                C = C[:2,:2]


            else:
                x_state_mean = beliefs[agent][0]
                mean_pos = x_state_mean[:2]
                C = beliefs[agent][1][0:2, 0:2]
            
            # save BP prediction
            x_per_step[f'pos_{name_method}'][agent].append(mean_pos)
            if model_params['number_state_variables'] >= 4:
                mean_vel = x_state_mean[2:4]
                x_per_step[f'vel_{name_method}'][agent].append(mean_vel)
                if model_params['number_state_variables'] >= 6:
                    mean_acc = x_state_mean[4:6]
                    x_per_step[f'acc_{name_method}'][agent].append(mean_acc)

            # compute RMSE
            RMSE_pos_BP.append((x[-1]-mean_pos[0])**2 + (y[-1]-mean_pos[1])**2)
            if model_params['number_state_variables'] >= 4:
                RMSE_vel_BP.append((vx[-1]-mean_vel[0])**2 + (vy[-1]-mean_vel[1])**2)
                if model_params['number_state_variables'] >= 6:
                    RMSE_acc_BP.append((ax[-1]-mean_acc[0])**2 + (ay[-1]-mean_acc[1])**2)


    if prediction_NetBP_LSTM is not None and time_n!=0:
    
        RMSE_pos_NetBP_LSTM = np.sqrt(np.mean(RMSE_pos_NetBP_LSTM))
        RMSE_per_step[f'pos_{name_method}'].append(RMSE_pos_NetBP_LSTM) 

        if model_params['number_state_variables'] >= 4:
            RMSE_vel_NetBP_LSTM = np.sqrt(np.mean(RMSE_vel_NetBP_LSTM))
            RMSE_per_step[f'vel_{name_method}'].append(RMSE_vel_NetBP_LSTM) 

            if model_params['number_state_variables'] >= 6:
                RMSE_acc_NetBP_LSTM = np.sqrt(np.mean(RMSE_acc_NetBP_LSTM))
                RMSE_per_step[f'acc_{name_method}'].append(RMSE_acc_NetBP_LSTM) 

    if BP_instance is not None:

        RMSE_pos_BP = np.sqrt(np.mean(RMSE_pos_BP))
        RMSE_per_step[f'pos_{name_method}'].append(RMSE_pos_BP) 

        if model_params['number_state_variables'] >= 4:
            RMSE_vel_BP = np.sqrt(np.mean(RMSE_vel_BP))
            RMSE_per_step[f'vel_{name_method}'].append(RMSE_vel_BP) 

            if model_params['number_state_variables'] >= 6:
                RMSE_acc_BP = np.sqrt(np.mean(RMSE_acc_BP))
                RMSE_per_step[f'acc_{name_method}'].append(RMSE_acc_BP) 















































































































































































































































































































































































































































































































































