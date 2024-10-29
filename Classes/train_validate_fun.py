import torch
from torch import nn
import pandas as pd
import numpy as np
from copy import deepcopy

def train(dataset_params, model_params, device, model, optimizer, train_loader):
    
    type_ = 'train'
    
    model.train()

    logs_all = []
    
    for data in train_loader:
        
        data = data.to(device)
        
        optimizer.zero_grad()
        
        output = model(data)
        
        loss = _compute_loss(device, model_params, output, data)
        
        metrics = compute_perform_metrics(device, model_params, output, data)
        
        logs = {**metrics, **{'loss': loss.item()}}
        
        log = {key + f'/{type_}': val for key, val in logs.items()}
        
        loss.backward()

        logs_all.append(log)
        
        optimizer.step()
    
    return epoch_end(logs_all)


def _compute_loss(device, model_params, outputs, batch):
    
    mse_loss = nn.MSELoss()
    weights_x = batch[0].mean_std_min_max_variables['limit_features_x_max'] - batch[0].mean_std_min_max_variables['limit_features_x_min']
    weights_x = 1/weights_x
    weights_x = torch.tensor(weights_x/weights_x.sum()).to(device)
    
    batch_size = model_params['batch_size_length_seq']
    T_message_steps = model_params['T_message_steps']
    loss = 0
    for n in range(batch_size):

        # GT
        node_next_x = batch[n].node_next_x

        # Prediction
        output_lstm = outputs['output_lstm'][n].squeeze()
        outputs_dict_MPNN = outputs['outputs_dict_MPNN'][n]

        # all message passing steps
        for step in range(len(outputs_dict_MPNN['x_hat'])):
            loss += weighted_mse_loss_fun(node_next_x, outputs_dict_MPNN['x_hat'][step], weights = weights_x) * torch.tensor([model_params['mu_mpnn']], dtype=torch.float).to(device)

    return loss




def evaluate(dataset_params, model_params, device, model, loader, thr=None):
    type_ = 'val'
    model.eval()

    logs_all = []

    with torch.no_grad():
        for data in loader:

            data = data.to(device)        
            
            output = model(data, testing = 1)
            
            loss = _compute_loss(device, model_params, output, data)
            
            metrics = compute_perform_metrics(device, model_params, output, data)
            
            logs = {**metrics, **{'loss': loss.item()}}
            
            log = {key + f'/{type_}': val for key, val in logs.items()}

            logs_all.append(log)            
            
    return epoch_end(logs_all)
    
def epoch_end(outputs):
    metrics = pd.DataFrame(outputs).mean(axis=0).to_dict()
    metrics = {metric_name: torch.as_tensor(metric).item() for metric_name, metric in metrics.items()}
    return metrics


def mse_loss_fun(input, target, reduction = None):
    if reduction is None or reduction == 'mean':
        return torch.mean((input - target) ** 2)
    elif reduction == 'sum':
        return torch.sum((input - target) ** 2)

def weighted_mse_loss_fun(input, target, reduction = None, weights = None):
    if reduction is None or reduction == 'mean':
        return torch.mean(weights * (input - target) ** 2)
    elif reduction == 'sum':
        return torch.sum(weights * (input - target) ** 2)

def compute_perform_metrics(device, model_params, outputs, batch):

    batch_size = model_params['batch_size_length_seq']
    T_message_steps = model_params['T_message_steps']
    rmse_pos = []
    rmse_vel = []
    rmse_acc = []
    for n in range(batch_size):

        # GT
        node_next_x = batch[n].node_next_x.detach().cpu().numpy()

        # Prediction
        output_lstm = outputs['output_lstm'][n].squeeze().detach().cpu().numpy()
        outputs_dict_MPNN_x_hat = outputs['outputs_dict_MPNN'][n]['x_hat']

        # Log
        if model_params['log_MPNN_LSTM']:
            agent = 0
            print('Input x not normalized: a {} {}'.format(agent,  batch[n].x_not_normalized.detach().cpu().numpy()[agent,:]))
            print('Input x: a {} {}'.format(agent,  batch[n].x.detach().cpu().numpy()[0,:]))
            print('GNSS z: a {} {}'.format(agent,  batch[n].node_attr_not_normalized.detach().cpu().numpy()[agent,:]))
            print('Real x_next: a {} {}'.format(agent,batch[n].node_next_x.detach().cpu().numpy()[agent,:]))
            print('Prediction x_next: a {} {}'.format(agent,output_lstm[0,:]))
            print('Update x_next: a {} {}\n'.format(agent,outputs_dict_MPNN_x_hat[-1][agent,:].detach().cpu().numpy()))
        
        # From LSTM-MPNN
        rmse_pos += [np.sqrt(np.sum((node_next_x[:,:2] - outputs_dict_MPNN_x_hat[-1].detach().cpu().numpy()[:,:2])**2, 1))]
        if model_params['number_state_variables'] >=4:
            rmse_vel += [np.sqrt(np.sum((node_next_x[:,2:4] - outputs_dict_MPNN_x_hat[-1].detach().cpu().numpy()[:,2:4])**2, 1))]
            if model_params['number_state_variables'] >=6:
                rmse_acc += [np.sqrt(np.sum((node_next_x[:,4:6] - outputs_dict_MPNN_x_hat[-1].detach().cpu().numpy()[:,4:6])**2, 1))]

    rmse_pos = np.mean(rmse_pos)
    if model_params['number_state_variables'] >=4:
        rmse_vel = np.mean(rmse_vel)
        if model_params['number_state_variables'] >=6:
            rmse_acc = np.mean(rmse_acc)

    return {'rmse_pos':rmse_pos, 'rmse_vel':rmse_vel, 'rmse_acc':rmse_acc}

def fast_compute_class_metric(test_preds, test_sols, class_metrics = ('accuracy', 'recall', 'precision')):
    """
    Computes manually (i.e. without sklearn functions) accuracy, recall and predicision.

    Args:
        test_preds: numpy array/ torch tensor of size N with discrete output vals
        test_sols: numpy array/torch tensor of size N with binary labels
        class_metrics: tuple with a subset of values from ('accuracy', 'recall', 'precision') indicating which
        metrics to report

    Returns:
        dictionary with values of accuracy, recall and precision
    """
    with torch.no_grad():

        TP = ((test_sols == 1) & (test_preds == 1)).sum().float()
        FP = ((test_sols == 0) & (test_preds == 1)).sum().float()
        TN = ((test_sols == 0) & (test_preds == 0)).sum().float()
        FN = ((test_sols == 1) & (test_preds == 0)).sum().float()

        accuracy = (TP + TN) / (TP + FP + TN + FN)
        recall = TP / (TP + FN) if TP + FN > 0 else torch.tensor(0)
        precision = TP / (TP + FP) if TP + FP > 0 else torch.tensor(0)

    class_metrics_dict =  {'accuracy': accuracy.item(), 'recall': recall.item(), 'precision': precision.item()}
    class_metrics_dict = {met_name: class_metrics_dict[met_name] for met_name in class_metrics}
    
    return class_metrics_dict, [TP.tolist(), FP.tolist(),  FN.tolist(), TN.tolist()]





##################################################################################################################################################################


# Federated averaging: FedAvg
def FedAvg(w):
    w_avg = deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def train_parallel(dataset_params, model_params, device, model, optimizer, train_loader):

    if model_params['centralized']:
        
        type_ = 'train'
        
        model.train()

        logs_all = []

        batch_size_real = model_params['batch_size_real']
        
        # batch_list contains a list of length batch_size_real of fake batches of length dataset_params['number_instants']-1, i.e., an entire instance composed of
        # dataset_params['number_instants']-1 graphs, each with dataset_params['num_agents'] nodes.
        batch_list = []
        for i,data in enumerate(train_loader):
            
            data = data.to(device)
            batch_list.append(data)

            if (i+1)%batch_size_real == 0:
        
                optimizer.zero_grad()
                
                output = model(batch_list)
                
                loss = _compute_loss_parallel(device, model_params, output, batch_list)
                
                metrics = compute_perform_metrics_parallel(device, model_params, output, batch_list)
                
                logs = {**metrics, **{'loss': loss.item()}}
                
                log = {key + f'/{type_}': val for key, val in logs.items()}
                
                loss.backward()

                logs_all.append(log)
                
                optimizer.step()

                batch_list = []
    
    # distributed
    else:

        # 2SS
        if model_params['distributed_strategy'] == 2:

            type_ = 'train'

            logs_all = []

            batch_size_real = model_params['batch_size_real']

            batch_list = []
            for i,data in enumerate(train_loader):
                
                data = data.to(device)
                batch_list.append(data)

                if (i+1)%batch_size_real == 0:

                    # timestamp
                    for n in range(model_params['batch_size_length_seq']):

                        # reshape such that we have a batch size composed of the number of instances (batch_size_real) * num_agents
                        x = torch.cat(([batch_list[b][n].x for b in range(batch_size_real)]), 0)
                        node_next_x = torch.cat(([batch_list[b][n].node_next_x for b in range(batch_size_real)]), 0)
                        node_id_original = torch.cat(([batch_list[b][n].node_id for b in range(batch_size_real)]), 0)
                        z_gnss = torch.cat(([batch_list[b][n].node_attr for b in range(batch_size_real)]), 0)
                        num_nodes = [batch_list[b][n].x.shape[0] for b in range(batch_size_real)]
                        # adjust indexes of edges for batch procedure
                        num_nodes[0] = 0
                        num_nodes = np.cumsum(num_nodes)
                        edge_index_original = torch.cat(([batch_list[b][n].edge_index for b in range(batch_size_real)]), 1)
                        edge_index = torch.cat(([batch_list[b][n].edge_index+num_nodes[b] for b in range(batch_size_real)]), 1)
                        z_inter_meas = torch.cat(([batch_list[b][n].edge_attr for b in range(batch_size_real)]), 0)

                        weights_x = batch_list[0][0].mean_std_min_max_variables['limit_features_x_max'] - batch_list[0][0].mean_std_min_max_variables['limit_features_x_min']
                        weights_x = 1/weights_x
                        weights_x = torch.tensor(weights_x/weights_x.sum()).to(device)

                        x_lstm_agents = []
                        outputs_dict_agents = []
                        encoded_z_inter_meas_agents = []
                        encoded_z_gnss_agents = []
                        latent_node_feats_agents = []
                        latent_edge_feats_agents = []
                        loss_agents = []
                        logs_agents = []

                        # LOCAL MODELS (ENCODING AND LSTM) 0
                        for agent in range(dataset_params['num_agents']):

                            # to recover the index of the nodes (for attributes) in the batch of graphs (all related to timestamp n)
                            node_indexes_related_to_agent = node_id_original == agent
                            edge_indexes_related_to_agent = edge_index_original[1,:] == agent

                            model[agent].train()
                
                            optimizer[agent].zero_grad()
                            
                            x_lstm = model[agent].lstm(x[node_indexes_related_to_agent], n).squeeze()
                            x_lstm_agents.append(x_lstm)
                        
                            encoded_z_inter_meas, encoded_z_gnss = model[agent].encoder(z_inter_meas[edge_indexes_related_to_agent], z_gnss[node_indexes_related_to_agent])
                            encoded_z_inter_meas_agents.append(encoded_z_inter_meas)
                            encoded_z_gnss_agents.append(encoded_z_gnss)

                            latent_node_feats = x_lstm
                            latent_edge_feats = encoded_z_inter_meas
                            latent_node_feats_agents.append(latent_node_feats)  
                            latent_edge_feats_agents.append(latent_edge_feats)

                            # During training, the feature vectors that the MPNetwork outputs for the  last self.num_class_steps message
                            # passing steps are classified in order to compute the loss.
                            first_class_step = model_params['T_message_steps'] - model_params['num_regress_steps'] + 1 # total 2 = 12  Number of message passing steps - # 11  Number of message passing steps during feature vectors are classified (after Message Passing) + 1
                            outputs_dict = {'x_hat': []}
                            outputs_dict_agents.append(outputs_dict)

                        for step in range(1, model_params['T_message_steps'] + 1):

                            for agent in range(dataset_params['num_agents']):

                                node_indexes_related_to_agent = node_id_original == agent
                                edge_indexes_related_to_agent = edge_index_original[1,:] == agent
                    
                                # Reattach the initially encoded embeddings before the update
                                latent_edge_feats_agents[agent] = torch.cat((encoded_z_inter_meas_agents[agent], latent_edge_feats_agents[agent].squeeze()), dim=1)
                    
                                # FORWARD 1
                                # Message Passing Step                           
                                # avoid backpropagate with outputs of other models     
                                latent_node_feats_agents_tmp = [latent_node_feats_agents[a] if a == agent else latent_node_feats_agents[a].clone().detach().requires_grad_(True) for a in range(dataset_params['num_agents'])]      
                                latent_node_feats_agents[agent], latent_edge_feats_agents[agent] = model[agent].MPNet(torch.cat(latent_node_feats_agents_tmp).squeeze(), x_lstm_agents[agent], encoded_z_gnss_agents[agent], edge_index[:,edge_indexes_related_to_agent], latent_edge_feats_agents[agent], node_indexes_related_to_agent, edge_indexes_related_to_agent)

                                # Regression Step              
                                _, dec_node_feats = model[agent].regressor(nodes_feats = latent_node_feats_agents[agent])
                                outputs_dict_agents[agent]['x_hat'].append(dec_node_feats)
                            
                                loss = _compute_loss_single_agent(device, weights_x, model_params, outputs_dict_agents[agent]['x_hat'], node_next_x[node_indexes_related_to_agent])

                                # COMPUTE GRADIENTS and BACK-PROPAGATE 2
                                loss.backward()

                                optimizer[agent].step()

                                if step == model_params['T_message_steps']:
                                    loss_agents.append(loss)

                                # Stop gradient for next message passing
                                latent_node_feats_agents[agent] = latent_node_feats_agents[agent].clone().detach().requires_grad_(True)
                                latent_edge_feats_agents[agent] = latent_edge_feats_agents[agent].clone().detach().requires_grad_(True)
                                x_lstm_agents[agent] = x_lstm_agents[agent].clone().detach().requires_grad_(True)
                                encoded_z_inter_meas_agents[agent] = encoded_z_inter_meas_agents[agent].clone().detach().requires_grad_(True)
                                encoded_z_gnss_agents[agent] = encoded_z_gnss_agents[agent].clone().detach().requires_grad_(True)  

                                model[agent].train()
                                optimizer[agent].zero_grad()          

                        # At the end of message passing, evaluate models
                        for agent in range(dataset_params['num_agents']):
                            node_indexes_related_to_agent = node_id_original == agent
                            edge_indexes_related_to_agent = edge_index_original[1,:] == agent
                            # Compute performance metrics
                            metrics = compute_perform_metrics_single_agent(device, model_params, outputs_dict_agents[agent]['x_hat'], node_next_x[node_indexes_related_to_agent])
                            logs = {**metrics, **{'loss': loss_agents[agent].item()}}
                            logs_agents.append(logs)

                        # Mean RMSE
                        logs = {k: sum([logs_agents[agent][k] for agent in range(dataset_params['num_agents'])]) / dataset_params['num_agents']  if type(logs_agents[0][k]) is not list else [] for k in logs_agents[0]}
                        log = {key + f'/{type_}': val for key, val in logs.items()}

                        # CONSENSUS 3    
                        # Consensus
                        model_global = FedAvg([deepcopy(model[agent].state_dict()) for agent in range(dataset_params['num_agents'])])

                        # # Update global model
                        for agent in range(dataset_params['num_agents']):
                            model[agent].load_state_dict(model_global)  

                    logs_all.append(log)
                    batch_list = []

        # 3SS
        elif model_params['distributed_strategy'] == 1:

            type_ = 'train'

            logs_all = []

            batch_size_real = model_params['batch_size_real']

            batch_list = []
            for i,data in enumerate(train_loader):
                
                data = data.to(device)
                batch_list.append(data)

                if (i+1)%batch_size_real == 0:

                    # timestamp
                    for n in range(model_params['batch_size_length_seq']):

                        # reshape such that we have a batch size composed of the number of instances (batch_size_real) * num_agents
                        x = torch.cat(([batch_list[b][n].x for b in range(batch_size_real)]), 0)
                        node_next_x = torch.cat(([batch_list[b][n].node_next_x for b in range(batch_size_real)]), 0)
                        node_id_original = torch.cat(([batch_list[b][n].node_id for b in range(batch_size_real)]), 0)
                        z_gnss = torch.cat(([batch_list[b][n].node_attr for b in range(batch_size_real)]), 0)
                        num_nodes = [batch_list[b][n].x.shape[0] for b in range(batch_size_real)]
                        # adjust indexes of edges for batch procedure
                        num_nodes[0] = 0
                        num_nodes = np.cumsum(num_nodes)
                        edge_index_original = torch.cat(([batch_list[b][n].edge_index for b in range(batch_size_real)]), 1)
                        edge_index = torch.cat(([batch_list[b][n].edge_index+num_nodes[b] for b in range(batch_size_real)]), 1)
                        z_inter_meas = torch.cat(([batch_list[b][n].edge_attr for b in range(batch_size_real)]), 0)

                        weights_x = batch_list[0][0].mean_std_min_max_variables['limit_features_x_max'] - batch_list[0][0].mean_std_min_max_variables['limit_features_x_min']
                        weights_x = 1/weights_x
                        weights_x = torch.tensor(weights_x/weights_x.sum()).to(device)

                        x_lstm_agents = []
                        outputs_dict_agents = []
                        encoded_z_inter_meas_agents = []
                        encoded_z_gnss_agents = []
                        latent_node_feats_agents = []
                        latent_edge_feats_agents = []
                        loss_agents = []
                        logs_agents = []
                        gradients_agents = []

                        # LOCAL MODELS (ENCODING AND LSTM) 0
                        for agent in range(dataset_params['num_agents']):

                            # to recover the index of the nodes (for attributes) in the batch of graphs (all related to timestamp n)
                            node_indexes_related_to_agent = node_id_original == agent
                            edge_indexes_related_to_agent = edge_index_original[1,:] == agent

                            model[agent].train()
                
                            optimizer[agent].zero_grad()
                            
                            x_lstm = model[agent].lstm(x[node_indexes_related_to_agent], n).squeeze()
                            x_lstm_agents.append(x_lstm)
                        
                            encoded_z_inter_meas, encoded_z_gnss = model[agent].encoder(z_inter_meas[edge_indexes_related_to_agent], z_gnss[node_indexes_related_to_agent])
                            encoded_z_inter_meas_agents.append(encoded_z_inter_meas)
                            encoded_z_gnss_agents.append(encoded_z_gnss)

                            latent_node_feats = x_lstm
                            latent_edge_feats = encoded_z_inter_meas
                            latent_node_feats_agents.append(latent_node_feats) 
                            latent_edge_feats_agents.append(latent_edge_feats)

                            # During training, the feature vectors that the MPNetwork outputs for the  last self.num_class_steps message
                            # passing steps are classified in order to compute the loss.
                            first_class_step = model_params['T_message_steps'] - model_params['num_regress_steps'] + 1 # total 2 = 12  Number of message passing steps - # 11  Number of message passing steps during feature vectors are classified (after Message Passing) + 1
                            outputs_dict = {'x_hat': []}
                            outputs_dict_agents.append(outputs_dict)

                        for step in range(1, model_params['T_message_steps'] + 1):

                            for agent in range(dataset_params['num_agents']):

                                # store input
                                if step == 1:
                                    original_latent_node_feats_agents = latent_node_feats_agents[agent]
                                    original_encoded_z_inter_meas_agents = encoded_z_inter_meas_agents[agent]
                                    original_latent_edge_feats_agents = latent_edge_feats_agents[agent]

                                node_indexes_related_to_agent = node_id_original == agent
                                edge_indexes_related_to_agent = edge_index_original[1,:] == agent
                    
                                # Reattach the initially encoded embeddings before the update
                                latent_edge_feats_agents[agent] = torch.cat((encoded_z_inter_meas_agents[agent], latent_edge_feats_agents[agent].squeeze()), dim=1)
                    
                                # FORWARD 1
                                # Message Passing Step                           
                                # avoid backpropagate with outputs of other models     
                                latent_node_feats_agents_tmp = [latent_node_feats_agents[a] if a == agent else latent_node_feats_agents[a].clone().detach().requires_grad_(True) for a in range(dataset_params['num_agents'])]      
                                latent_node_feats_agents[agent], latent_edge_feats_agents[agent] = model[agent].MPNet(torch.cat(latent_node_feats_agents_tmp).squeeze(), x_lstm_agents[agent], encoded_z_gnss_agents[agent], edge_index[:,edge_indexes_related_to_agent], latent_edge_feats_agents[agent], node_indexes_related_to_agent, edge_indexes_related_to_agent)

                                
                                # Regression Step              
                                _, dec_node_feats = model[agent].regressor(nodes_feats = latent_node_feats_agents[agent])
                                outputs_dict_agents[agent]['x_hat'].append(dec_node_feats)
                            
                                loss = _compute_loss_single_agent(device, weights_x, model_params, outputs_dict_agents[agent]['x_hat'], node_next_x[node_indexes_related_to_agent])

                                # COMPUTE GRADIENTS 2
                                loss.backward()
                                gradients_agents.append({k:v.grad.clone().detach() for k, v in zip(model[agent].state_dict(), model[agent].parameters())})

                                # optimizer[agent].step()

                                if step == model_params['T_message_steps']:
                                    loss_agents.append(loss)

                                # Stop gradient for next message passing
                                latent_node_feats_agents[agent] = latent_node_feats_agents[agent].clone().detach().requires_grad_(True)
                                latent_edge_feats_agents[agent] = latent_edge_feats_agents[agent].clone().detach().requires_grad_(True)
                                x_lstm_agents[agent] = x_lstm_agents[agent].clone().detach().requires_grad_(True)
                                encoded_z_inter_meas_agents[agent] = encoded_z_inter_meas_agents[agent].clone().detach().requires_grad_(True)
                                encoded_z_gnss_agents[agent] = encoded_z_gnss_agents[agent].clone().detach().requires_grad_(True)  

                            # CONSENSUS 3
                            # Consensus
                            for agent in range(dataset_params['num_agents']):
                                
                                gradients_global = FedAvg(gradients_agents)
                                for k, v in zip(model[agent].state_dict(), model[agent].parameters()):
                                    model[agent].grad = gradients_global[k]
                                optimizer[agent].step()

                                model[agent].train()
                                optimizer[agent].zero_grad()          

                        # At the end of message passing, evaluate models
                        for agent in range(dataset_params['num_agents']):
                            node_indexes_related_to_agent = node_id_original == agent
                            edge_indexes_related_to_agent = edge_index_original[1,:] == agent
                            # Compute performance metrics
                            metrics = compute_perform_metrics_single_agent(device, model_params, outputs_dict_agents[agent]['x_hat'], node_next_x[node_indexes_related_to_agent])
                            logs = {**metrics, **{'loss': loss_agents[agent].item()}}
                            logs_agents.append(logs)

                        # Mean RMSE
                        logs = {k: sum([logs_agents[agent][k] for agent in range(dataset_params['num_agents'])]) / dataset_params['num_agents']  if type(logs_agents[0][k]) is not list else [] for k in logs_agents[0]}
                        log = {key + f'/{type_}': val for key, val in logs.items()}                           

                    logs_all.append(log)
                    batch_list = []


        # D-MPNN
        elif model_params['distributed_strategy'] == 3:

            type_ = 'train'

            logs_all = []

            batch_size_real = model_params['batch_size_real']

            batch_list = []
            for i,data in enumerate(train_loader):
                
                data = data.to(device)
                batch_list.append(data)

                if (i+1)%batch_size_real == 0:

                    # timestamp
                    for n in range(model_params['batch_size_length_seq']):

                        # reshape such that we have a batch size composed of the number of instances (batch_size_real) * num_agents
                        x = torch.cat(([batch_list[b][n].x for b in range(batch_size_real)]), 0)
                        node_next_x = torch.cat(([batch_list[b][n].node_next_x for b in range(batch_size_real)]), 0)
                        node_id_original = torch.cat(([batch_list[b][n].node_id for b in range(batch_size_real)]), 0)
                        z_gnss = torch.cat(([batch_list[b][n].node_attr for b in range(batch_size_real)]), 0)
                        num_nodes = [batch_list[b][n].x.shape[0] for b in range(batch_size_real)]
                        # adjust indexes of edges for batch procedure
                        num_nodes[0] = 0
                        num_nodes = np.cumsum(num_nodes)
                        edge_index_original = torch.cat(([batch_list[b][n].edge_index for b in range(batch_size_real)]), 1)
                        edge_index = torch.cat(([batch_list[b][n].edge_index+num_nodes[b] for b in range(batch_size_real)]), 1)
                        z_inter_meas = torch.cat(([batch_list[b][n].edge_attr for b in range(batch_size_real)]), 0)

                        weights_x = batch_list[0][0].mean_std_min_max_variables['limit_features_x_max'] - batch_list[0][0].mean_std_min_max_variables['limit_features_x_min']
                        weights_x = 1/weights_x
                        weights_x = torch.tensor(weights_x/weights_x.sum()).to(device)

                        x_lstm_agents = []
                        outputs_dict_agents = []
                        encoded_z_inter_meas_agents = []
                        encoded_z_gnss_agents = []
                        latent_node_feats_agents = []
                        latent_edge_feats_agents = []
                        loss_agents = []
                        logs_agents = []

                        # LOCAL MODELS (ENCODING AND LSTM) 0
                        for agent in range(dataset_params['num_agents']):

                            # to recover the index of the nodes (for attributes) in the batch of graphs (all related to timestamp n)
                            node_indexes_related_to_agent = node_id_original == agent
                            edge_indexes_related_to_agent = edge_index_original[1,:] == agent

                            model[agent].train()
                
                            optimizer[agent].zero_grad()
                            
                            x_lstm = model[agent].lstm(x[node_indexes_related_to_agent], n).squeeze()
                            x_lstm_agents.append(x_lstm)
                        
                            encoded_z_inter_meas, encoded_z_gnss = model[agent].encoder(z_inter_meas[edge_indexes_related_to_agent], z_gnss[node_indexes_related_to_agent])
                            encoded_z_inter_meas_agents.append(encoded_z_inter_meas)
                            encoded_z_gnss_agents.append(encoded_z_gnss)

                            latent_node_feats = x_lstm
                            latent_edge_feats = encoded_z_inter_meas
                            latent_node_feats_agents.append(latent_node_feats) 
                            latent_edge_feats_agents.append(latent_edge_feats)

                            # During training, the feature vectors that the MPNetwork outputs for the  last self.num_class_steps message
                            # passing steps are classified in order to compute the loss.
                            first_class_step = model_params['T_message_steps'] - model_params['num_regress_steps'] + 1 # total 2 = 12  Number of message passing steps - # 11  Number of message passing steps during feature vectors are classified (after Message Passing) + 1
                            outputs_dict = {'x_hat': []}
                            outputs_dict_agents.append(outputs_dict)

                        for step in range(1, model_params['T_message_steps'] + 1):

                            for agent in range(dataset_params['num_agents']):

                                node_indexes_related_to_agent = node_id_original == agent
                                edge_indexes_related_to_agent = edge_index_original[1,:] == agent
                    
                                # Reattach the initially encoded embeddings before the update
                                latent_edge_feats_agents[agent] = torch.cat((encoded_z_inter_meas_agents[agent], latent_edge_feats_agents[agent].squeeze()), dim=1)
                    
                                # Message Passing Step                           
                                # avoid backpropagate with outputs of other models     
                                latent_node_feats_agents_tmp = [latent_node_feats_agents[a] if a == agent else latent_node_feats_agents[a].clone().detach().requires_grad_(True) for a in range(dataset_params['num_agents'])]      
                                latent_node_feats_agents[agent], latent_edge_feats_agents[agent] = model[agent].MPNet(torch.cat(latent_node_feats_agents_tmp).squeeze(), x_lstm_agents[agent], encoded_z_gnss_agents[agent], edge_index[:,edge_indexes_related_to_agent], latent_edge_feats_agents[agent], node_indexes_related_to_agent, edge_indexes_related_to_agent)

                                
                                # Regression Step              
                                _, dec_node_feats = model[agent].regressor(nodes_feats = latent_node_feats_agents[agent])
                                outputs_dict_agents[agent]['x_hat'].append(dec_node_feats)

                        # At the end of message passing, evaluate models
                        for agent in range(dataset_params['num_agents']):
                            node_indexes_related_to_agent = node_id_original == agent
                            edge_indexes_related_to_agent = edge_index_original[1,:] == agent

                            # loss
                            loss = _compute_loss_single_agent_all_message_passing(device, weights_x, model_params, outputs_dict_agents[agent]['x_hat'], node_next_x[node_indexes_related_to_agent])

                            # COMPUTE GRADIENTS and BACK-PROPAGATE 2
                            loss.backward()

                            optimizer[agent].step()

                            loss_agents.append(loss)

                            model[agent].train()
                            optimizer[agent].zero_grad()     

                            # Compute performance metrics
                            metrics = compute_perform_metrics_single_agent(device, model_params, outputs_dict_agents[agent]['x_hat'], node_next_x[node_indexes_related_to_agent])
                            logs = {**metrics, **{'loss': loss_agents[agent].item()}}
                            logs_agents.append(logs)

                        # Mean RMSE
                        logs = {k: sum([logs_agents[agent][k] for agent in range(dataset_params['num_agents'])]) / dataset_params['num_agents']  if type(logs_agents[0][k]) is not list else [] for k in logs_agents[0]}
                        log = {key + f'/{type_}': val for key, val in logs.items()}

                        # CONSENSUS 3    
                        # Consensus
                        model_global = FedAvg([deepcopy(model[agent].state_dict()) for agent in range(dataset_params['num_agents'])])

                        # # Update global model
                        for agent in range(dataset_params['num_agents']):
                            model[agent].load_state_dict(model_global)  

                    logs_all.append(log)
                    batch_list = []


        # Consensus -> just 1 iteration of MPNN in the forward part
        elif model_params['distributed_strategy'] == 4:

            type_ = 'train'

            logs_all = []

            batch_size_real = model_params['batch_size_real']

            batch_list = []
            for i,data in enumerate(train_loader):
                
                data = data.to(device)
                batch_list.append(data)

                if (i+1)%batch_size_real == 0:

                    # timestamp
                    for n in range(model_params['batch_size_length_seq']):

                        # reshape such that we have a batch size composed of the number of instances (batch_size_real) * num_agents
                        x = torch.cat(([batch_list[b][n].x for b in range(batch_size_real)]), 0)
                        node_next_x = torch.cat(([batch_list[b][n].node_next_x for b in range(batch_size_real)]), 0)
                        node_id_original = torch.cat(([batch_list[b][n].node_id for b in range(batch_size_real)]), 0)
                        z_gnss = torch.cat(([batch_list[b][n].node_attr for b in range(batch_size_real)]), 0)
                        num_nodes = [batch_list[b][n].x.shape[0] for b in range(batch_size_real)]
                        # adjust indexes of edges for batch procedure
                        num_nodes[0] = 0
                        num_nodes = np.cumsum(num_nodes)
                        edge_index_original = torch.cat(([batch_list[b][n].edge_index for b in range(batch_size_real)]), 1)
                        edge_index = torch.cat(([batch_list[b][n].edge_index+num_nodes[b] for b in range(batch_size_real)]), 1)
                        z_inter_meas = torch.cat(([batch_list[b][n].edge_attr for b in range(batch_size_real)]), 0)

                        weights_x = batch_list[0][0].mean_std_min_max_variables['limit_features_x_max'] - batch_list[0][0].mean_std_min_max_variables['limit_features_x_min']
                        weights_x = 1/weights_x
                        weights_x = torch.tensor(weights_x/weights_x.sum()).to(device)

                        x_lstm_agents = []
                        outputs_dict_agents = []
                        encoded_z_inter_meas_agents = []
                        encoded_z_gnss_agents = []
                        latent_node_feats_agents = []
                        latent_edge_feats_agents = []
                        loss_agents = []
                        logs_agents = []

                        # LOCAL MODELS (ENCODING AND LSTM) 0
                        for agent in range(dataset_params['num_agents']):

                            # to recover the index of the nodes (for attributes) in the batch of graphs (all related to timestamp n)
                            node_indexes_related_to_agent = node_id_original == agent
                            edge_indexes_related_to_agent = edge_index_original[1,:] == agent

                            model[agent].train()
                
                            optimizer[agent].zero_grad()
                            
                            x_lstm = model[agent].lstm(x[node_indexes_related_to_agent], n).squeeze()
                            x_lstm_agents.append(x_lstm)
                        
                            encoded_z_inter_meas, encoded_z_gnss = model[agent].encoder(z_inter_meas[edge_indexes_related_to_agent], z_gnss[node_indexes_related_to_agent])
                            encoded_z_inter_meas_agents.append(encoded_z_inter_meas)
                            encoded_z_gnss_agents.append(encoded_z_gnss)

                            latent_node_feats = x_lstm
                            latent_edge_feats = encoded_z_inter_meas
                            latent_node_feats_agents.append(latent_node_feats)  
                            latent_edge_feats_agents.append(latent_edge_feats)

                            # During training, the feature vectors that the MPNetwork outputs for the  last self.num_class_steps message
                            # passing steps are classified in order to compute the loss.
                            first_class_step = model_params['T_message_steps'] - model_params['num_regress_steps'] + 1 # total 2 = 12  Number of message passing steps - # 11  Number of message passing steps during feature vectors are classified (after Message Passing) + 1
                            outputs_dict = {'x_hat': []}
                            outputs_dict_agents.append(outputs_dict)

                        for step in range(0, 1):

                            for agent in range(dataset_params['num_agents']):

                                node_indexes_related_to_agent = node_id_original == agent
                                edge_indexes_related_to_agent = edge_index_original[1,:] == agent
                    
                                # Reattach the initially encoded embeddings before the update
                                latent_edge_feats_agents[agent] = torch.cat((encoded_z_inter_meas_agents[agent], latent_edge_feats_agents[agent].squeeze()), dim=1)
                    
                                # Message Passing Step                           
                                # avoid backpropagate with outputs of other models   
                                latent_node_feats_agents_tmp = [latent_node_feats_agents[a] if a == agent else latent_node_feats_agents[a].clone().detach().requires_grad_(True) for a in range(dataset_params['num_agents'])]      
                                latent_node_feats_agents[agent], latent_edge_feats_agents[agent] = model[agent].MPNet(torch.cat(latent_node_feats_agents_tmp).squeeze(), x_lstm_agents[agent], encoded_z_gnss_agents[agent], edge_index[:,edge_indexes_related_to_agent], latent_edge_feats_agents[agent], node_indexes_related_to_agent, edge_indexes_related_to_agent)

                                # Regression Step              
                                _, dec_node_feats = model[agent].regressor(nodes_feats = latent_node_feats_agents[agent])
                                outputs_dict_agents[agent]['x_hat'].append(dec_node_feats)
                            
                        # At the end of message passing, evaluate models
                        for agent in range(dataset_params['num_agents']):
                            node_indexes_related_to_agent = node_id_original == agent
                            edge_indexes_related_to_agent = edge_index_original[1,:] == agent

                            # loss
                            loss = _compute_loss_single_agent_all_message_passing(device, weights_x, model_params, outputs_dict_agents[agent]['x_hat'], node_next_x[node_indexes_related_to_agent])

                            # COMPUTE GRADIENTS and BACK-PROPAGATE 2
                            loss.backward()

                            optimizer[agent].step()

                            loss_agents.append(loss)

                            model[agent].train()
                            optimizer[agent].zero_grad()     

                            # Compute performance metrics
                            metrics = compute_perform_metrics_single_agent(device, model_params, outputs_dict_agents[agent]['x_hat'], node_next_x[node_indexes_related_to_agent])
                            logs = {**metrics, **{'loss': loss_agents[agent].item()}}
                            logs_agents.append(logs)

                        # Mean RMSE
                        logs = {k: sum([logs_agents[agent][k] for agent in range(dataset_params['num_agents'])]) / dataset_params['num_agents']  if type(logs_agents[0][k]) is not list else [] for k in logs_agents[0]}
                        log = {key + f'/{type_}': val for key, val in logs.items()}

                        # CONSENSUS 3    
                        # Consensus
                        model_global = FedAvg([deepcopy(model[agent].state_dict()) for agent in range(dataset_params['num_agents'])])

                        # # Update global model
                        for agent in range(dataset_params['num_agents']):
                            model[agent].load_state_dict(model_global)  

                    logs_all.append(log)
                    batch_list = []

    return epoch_end(logs_all)


def _compute_loss_single_agent(device, weights_x, model_params, outputs, node_next_x):

    loss = 0

    # Prediction
    outputs_dict_MPNN = outputs[-1]

    # one single message passing steps
    loss += weighted_mse_loss_fun(node_next_x, outputs_dict_MPNN, weights = weights_x) * torch.tensor([model_params['mu_mpnn']], dtype=torch.float).to(device)  

    return loss

def _compute_loss_single_agent_all_message_passing(device, weights_x, model_params, outputs, node_next_x):
    
    loss = 0

    # Prediction
    outputs_dict_MPNN = outputs

    # all message passing steps
    for step in range(len(outputs_dict_MPNN)):
        loss += weighted_mse_loss_fun(node_next_x, outputs_dict_MPNN[step], weights = weights_x) * torch.tensor([model_params['mu_mpnn']], dtype=torch.float).to(device)  

    return loss

def compute_perform_metrics_single_agent(device, model_params, outputs, node_next_x):

    rmse_pos = []
    rmse_vel = []
    rmse_acc = []

    # Prediction
    outputs_dict_MPNN_x_hat =  outputs[-1].detach().cpu().numpy()

    # GT
    node_next_x = node_next_x.detach().cpu().numpy()
    
    # From LSTM-MPNN
    rmse_pos += [np.sqrt(np.sum((node_next_x[:,:2] - outputs_dict_MPNN_x_hat[:,:2])**2, 1))]
    if model_params['number_state_variables'] >=4:
        rmse_vel += [np.sqrt(np.sum((node_next_x[:,2:4] - outputs_dict_MPNN_x_hat[:,2:4])**2, 1))]
        if model_params['number_state_variables'] >=6:
            rmse_acc += [np.sqrt(np.sum((node_next_x[:,4:6] - outputs_dict_MPNN_x_hat[:,4:6])**2, 1))]

    rmse_pos = np.mean(rmse_pos)
    if model_params['number_state_variables'] >=4:
        rmse_vel = np.mean(rmse_vel)
        if model_params['number_state_variables'] >=6:
            rmse_acc = np.mean(rmse_acc)

    return {'rmse_pos':rmse_pos, 'rmse_vel':rmse_vel, 'rmse_acc':rmse_acc}

def _compute_loss_parallel(device, model_params, outputs, batch_list):
    
    mse_loss = nn.MSELoss()
    weights_x = batch_list[0][0].mean_std_min_max_variables['limit_features_x_max'] - batch_list[0][0].mean_std_min_max_variables['limit_features_x_min']
    weights_x = 1/weights_x
    weights_x = torch.tensor(weights_x/weights_x.sum()).to(device)
    
    batch_size_length_seq = model_params['batch_size_length_seq']
    batch_size_real = model_params['batch_size_real']
    T_message_steps = model_params['T_message_steps']
    loss = 0
    for n in range(batch_size_length_seq):

        # GT
        node_next_x = torch.cat(([batch_list[b][n].node_next_x for b in range(batch_size_real)]), 0)

        # Prediction
        output_lstm = outputs['output_lstm'][n].squeeze()
        outputs_dict_MPNN = outputs['outputs_dict_MPNN'][n]

        # all message passing steps
        for step in range(len(outputs_dict_MPNN['x_hat'])):
            loss += weighted_mse_loss_fun(node_next_x, outputs_dict_MPNN['x_hat'][step], weights = weights_x) * torch.tensor([model_params['mu_mpnn']], dtype=torch.float).to(device)

    return loss




def evaluate_parallel(dataset_params, model_params, device, model, loader, thr=None):

    if model_params['centralized']:

        type_ = 'val'
        model.eval()

        logs_all = []

        batch_size_real = model_params['batch_size_real']
        
        batch_list = []
        with torch.no_grad():
            for i,data in enumerate(loader):
                
                data = data.to(device)
                batch_list.append(data)

                if (i+1)%batch_size_real == 0:
    
                    output = model(batch_list, testing = 1)
                    
                    loss = _compute_loss_parallel(device, model_params, output, batch_list)
                    
                    metrics = compute_perform_metrics_parallel(device, model_params, output, batch_list)
                    
                    logs = {**metrics, **{'loss': loss.item()}}
                    
                    log = {key + f'/{type_}': val for key, val in logs.items()}

                    logs_all.append(log)         
       
    # distributed
    else:

        if model_params['distributed_strategy'] == 2:
            
            type_ = 'val'

            logs_all = []

            batch_size_real = model_params['batch_size_real']

            batch_list = []
            with torch.no_grad():
                for i,data in enumerate(loader):
                    
                    data = data.to(device)
                    batch_list.append(data)

                    if (i+1)%batch_size_real == 0:

                        # timestamp
                        for n in range(model_params['batch_size_length_seq']):

                            # reshape such that we have a batch size composed of the number of instances (batch_size_real) * num_agents
                            x = torch.cat(([batch_list[b][n].x for b in range(batch_size_real)]), 0)
                            node_next_x = torch.cat(([batch_list[b][n].node_next_x for b in range(batch_size_real)]), 0)
                            node_id_original = torch.cat(([batch_list[b][n].node_id for b in range(batch_size_real)]), 0)
                            z_gnss = torch.cat(([batch_list[b][n].node_attr for b in range(batch_size_real)]), 0)
                            num_nodes = [batch_list[b][n].x.shape[0] for b in range(batch_size_real)]
                            # adjust indexes of edges for batch procedure
                            num_nodes[0] = 0
                            num_nodes = np.cumsum(num_nodes)
                            edge_index_original = torch.cat(([batch_list[b][n].edge_index for b in range(batch_size_real)]), 1)
                            edge_index = torch.cat(([batch_list[b][n].edge_index+num_nodes[b] for b in range(batch_size_real)]), 1)
                            z_inter_meas = torch.cat(([batch_list[b][n].edge_attr for b in range(batch_size_real)]), 0)

                            weights_x = batch_list[0][0].mean_std_min_max_variables['limit_features_x_max'] - batch_list[0][0].mean_std_min_max_variables['limit_features_x_min']
                            weights_x = 1/weights_x
                            weights_x = torch.tensor(weights_x/weights_x.sum()).to(device)

                            x_lstm_agents = []
                            outputs_dict_agents = []
                            encoded_z_inter_meas_agents = []
                            encoded_z_gnss_agents = []
                            latent_node_feats_agents = []
                            latent_edge_feats_agents = []
                            loss_agents = []
                            logs_agents = []

                            for agent in range(dataset_params['num_agents']):

                                # to recover the index of the nodes (for attributes) in the batch of graphs (all related to timestamp n)
                                node_indexes_related_to_agent = node_id_original == agent
                                edge_indexes_related_to_agent = edge_index_original[1,:] == agent

                                model[agent].eval()
                                            
                                x_lstm = model[agent].lstm(x[node_indexes_related_to_agent], n, testing=1).squeeze()
                                x_lstm_agents.append(x_lstm)
                            
                                encoded_z_inter_meas, encoded_z_gnss = model[agent].encoder(z_inter_meas[edge_indexes_related_to_agent], z_gnss[node_indexes_related_to_agent])
                                encoded_z_inter_meas_agents.append(encoded_z_inter_meas)
                                encoded_z_gnss_agents.append(encoded_z_gnss)

                                latent_node_feats = x_lstm
                                latent_edge_feats = encoded_z_inter_meas
                                latent_node_feats_agents.append(latent_node_feats) 
                                latent_edge_feats_agents.append(latent_edge_feats)

                                # During training, the feature vectors that the MPNetwork outputs for the  last self.num_class_steps message
                                # passing steps are classified in order to compute the loss.
                                first_class_step = model_params['T_message_steps'] - model_params['num_regress_steps'] + 1 # total 2 = 12  Number of message passing steps - # 11  Number of message passing steps during feature vectors are classified (after Message Passing) + 1
                                outputs_dict = {'x_hat': []}
                                outputs_dict_agents.append(outputs_dict)

                            for step in range(1, model_params['T_message_steps'] + 1):

                                for agent in range(dataset_params['num_agents']):

                                    node_indexes_related_to_agent = node_id_original == agent
                                    edge_indexes_related_to_agent = edge_index_original[1,:] == agent
                        
                                    # Reattach the initially encoded embeddings before the update
                                    latent_edge_feats_agents[agent] = torch.cat((encoded_z_inter_meas_agents[agent], latent_edge_feats_agents[agent].squeeze()), dim=1)
                        
                                    # Message Passing Step                           
                                    # avoid backpropagate with outputs of other models     
                                    # latent_node_feats_agents_tmp = [latent_node_feats_agents[a] if a == agent else latent_node_feats_agents[a].clone().detach().requires_grad_(True) for a in range(dataset_params['num_agents'])]      
                                    latent_node_feats_agents[agent], latent_edge_feats_agents[agent] = model[agent].MPNet(torch.cat(latent_node_feats_agents).squeeze(), x_lstm_agents[agent], encoded_z_gnss_agents[agent], edge_index[:,edge_indexes_related_to_agent], latent_edge_feats_agents[agent], node_indexes_related_to_agent, edge_indexes_related_to_agent)

                                    # Regression Step              
                                    _, dec_node_feats = model[agent].regressor(nodes_feats = latent_node_feats_agents[agent])
                                    outputs_dict_agents[agent]['x_hat'].append(dec_node_feats)
                                
                                    loss = _compute_loss_single_agent(device, weights_x, model_params, outputs_dict_agents[agent]['x_hat'], node_next_x[node_indexes_related_to_agent])

                                    if step == model_params['T_message_steps']:
                                        loss_agents.append(loss)               

                            # At the end of message passing, evaluate models
                            for agent in range(dataset_params['num_agents']):
                                node_indexes_related_to_agent = node_id_original == agent
                                edge_indexes_related_to_agent = edge_index_original[1,:] == agent
                                # Compute performance metrics
                                metrics = compute_perform_metrics_single_agent(device, model_params, outputs_dict_agents[agent]['x_hat'], node_next_x[node_indexes_related_to_agent])
                                logs = {**metrics, **{'loss': loss_agents[agent].item()}}
                                logs_agents.append(logs)

                            # Mean RMSE
                            logs = {k: sum([logs_agents[agent][k] for agent in range(dataset_params['num_agents'])]) / dataset_params['num_agents']  if type(logs_agents[0][k]) is not list else [] for k in logs_agents[0]}
                            log = {key + f'/{type_}': val for key, val in logs.items()}
                            
                        logs_all.append(log)
                        batch_list = []

        elif model_params['distributed_strategy'] == 1:
            
            type_ = 'val'

            logs_all = []

            batch_size_real = model_params['batch_size_real']

            batch_list = []
            with torch.no_grad():
                for i,data in enumerate(loader):
                    
                    data = data.to(device)
                    batch_list.append(data)

                    if (i+1)%batch_size_real == 0:

                        # timestamp
                        for n in range(model_params['batch_size_length_seq']):

                            # reshape such that we have a batch size composed of the number of instances (batch_size_real) * num_agents
                            x = torch.cat(([batch_list[b][n].x for b in range(batch_size_real)]), 0)
                            node_next_x = torch.cat(([batch_list[b][n].node_next_x for b in range(batch_size_real)]), 0)
                            node_id_original = torch.cat(([batch_list[b][n].node_id for b in range(batch_size_real)]), 0)
                            z_gnss = torch.cat(([batch_list[b][n].node_attr for b in range(batch_size_real)]), 0)
                            num_nodes = [batch_list[b][n].x.shape[0] for b in range(batch_size_real)]
                            # adjust indexes of edges for batch procedure
                            num_nodes[0] = 0
                            num_nodes = np.cumsum(num_nodes)
                            edge_index_original = torch.cat(([batch_list[b][n].edge_index for b in range(batch_size_real)]), 1)
                            edge_index = torch.cat(([batch_list[b][n].edge_index+num_nodes[b] for b in range(batch_size_real)]), 1)
                            z_inter_meas = torch.cat(([batch_list[b][n].edge_attr for b in range(batch_size_real)]), 0)

                            weights_x = batch_list[0][0].mean_std_min_max_variables['limit_features_x_max'] - batch_list[0][0].mean_std_min_max_variables['limit_features_x_min']
                            weights_x = 1/weights_x
                            weights_x = torch.tensor(weights_x/weights_x.sum()).to(device)

                            x_lstm_agents = []
                            outputs_dict_agents = []
                            encoded_z_inter_meas_agents = []
                            encoded_z_gnss_agents = []
                            latent_node_feats_agents = []
                            latent_edge_feats_agents = []
                            loss_agents = []
                            logs_agents = []
                            gradients_agents = []

                            for agent in range(dataset_params['num_agents']):

                                # to recover the index of the nodes (for attributes) in the batch of graphs (all related to timestamp n)
                                node_indexes_related_to_agent = node_id_original == agent
                                edge_indexes_related_to_agent = edge_index_original[1,:] == agent

                                model[agent].eval()
                                
                                x_lstm = model[agent].lstm(x[node_indexes_related_to_agent], n).squeeze()
                                x_lstm_agents.append(x_lstm)
                            
                                encoded_z_inter_meas, encoded_z_gnss = model[agent].encoder(z_inter_meas[edge_indexes_related_to_agent], z_gnss[node_indexes_related_to_agent])
                                encoded_z_inter_meas_agents.append(encoded_z_inter_meas)
                                encoded_z_gnss_agents.append(encoded_z_gnss)

                                latent_node_feats = x_lstm
                                latent_edge_feats = encoded_z_inter_meas
                                latent_node_feats_agents.append(latent_node_feats) 
                                latent_edge_feats_agents.append(latent_edge_feats)

                                # During training, the feature vectors that the MPNetwork outputs for the  last self.num_class_steps message
                                # passing steps are classified in order to compute the loss.
                                first_class_step = model_params['T_message_steps'] - model_params['num_regress_steps'] + 1 # total 2 = 12  Number of message passing steps - # 11  Number of message passing steps during feature vectors are classified (after Message Passing) + 1
                                outputs_dict = {'x_hat': []}
                                outputs_dict_agents.append(outputs_dict)

                            for step in range(1, model_params['T_message_steps'] + 1):

                                for agent in range(dataset_params['num_agents']):

                                    # store input
                                    if step == 1:
                                        original_latent_node_feats_agents = latent_node_feats_agents[agent]
                                        original_encoded_z_inter_meas_agents = encoded_z_inter_meas_agents[agent]
                                        original_latent_edge_feats_agents = latent_edge_feats_agents[agent]

                                    node_indexes_related_to_agent = node_id_original == agent
                                    edge_indexes_related_to_agent = edge_index_original[1,:] == agent
                        
                                    # Reattach the initially encoded embeddings before the update
                                    latent_edge_feats_agents[agent] = torch.cat((encoded_z_inter_meas_agents[agent], latent_edge_feats_agents[agent].squeeze()), dim=1)
                        
                                    # Message Passing Step                           
                                    # avoid backpropagate with outputs of other models     
                                    # latent_node_feats_agents_tmp = [latent_node_feats_agents[a] if a == agent else latent_node_feats_agents[a].clone().detach().requires_grad_(True) for a in range(dataset_params['num_agents'])]      
                                    latent_node_feats_agents[agent], latent_edge_feats_agents[agent] = model[agent].MPNet(torch.cat(latent_node_feats_agents).squeeze(), x_lstm_agents[agent], encoded_z_gnss_agents[agent], edge_index[:,edge_indexes_related_to_agent], latent_edge_feats_agents[agent], node_indexes_related_to_agent, edge_indexes_related_to_agent)

                                    
                                    # Regression Step              
                                    _, dec_node_feats = model[agent].regressor(nodes_feats = latent_node_feats_agents[agent])
                                    outputs_dict_agents[agent]['x_hat'].append(dec_node_feats)
                                
                                    loss = _compute_loss_single_agent(device, weights_x, model_params, outputs_dict_agents[agent]['x_hat'], node_next_x[node_indexes_related_to_agent])

                                    if step == model_params['T_message_steps']:
                                        loss_agents.append(loss)     

                            # At the end of message passing, evaluate models
                            for agent in range(dataset_params['num_agents']):
                                node_indexes_related_to_agent = node_id_original == agent
                                edge_indexes_related_to_agent = edge_index_original[1,:] == agent
                                # Compute performance metrics
                                metrics = compute_perform_metrics_single_agent(device, model_params, outputs_dict_agents[agent]['x_hat'], node_next_x[node_indexes_related_to_agent])
                                logs = {**metrics, **{'loss': loss_agents[agent].item()}}
                                logs_agents.append(logs)

                            # Mean RMSE
                            logs = {k: sum([logs_agents[agent][k] for agent in range(dataset_params['num_agents'])]) / dataset_params['num_agents']  if type(logs_agents[0][k]) is not list else [] for k in logs_agents[0]}
                            log = {key + f'/{type_}': val for key, val in logs.items()}                           

                        logs_all.append(log)
                        batch_list = []

        elif model_params['distributed_strategy'] == 3:

            type_ = 'val'

            logs_all = []

            batch_size_real = model_params['batch_size_real']

            batch_list = []
            with torch.no_grad():
                for i,data in enumerate(loader):
                    
                    data = data.to(device)
                    batch_list.append(data)

                    if (i+1)%batch_size_real == 0:

                        # timestamp
                        for n in range(model_params['batch_size_length_seq']):

                            # reshape such that we have a batch size composed of the number of instances (batch_size_real) * num_agents
                            x = torch.cat(([batch_list[b][n].x for b in range(batch_size_real)]), 0)
                            node_next_x = torch.cat(([batch_list[b][n].node_next_x for b in range(batch_size_real)]), 0)
                            node_id_original = torch.cat(([batch_list[b][n].node_id for b in range(batch_size_real)]), 0)
                            z_gnss = torch.cat(([batch_list[b][n].node_attr for b in range(batch_size_real)]), 0)
                            num_nodes = [batch_list[b][n].x.shape[0] for b in range(batch_size_real)]
                            # adjust indexes of edges for batch procedure
                            num_nodes[0] = 0
                            num_nodes = np.cumsum(num_nodes)
                            edge_index_original = torch.cat(([batch_list[b][n].edge_index for b in range(batch_size_real)]), 1)
                            edge_index = torch.cat(([batch_list[b][n].edge_index+num_nodes[b] for b in range(batch_size_real)]), 1)
                            z_inter_meas = torch.cat(([batch_list[b][n].edge_attr for b in range(batch_size_real)]), 0)

                            weights_x = batch_list[0][0].mean_std_min_max_variables['limit_features_x_max'] - batch_list[0][0].mean_std_min_max_variables['limit_features_x_min']
                            weights_x = 1/weights_x
                            weights_x = torch.tensor(weights_x/weights_x.sum()).to(device)

                            x_lstm_agents = []
                            outputs_dict_agents = []
                            encoded_z_inter_meas_agents = []
                            encoded_z_gnss_agents = []
                            latent_node_feats_agents = []
                            latent_edge_feats_agents = []
                            loss_agents = []
                            logs_agents = []

                            for agent in range(dataset_params['num_agents']):

                                # to recover the index of the nodes (for attributes) in the batch of graphs (all related to timestamp n)
                                node_indexes_related_to_agent = node_id_original == agent
                                edge_indexes_related_to_agent = edge_index_original[1,:] == agent

                                model[agent].eval()
                                
                                x_lstm = model[agent].lstm(x[node_indexes_related_to_agent], n).squeeze()
                                x_lstm_agents.append(x_lstm)
                            
                                encoded_z_inter_meas, encoded_z_gnss = model[agent].encoder(z_inter_meas[edge_indexes_related_to_agent], z_gnss[node_indexes_related_to_agent])
                                encoded_z_inter_meas_agents.append(encoded_z_inter_meas)
                                encoded_z_gnss_agents.append(encoded_z_gnss)

                                latent_node_feats = x_lstm
                                latent_edge_feats = encoded_z_inter_meas
                                latent_node_feats_agents.append(latent_node_feats) 
                                latent_edge_feats_agents.append(latent_edge_feats)

                                # During training, the feature vectors that the MPNetwork outputs for the  last self.num_class_steps message
                                # passing steps are classified in order to compute the loss.
                                first_class_step = model_params['T_message_steps'] - model_params['num_regress_steps'] + 1 # total 2 = 12  Number of message passing steps - # 11  Number of message passing steps during feature vectors are classified (after Message Passing) + 1
                                outputs_dict = {'x_hat': []}
                                outputs_dict_agents.append(outputs_dict)

                            for step in range(1, model_params['T_message_steps'] + 1):

                                for agent in range(dataset_params['num_agents']):

                                    node_indexes_related_to_agent = node_id_original == agent
                                    edge_indexes_related_to_agent = edge_index_original[1,:] == agent
                        
                                    # Reattach the initially encoded embeddings before the update
                                    latent_edge_feats_agents[agent] = torch.cat((encoded_z_inter_meas_agents[agent], latent_edge_feats_agents[agent].squeeze()), dim=1)
                        
                                    # Message Passing Step                           
                                    # avoid backpropagate with outputs of other models     
                                    # latent_node_feats_agents_tmp = [latent_node_feats_agents[a] if a == agent else latent_node_feats_agents[a].clone().detach().requires_grad_(True) for a in range(dataset_params['num_agents'])]      
                                    latent_node_feats_agents[agent], latent_edge_feats_agents[agent] = model[agent].MPNet(torch.cat(latent_node_feats_agents).squeeze(), x_lstm_agents[agent], encoded_z_gnss_agents[agent], edge_index[:,edge_indexes_related_to_agent], latent_edge_feats_agents[agent], node_indexes_related_to_agent, edge_indexes_related_to_agent)

                                    
                                    # Regression Step              
                                    _, dec_node_feats = model[agent].regressor(nodes_feats = latent_node_feats_agents[agent])
                                    outputs_dict_agents[agent]['x_hat'].append(dec_node_feats)
                                

                            # At the end of message passing, evaluate models
                            for agent in range(dataset_params['num_agents']):
                                node_indexes_related_to_agent = node_id_original == agent
                                edge_indexes_related_to_agent = edge_index_original[1,:] == agent

                                # loss
                                loss = _compute_loss_single_agent_all_message_passing(device, weights_x, model_params, outputs_dict_agents[agent]['x_hat'], node_next_x[node_indexes_related_to_agent])

                                loss_agents.append(loss)

                                model[agent].eval()

                                # Compute performance metrics
                                metrics = compute_perform_metrics_single_agent(device, model_params, outputs_dict_agents[agent]['x_hat'], node_next_x[node_indexes_related_to_agent])
                                logs = {**metrics, **{'loss': loss_agents[agent].item()}}
                                logs_agents.append(logs)

                            # Mean RMSE
                            logs = {k: sum([logs_agents[agent][k] for agent in range(dataset_params['num_agents'])]) / dataset_params['num_agents']  if type(logs_agents[0][k]) is not list else [] for k in logs_agents[0]}
                            log = {key + f'/{type_}': val for key, val in logs.items()}
                                
                        logs_all.append(log)
                        batch_list = []

        # Consensus -> just 1 iteration of MPNN in the forward part
        elif model_params['distributed_strategy'] == 4:

            type_ = 'val'

            logs_all = []

            batch_size_real = model_params['batch_size_real']

            batch_list = []
            with torch.no_grad():
                for i,data in enumerate(loader):
                    
                    data = data.to(device)
                    batch_list.append(data)

                    if (i+1)%batch_size_real == 0:

                        # timestamp
                        for n in range(model_params['batch_size_length_seq']):

                            # reshape such that we have a batch size composed of the number of instances (batch_size_real) * num_agents
                            x = torch.cat(([batch_list[b][n].x for b in range(batch_size_real)]), 0)
                            node_next_x = torch.cat(([batch_list[b][n].node_next_x for b in range(batch_size_real)]), 0)
                            node_id_original = torch.cat(([batch_list[b][n].node_id for b in range(batch_size_real)]), 0)
                            z_gnss = torch.cat(([batch_list[b][n].node_attr for b in range(batch_size_real)]), 0)
                            num_nodes = [batch_list[b][n].x.shape[0] for b in range(batch_size_real)]
                            # adjust indexes of edges for batch procedure
                            num_nodes[0] = 0
                            num_nodes = np.cumsum(num_nodes)
                            edge_index_original = torch.cat(([batch_list[b][n].edge_index for b in range(batch_size_real)]), 1)
                            edge_index = torch.cat(([batch_list[b][n].edge_index+num_nodes[b] for b in range(batch_size_real)]), 1)
                            z_inter_meas = torch.cat(([batch_list[b][n].edge_attr for b in range(batch_size_real)]), 0)

                            weights_x = batch_list[0][0].mean_std_min_max_variables['limit_features_x_max'] - batch_list[0][0].mean_std_min_max_variables['limit_features_x_min']
                            weights_x = 1/weights_x
                            weights_x = torch.tensor(weights_x/weights_x.sum()).to(device)

                            x_lstm_agents = []
                            outputs_dict_agents = []
                            encoded_z_inter_meas_agents = []
                            encoded_z_gnss_agents = []
                            latent_node_feats_agents = []
                            latent_edge_feats_agents = []
                            loss_agents = []
                            logs_agents = []

                            for agent in range(dataset_params['num_agents']):

                                # to recover the index of the nodes (for attributes) in the batch of graphs (all related to timestamp n)
                                node_indexes_related_to_agent = node_id_original == agent
                                edge_indexes_related_to_agent = edge_index_original[1,:] == agent

                                model[agent].eval()
                                
                                x_lstm = model[agent].lstm(x[node_indexes_related_to_agent], n).squeeze()
                                x_lstm_agents.append(x_lstm)
                            
                                encoded_z_inter_meas, encoded_z_gnss = model[agent].encoder(z_inter_meas[edge_indexes_related_to_agent], z_gnss[node_indexes_related_to_agent])
                                encoded_z_inter_meas_agents.append(encoded_z_inter_meas)
                                encoded_z_gnss_agents.append(encoded_z_gnss)

                                latent_node_feats = x_lstm
                                latent_edge_feats = encoded_z_inter_meas
                                latent_node_feats_agents.append(latent_node_feats) 
                                latent_edge_feats_agents.append(latent_edge_feats)

                                # During training, the feature vectors that the MPNetwork outputs for the  last self.num_class_steps message
                                # passing steps are classified in order to compute the loss.
                                first_class_step = model_params['T_message_steps'] - model_params['num_regress_steps'] + 1 # total 2 = 12  Number of message passing steps - # 11  Number of message passing steps during feature vectors are classified (after Message Passing) + 1
                                outputs_dict = {'x_hat': []}
                                outputs_dict_agents.append(outputs_dict)

                            for step in range(0, 1):

                                for agent in range(dataset_params['num_agents']):

                                    node_indexes_related_to_agent = node_id_original == agent
                                    edge_indexes_related_to_agent = edge_index_original[1,:] == agent
                        
                                    # Reattach the initially encoded embeddings before the update
                                    latent_edge_feats_agents[agent] = torch.cat((encoded_z_inter_meas_agents[agent], latent_edge_feats_agents[agent].squeeze()), dim=1)
                        
                                    # Message Passing Step                           
                                    # avoid backpropagate with outputs of other models     
                                    latent_node_feats_agents[agent], latent_edge_feats_agents[agent] = model[agent].MPNet(torch.cat(latent_node_feats_agents).squeeze(), x_lstm_agents[agent], encoded_z_gnss_agents[agent], edge_index[:,edge_indexes_related_to_agent], latent_edge_feats_agents[agent], node_indexes_related_to_agent, edge_indexes_related_to_agent)
 
                                    # Regression Step              
                                    _, dec_node_feats = model[agent].regressor(nodes_feats = latent_node_feats_agents[agent])
                                    outputs_dict_agents[agent]['x_hat'].append(dec_node_feats)
                                
       

                            # At the end of message passing, evaluate models
                            for agent in range(dataset_params['num_agents']):
                                node_indexes_related_to_agent = node_id_original == agent
                                edge_indexes_related_to_agent = edge_index_original[1,:] == agent

                                # loss
                                loss = _compute_loss_single_agent_all_message_passing(device, weights_x, model_params, outputs_dict_agents[agent]['x_hat'], node_next_x[node_indexes_related_to_agent])

                                loss_agents.append(loss)

                                model[agent].eval()

                                # Compute performance metrics
                                metrics = compute_perform_metrics_single_agent(device, model_params, outputs_dict_agents[agent]['x_hat'], node_next_x[node_indexes_related_to_agent])
                                logs = {**metrics, **{'loss': loss_agents[agent].item()}}
                                logs_agents.append(logs)

                            # Mean RMSE
                            logs = {k: sum([logs_agents[agent][k] for agent in range(dataset_params['num_agents'])]) / dataset_params['num_agents']  if type(logs_agents[0][k]) is not list else [] for k in logs_agents[0]}
                            log = {key + f'/{type_}': val for key, val in logs.items()}
                                
                        logs_all.append(log)
                        batch_list = []

    return epoch_end(logs_all)



def compute_perform_metrics_parallel(device, model_params, outputs, batch_list):

    batch_size_length_seq = model_params['batch_size_length_seq']
    batch_size_real = model_params['batch_size_real']
    T_message_steps = model_params['T_message_steps']
    rmse_pos = []
    rmse_vel = []
    rmse_acc = []
    for n in range(batch_size_length_seq):

        # GT
        node_next_x = np.concatenate(([batch_list[b][n].node_next_x.detach().cpu().numpy() for b in range(batch_size_real)]), 0)

        # Prediction
        output_lstm = outputs['output_lstm'][n].squeeze().detach().cpu().numpy()
        outputs_dict_MPNN_x_hat = outputs['outputs_dict_MPNN'][n]['x_hat']

        # Log
        if model_params['log_MPNN_LSTM']:
            agent = 0
            print('Input x not normalized: a {} {}'.format(agent,  batch_list[0][n].x_not_normalized.detach().cpu().numpy()[agent,:]))
            print('Input x: a {} {}'.format(agent,  batch_list[0][n].x.detach().cpu().numpy()[0,:]))
            print('GNSS z: a {} {}'.format(agent,  batch_list[0][n].node_attr_not_normalized.detach().cpu().numpy()[agent,:]))
            print('Real x_next: a {} {}'.format(agent,batch_list[0][n].node_next_x.detach().cpu().numpy()[agent,:]))
            print('Prediction x_next: a {} {}'.format(agent,output_lstm[0,:]))
            print('Update x_next: a {} {}\n'.format(agent,outputs_dict_MPNN_x_hat[-1][agent,:].detach().cpu().numpy()))
        
        # From LSTM-MPNN
        rmse_pos += [np.sqrt(np.sum((node_next_x[:,:2] - outputs_dict_MPNN_x_hat[-1].detach().cpu().numpy()[:,:2])**2, 1))]
        if model_params['number_state_variables'] >=4:
            rmse_vel += [np.sqrt(np.sum((node_next_x[:,2:4] - outputs_dict_MPNN_x_hat[-1].detach().cpu().numpy()[:,2:4])**2, 1))]
            if model_params['number_state_variables'] >=6:
                rmse_acc += [np.sqrt(np.sum((node_next_x[:,4:6] - outputs_dict_MPNN_x_hat[-1].detach().cpu().numpy()[:,4:6])**2, 1))]

    rmse_pos = np.mean(rmse_pos)
    if model_params['number_state_variables'] >=4:
        rmse_vel = np.mean(rmse_vel)
        if model_params['number_state_variables'] >=6:
            rmse_acc = np.mean(rmse_acc)

    return {'rmse_pos':rmse_pos, 'rmse_vel':rmse_vel, 'rmse_acc':rmse_acc}








































































































































































































































































































































































































































































































































