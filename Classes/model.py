import torch
from torch import nn
from torch_scatter import scatter_mean, scatter_max, scatter_add
import numpy as np
from copy import copy, deepcopy
from functools import lru_cache
import asyncio

# Parallel functions
def background(f):
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)

    return wrapped

class MLP(nn.Module):
    def __init__(self, input_dim, fc_dims, dropout_p=0.4, use_batchnorm=False):
        super(MLP, self).__init__()

        assert isinstance(fc_dims, (list, tuple)), 'fc_dims must be either a list or a tuple, but got {}'.format(
            type(fc_dims))

        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            if use_batchnorm and dim != 1:
                layers.append(nn.BatchNorm1d(dim))

            if dim != 1:
                layers.append(nn.ReLU(inplace=False))

            if dropout_p is not None and dim != 1:
                layers.append(nn.Dropout(p=dropout_p))

            input_dim = dim

        self.fc_layers = nn.Sequential(*layers)

    def forward(self, input):
        return self.fc_layers(input)# Models

class MLP_sparse(nn.Module):
    def __init__(self, input_dim, fc_dims, dropout_p=0.4, use_batchnorm=False):
        super(MLP_sparse, self).__init__()

        assert isinstance(fc_dims, (list, tuple)), 'fc_dims must be either a list or a tuple, but got {}'.format(
            type(fc_dims))

        layers = []
        # layers.append(nn.Linear(input_dim, 2*input_dim))
        # layers.append(Maxout(2*input_dim, input_dim, 2))
        for i, dim in enumerate(fc_dims):
            layers.append(nn.Linear(input_dim, dim))
            if use_batchnorm and dim != 1 and i!=(len(fc_dims)-1):
                layers.append(nn.BatchNorm1d(dim))

            if dim != 1 and i!=(len(fc_dims)-1):
                layers.append(nn.ReLU(inplace=False))
                # layers.append(nn.GELU())

            if dropout_p is not None and dim != 1 and i!=(len(fc_dims)-1):
                layers.append(nn.Dropout(p=dropout_p))

            input_dim = dim

        self.fc_layers = nn.Sequential(*layers)

    def forward(self, input):
        return self.fc_layers(input)# Models

class Maxout(nn.Module):
    def __init__(self, d_in, d_out, pool_size):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.pool_size = pool_size
        self.lin = nn.Linear(d_in, d_out * pool_size)

    def forward(self, x):
        shape = list(x.shape)
        shape[-1] = self.d_out
        shape.append(self.pool_size)
        x = self.lin(x)
        x = x.view(*shape)
        x = x.max(-1)[0]
        return x

# Building models
# 2 encoder MLPs (1 for nodes (only second part), 1 for edges) that provide the initial node and edge embeddings 
# final binary classifier for edges
class MLPGraphIndependent(nn.Module):
    """
    Class used to to encode (resp. classify) features before (resp. after) neural message passing.
    It consists of two MLPs, one for nodes and one for edges, and they are applied independently to node and edge
    features, respectively.
    """

    def __init__(self, edge_in_dim = None, node_in_dim = None, edge_out_dim = None, node_out_dim = None,
                 node_fc_dims = None, edge_fc_dims = None, dropout_p = None, use_batchnorm = None):
        super(MLPGraphIndependent, self).__init__()

        if node_in_dim is not None :
            self.node_mlp = MLP(input_dim=node_in_dim, fc_dims=list(node_fc_dims) + [node_out_dim],
                                dropout_p=dropout_p, use_batchnorm=use_batchnorm)
        else:
            self.node_mlp = None

        if edge_in_dim is not None :
            self.edge_mlp = MLP(input_dim=edge_in_dim, fc_dims=list(edge_fc_dims) + [edge_out_dim],
                                dropout_p=dropout_p, use_batchnorm=use_batchnorm)
        else:
            self.edge_mlp = None

    def forward(self, edge_feats = None, nodes_feats = None):

        if self.node_mlp is not None:
            out_node_feats = self.node_mlp(nodes_feats)

        else:
            out_node_feats = nodes_feats

        if self.edge_mlp is not None:
            out_edge_feats = self.edge_mlp(edge_feats)

        else:
            out_edge_feats = edge_feats

        return out_edge_feats, out_node_feats

class MLPGraphIndependent_sparse(nn.Module):
    """
    Class used to to encode (resp. classify) features before (resp. after) neural message passing.
    It consists of two MLPs, one for nodes and one for edges, and they are applied independently to node and edge
    features, respectively.
    """

    def __init__(self, edge_in_dim = None, node_in_dim = None, edge_out_dim = None, node_out_dim = None,
                 node_fc_dims = None, edge_fc_dims = None, dropout_p = None, use_batchnorm = None):
        super(MLPGraphIndependent_sparse, self).__init__()

        if node_in_dim is not None :
            self.node_mlp = MLP_sparse(input_dim=node_in_dim, fc_dims=list(node_fc_dims) + [node_out_dim],
                                dropout_p=dropout_p, use_batchnorm=use_batchnorm)
        else:
            self.node_mlp = None

        if edge_in_dim is not None :
            self.edge_mlp = MLP_sparse(input_dim=edge_in_dim, fc_dims=list(edge_fc_dims) + [edge_out_dim],
                                dropout_p=dropout_p, use_batchnorm=use_batchnorm)
        else:
            self.edge_mlp = None

    def forward(self, edge_feats = None, nodes_feats = None):

        if self.node_mlp is not None:
            out_node_feats = self.node_mlp(nodes_feats)

        else:
            out_node_feats = nodes_feats

        if self.edge_mlp is not None:
            out_edge_feats = self.edge_mlp(edge_feats)

        else:
            out_edge_feats = edge_feats

        return out_edge_feats, out_node_feats

# To perform forward of edge model used in the 'core' Message Passing Network
class EdgeModel(nn.Module):
    """
    Class used to peform the edge update during Neural message passing
    """
    def __init__(self, edge_mlp):
        super(EdgeModel, self).__init__()
        self.edge_mlp = edge_mlp

    def forward(self, source, target, edge_attr):
        # E X 96 = (E X 32) (E X 32) (E X 16)
        out = torch.cat([source, target, edge_attr], dim=1)
        
        # return      E X 16
        return self.edge_mlp(out)


# To perform forward of node models:
# 2 node update (fut, past) + 1 node update model
class TimeAwareNodeModel(nn.Module):
    """
    Class used to peform the node update during Neural mwssage passing
    """
    def __init__(self, flow_in_mlp, flow_out_mlp, node_mlp, node_agg_fn):
        super(TimeAwareNodeModel, self).__init__()

        self.flow_in_mlp = flow_in_mlp   # in 48, out 32
        self.flow_out_mlp = flow_out_mlp # in 48, out 32
        self.node_mlp = node_mlp         # in 64, out 32
        self.node_agg_fn = node_agg_fn # 'sum'

                    # (N X 32)  (2 X E) (E X 16)
    def forward(self, x, edge_index, edge_attr): # eg N=915, E=93772, FEAT_EDGES=16, FEAT_NODES=32
        row, col = edge_index
        flow_out_mask = row < col
        flow_out_row, flow_out_col = row[flow_out_mask], col[flow_out_mask]
        #  (E/2 X 48)           # (E/2 X 32)         (E/2 X 16)
        flow_out_input = torch.cat([x[flow_out_col], edge_attr[flow_out_mask]], dim=1)                                
        flow_out = self.flow_out_mlp(flow_out_input) # eg flow_out_input = E/2 X 48
        # N X 32                  # (E/2 X 32)     E/2          N 
        flow_out = self.node_agg_fn(flow_out, flow_out_row, x.size(0))

        flow_in_mask = row > col
        flow_in_row, flow_in_col = row[flow_in_mask], col[flow_in_mask]
        flow_in_input = torch.cat([x[flow_in_col], edge_attr[flow_in_mask]], dim=1)
        flow_in = self.flow_in_mlp(flow_in_input)
        flow_in = self.node_agg_fn(flow_in, flow_in_row, x.size(0))
        
        # (N X 64)        (N X 32) (N X 32)
        flow = torch.cat((flow_in, flow_out), dim=1)
        
                # return (N X 32)
        return self.node_mlp(flow)


class MetaLayer(torch.nn.Module):
    """
    Core Message Passing Network Class. Extracted from torch_geometric, with minor modifications.
    (https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html)
    """
    def __init__(self, edge_model=None, node_model=None):
        """
        Args:
            edge_model: Callable Edge Update Model
            node_model: Callable Node Update Model
        """
        super(MetaLayer, self).__init__()

        self.edge_model = edge_model
        self.node_model = node_model
        self.reset_parameters()

    def reset_parameters(self):
        for item in [self.node_model, self.edge_model]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        """
        Does a single node and edge feature vectors update.
        Args:
            x: node features matrix
            edge_index: tensor with shape [2, M], with M being the number of edges, indicating nonzero entries in the
            graph adjacency (i.e. edges)
            edge_attr: edge features matrix (ordered by edge_index)

        Returns: Updated Node and Edge Feature matrices

        """
        row, col = edge_index  # row = first row, col = second row

        # Edge Update
        if self.edge_model is not None:#(E X 32) (E X 32)      (E X 32)
            edge_attr = self.edge_model(x[row],        x[col],       edge_attr)

        # Node Update
        if self.node_model is not None:
             #                (N X 32)  (2 X E)     (E X 16)
            x = self.node_model(x,      edge_index, edge_attr)

        return x, edge_attr

    def __repr__(self):
        return '{}(edge_model={}, node_model={})'.format(self.__class__.__name__, self.edge_model, self.node_model)


class Net(torch.nn.Module):
    """
    Main Model Class. Contains all the components of the model. It consists of of several networks:
    - 2 encoder MLPs (1 for nodes, 1 for edges) that provide the initial node and edge embeddings, respectively,
    - 4 update MLPs (3 for nodes, 1 per edges) used in the 'core' Message Passing Network
    - 1 edge classifier MLP that performs binary classification over the Message Passing Network's output.

    This class was initially based on: https://github.com/deepmind/graph_nets tensorflow implementation.
    """
    def __init__(self, params= None):
        super(Net, self).__init__()
        

        self.node_cnn = None # CNN used to encode bounding box apperance information.

        # Assign paramters
        self.params = params
        if isinstance(self.params, dict):
            for key, value in self.params.items():
                setattr(self, key, value)
 
        # Define Encoder and Classifier Networks
        # parameters
   
        # Change to encode directly positions of 8 angles
        # encoder_feats_dict = {'edge_in_dim': 6, 'edge_fc_dims': [18, 18], 'edge_out_dim': 16, 'node_in_dim': 2048, 'node_fc_dims': [128], 'node_out_dim': 32, 'dropout_p': 0, 'use_batchnorm': False}
        encoder_feats_dict = {'edge_in_dim': 6, 'edge_fc_dims': [18, 18], 'edge_out_dim': 16, 'node_in_dim': 3*8, 'node_fc_dims': [72], 'node_out_dim': 32, 'dropout_p': 0, 'use_batchnorm': False}
        classifier_feats_dict = {'edge_in_dim': 16, 'edge_fc_dims': [8], 'edge_out_dim': 1, 'dropout_p': 0, 'use_batchnorm': False}

        # 2 encoder MLPs (1 for nodes (only second part), 1 for edges) that provide the initial node and edge embeddings 
        self.encoder = MLPGraphIndependent(**encoder_feats_dict)
        # final binary classifier for edges
        self.classifier = MLPGraphIndependent(**classifier_feats_dict)

        # Define the 'Core' message passing network (i.e. node and edge update models)
        self.MPNet = self._build_core_MPNet(encoder_feats_dict=encoder_feats_dict)

        if (self.params is None) or ("num_enc_steps" not in self.params):  
            self.num_enc_steps = 1#12 #  Number of message passing steps
            self.num_class_steps = 1#11 #  Number of message passing steps during feature vectors are classified (after Message Passing)


    # Building core network, called in __init__
    # - 4 update MLPs (3 for nodes, 1 per edges) used in the 'core' Message Passing Network
    def _build_core_MPNet(self, encoder_feats_dict):
        """
        Builds the core part of the Message Passing Network: Node Update and Edge Update models.
        Args:
            encoder_feats_dict: dictionary containing the hyperparameters for the initial node/edge encoder
        """

        # Define an aggregation operator for nodes to 'gather' messages from incident edges
        node_agg_fn = 'sum'
        assert node_agg_fn.lower() in ('mean', 'max', 'sum'), "node_agg_fn can only be 'max', 'mean' or 'sum'."

        if node_agg_fn == 'mean':
            node_agg_fn = lambda out, row, x_size: scatter_mean(out, row, dim=0, dim_size=x_size)

        elif node_agg_fn == 'max':
            node_agg_fn = lambda out, row, x_size: scatter_max(out, row, dim=0, dim_size=x_size)[0]

        elif node_agg_fn == 'sum':
            node_agg_fn = lambda out, row, x_size: scatter_add(out, row, dim=0, dim_size=x_size)

        # Define all multi-layer perceptrons (MLPs) involved in the graph network
        # For both nodes and edges, the initial encoded features (i.e. output of self.encoder) can either be
        # reattached or not after each Message Passing Step. This affects MLPs input dimensions
        self.reattach_initial_nodes = False # Determines whether initially encoded node feats are used during node updates
        self.reattach_initial_edges = True  # Determines whether initially encoded edge feats are used during node updates

        edge_factor = 2 if self.reattach_initial_edges else 1  # 2
        node_factor = 2 if self.reattach_initial_nodes else 1  # 1
                          # 1 * 2 * 32 + 2 * 16
        # 96
        edge_model_in_dim = node_factor * 2 * encoder_feats_dict['node_out_dim'] + edge_factor * encoder_feats_dict[
            'edge_out_dim'] 
        # 48
        node_model_in_dim = node_factor * encoder_feats_dict['node_out_dim'] + encoder_feats_dict['edge_out_dim']


        # Define all MLPs used within the MPN
        edge_mlp = MLP(input_dim=96,
                       fc_dims=[80, 16],
                       dropout_p=0,
                       use_batchnorm=0)

        flow_in_mlp = MLP(input_dim=48,
                          fc_dims=[56, 32],
                          dropout_p=0,
                          use_batchnorm=0)

        flow_out_mlp = MLP(input_dim=48,
                           fc_dims=[56, 32],
                           dropout_p=0,
                           use_batchnorm=0)

        node_mlp = nn.Sequential(*[nn.Linear(2 * encoder_feats_dict['node_out_dim'], # 64
                                 encoder_feats_dict['node_out_dim']), # 32
                                   nn.ReLU(inplace=True)])

        # Define all MLPs used within the MPN
        return MetaLayer(edge_model=EdgeModel(edge_mlp = edge_mlp), # edge update model
                                                                     # 2 node update (fut, past) + 1 node update model
                         node_model=TimeAwareNodeModel(flow_in_mlp = flow_in_mlp,  # in 48, out 32
                                                       flow_out_mlp = flow_out_mlp, # in 48, out 32
                                                       node_mlp = node_mlp, # in 64, out 32
                                                       node_agg_fn = node_agg_fn)) # aggregation function (sum)
        

    def forward(self, data):
        """
        Provides a fractional solution to the data association problem.
        First, node and edge features are independently encoded by the encoder network. Then, they are iteratively
        'combined' for a fixed number of steps via the Message Passing Network (self.MPNet). Finally, they are
        classified independently by the classifiernetwork.
        Args:
            data: object containing attribues
              - x: node features matrix
              - edge_index: tensor with shape [2, M], with M being the number of edges, indicating nonzero entries in the
                graph adjacency (i.e. edges) (i.e. sparse adjacency)
              - edge_attr: edge features matrix (sorted by edge apperance in edge_index) #edges X #edge_features (6)

        Returns:
            classified_edges: list of unnormalized node probabilites after each MP step
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # First encoding of node images of boxes/features
        x_is_img = len(x.shape) == 4
        if self.node_cnn is not None and x_is_img:
            x = self.node_cnn(x)

            emb_dists = nn.functional.pairwise_distance(x[edge_index[0]], x[edge_index[1]]).view(-1, 1)
            edge_attr = torch.cat((edge_attr, emb_dists), dim = 1)

        # Encoding features step
        latent_edge_feats, latent_node_feats = self.encoder(edge_attr, x) # edge_attr (E X 6), x are the node features (N X 2048) -> change to (N X 8)
        # latent_edge_feats (E X 16)
        # latent_node_feats (N X 32)
        
        initial_edge_feats = latent_edge_feats
        initial_node_feats = latent_node_feats

        # During training, the feature vectors that the MPNetwork outputs for the  last self.num_class_steps message
        # passing steps are classified in order to compute the loss.
        first_class_step = self.num_enc_steps - self.num_class_steps + 1 # total 2 = 12  Number of message passing steps - # 11  Number of message passing steps during feature vectors are classified (after Message Passing) + 1
        outputs_dict = {'classified_edges': []}
        for step in range(1, self.num_enc_steps + 1):

            # Reattach the initially encoded embeddings before the update
            if self.reattach_initial_edges: # True 
                latent_edge_feats = torch.cat((initial_edge_feats, latent_edge_feats), dim=1)
            if self.reattach_initial_nodes: # False 
                latent_node_feats = torch.cat((initial_node_feats, latent_node_feats), dim=1)
 
            # Message Passing Step                            (N X 32)           (2 X E)      (E X 32)
            #  (N X 32)        (E X 16)
            latent_node_feats, latent_edge_feats = self.MPNet(latent_node_feats, edge_index, latent_edge_feats)

            if step >= first_class_step:
                # Classification Step               (E X 16)
                dec_edge_feats, _ = self.classifier(latent_edge_feats)
                outputs_dict['classified_edges'].append(dec_edge_feats)

        if self.num_enc_steps == 0:
            dec_edge_feats, _ = self.classifier(latent_edge_feats)
            outputs_dict['classified_edges'].append(dec_edge_feats)

        return outputs_dict
        

################################################################################################################
################################################################################################################
class Net_FP(torch.nn.Module):
    """
    Main Model Class. Contains all the components of the model. It consists of of several networks:
    - 2 encoder MLPs (1 for nodes, 1 for edges) that provide the initial node and edge embeddings, respectively,
    - 4 update MLPs (3 for nodes, 1 per edges) used in the 'core' Message Passing Network
    - 1 edge classifier MLP that performs binary classification over the Message Passing Network's output to classify active edges.
    - 1 node classifier MLP that performs binary classification over the Message Passing Network's output to classify FP.

    """
    def __init__(self, params= None):
        super(Net_FP, self).__init__()
        

        self.node_cnn = None # CNN used to encode bounding box apperance information.

        # Assign paramters
        self.params = params
        if isinstance(self.params, dict):
            for key, value in self.params.items():
                setattr(self, key, value)
 
        # Define tasks
        if (self.params is None) or ("edge_classification_task" not in self.params):  
            self.edge_classification_task = 1 # default solve assocoaition
        if (self.params is None) or ("node_classification_task" not in self.params):  
            self.node_classification_task = 0


        # Define Encoder and Classifier Networks
        # parameters
   
        # Change to encode directly positions of 8 angles
        # encoder_feats_dict = {'edge_in_dim': 6, 'edge_fc_dims': [18, 18], 'edge_out_dim': 16, 'node_in_dim': 2048, 'node_fc_dims': [128], 'node_out_dim': 32, 'dropout_p': 0, 'use_batchnorm': False}
        encoder_feats_dict = {'edge_in_dim': 6, 'edge_fc_dims': [18, 18], 'edge_out_dim': 16, 'node_in_dim': 3*8, 'node_fc_dims': [72], 'node_out_dim': 32, 'dropout_p': 0, 'use_batchnorm': False}

        if self.edge_classification_task:
            classifier_feats_dict = {'edge_in_dim': 16, 'edge_fc_dims': [8], 'edge_out_dim': 1, 'dropout_p': 0, 'use_batchnorm': False}
            # final binary classifier for edges
            self.classifier = MLPGraphIndependent(**classifier_feats_dict)

        if self.node_classification_task:
            node_classifier_feats_dict = {'edge_in_dim': 32, 'edge_fc_dims': [16], 'edge_out_dim': 1, 'dropout_p': 0, 'use_batchnorm': False}
            # final binary classifier for nodes FP
            self.node_classifier = MLPGraphIndependent(**node_classifier_feats_dict)

        # 2 encoder MLPs (1 for nodes (only second part), 1 for edges) that provide the initial node and edge embeddings 
        self.encoder = MLPGraphIndependent(**encoder_feats_dict)


        # Define the 'Core' message passing network (i.e. node and edge update models)
        self.MPNet = self._build_core_MPNet(encoder_feats_dict=encoder_feats_dict)

        if (self.params is None) or ("num_enc_steps" not in self.params):  
            self.num_enc_steps = 1#12 #  Number of message passing steps
            self.num_class_steps = 1#11 #  Number of message passing steps during feature vectors are classified (after Message Passing)


    # Building core network, called in __init__
    # - 4 update MLPs (3 for nodes, 1 per edges) used in the 'core' Message Passing Network
    def _build_core_MPNet(self, encoder_feats_dict):
        """
        Builds the core part of the Message Passing Network: Node Update and Edge Update models.
        Args:
            encoder_feats_dict: dictionary containing the hyperparameters for the initial node/edge encoder
        """

        # Define an aggregation operator for nodes to 'gather' messages from incident edges
        node_agg_fn = 'sum'
        assert node_agg_fn.lower() in ('mean', 'max', 'sum'), "node_agg_fn can only be 'max', 'mean' or 'sum'."

        if node_agg_fn == 'mean':
            node_agg_fn = lambda out, row, x_size: scatter_mean(out, row, dim=0, dim_size=x_size)

        elif node_agg_fn == 'max':
            node_agg_fn = lambda out, row, x_size: scatter_max(out, row, dim=0, dim_size=x_size)[0]

        elif node_agg_fn == 'sum':
            node_agg_fn = lambda out, row, x_size: scatter_add(out, row, dim=0, dim_size=x_size)

        # Define all multi-layer perceptrons (MLPs) involved in the graph network
        # For both nodes and edges, the initial encoded features (i.e. output of self.encoder) can either be
        # reattached or not after each Message Passing Step. This affects MLPs input dimensions
        self.reattach_initial_nodes = 0 # Determines whether initially encoded node feats are used during node updates     (skip-connection)
        self.reattach_initial_edges = True  # Determines whether initially encoded edge feats are used during node updates (skip-connection)

        edge_factor = 2 if self.reattach_initial_edges else 1  # 2
        node_factor = 2 if self.reattach_initial_nodes else 1  # 1
                          # 1 * 2 * 32 + 2 * 16
        # 96
        edge_model_in_dim = node_factor * 2 * encoder_feats_dict['node_out_dim'] + edge_factor * encoder_feats_dict[
            'edge_out_dim'] 
        # 48
        node_model_in_dim = node_factor * encoder_feats_dict['node_out_dim'] + encoder_feats_dict['edge_out_dim']


        # Define all MLPs used within the MPN
        edge_mlp = MLP(input_dim=96,
                       fc_dims=[80, 16],
                       dropout_p=0,
                       use_batchnorm=0)

        flow_in_mlp = MLP(input_dim=48,
                          fc_dims=[56, 32],
                          dropout_p=0,
                          use_batchnorm=0)

        flow_out_mlp = MLP(input_dim=48,
                           fc_dims=[56, 32],
                           dropout_p=0,
                           use_batchnorm=0)

        node_mlp = nn.Sequential(*[nn.Linear(2 * encoder_feats_dict['node_out_dim'], # 64
                                 encoder_feats_dict['node_out_dim']), # 32
                                   nn.ReLU(inplace=True)])

        # Define all MLPs used within the MPN
        return MetaLayer(edge_model=EdgeModel(edge_mlp = edge_mlp), # edge update model
                                                                     # 2 node update (fut, past) + 1 node update model
                         node_model=TimeAwareNodeModel(flow_in_mlp = flow_in_mlp,  # in 48, out 32
                                                       flow_out_mlp = flow_out_mlp, # in 48, out 32
                                                       node_mlp = node_mlp, # in 64, out 32
                                                       node_agg_fn = node_agg_fn)) # aggregation function (sum)
        

    def forward(self, data):
        """
        Provides a fractional solution to the data association problem.
        First, node and edge features are independently encoded by the encoder network. Then, they are iteratively
        'combined' for a fixed number of steps via the Message Passing Network (self.MPNet). Finally, they are
        classified independently by the classifiernetwork.
        Args:
            data: object containing attribues
              - x: node features matrix
              - edge_index: tensor with shape [2, M], with M being the number of edges, indicating nonzero entries in the
                graph adjacency (i.e. edges) (i.e. sparse adjacency)
              - edge_attr: edge features matrix (sorted by edge apperance in edge_index) #edges X #edge_features (6)

        Returns:
            classified_edges: list of unnormalized node probabilites after each MP step
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # First encoding of node images of boxes/features
        x_is_img = len(x.shape) == 4
        if self.node_cnn is not None and x_is_img:
            x = self.node_cnn(x)

            emb_dists = nn.functional.pairwise_distance(x[edge_index[0]], x[edge_index[1]]).view(-1, 1)
            edge_attr = torch.cat((edge_attr, emb_dists), dim = 1)

        # Encoding features step
        latent_edge_feats, latent_node_feats = self.encoder(edge_attr, x) # edge_attr (E X 6), x are the node features (N X 2048) -> change to (N X 8)
        # latent_edge_feats (E X 16)
        # latent_node_feats (N X 32)
        
        initial_edge_feats = latent_edge_feats
        initial_node_feats = latent_node_feats

        # During training, the feature vectors that the MPNetwork outputs for the  last self.num_class_steps message
        # passing steps are classified in order to compute the loss.
        first_class_step = self.num_enc_steps - self.num_class_steps + 1 # total 2 = 12  Number of message passing steps - # 11  Number of message passing steps during feature vectors are classified (after Message Passing) + 1
        
        if self.edge_classification_task and self.node_classification_task:
            outputs_dict = {'classified_edges': [], 'classified_nodes':[]}
        elif self.edge_classification_task:
            outputs_dict = {'classified_edges': []}
        elif self.node_classification_task: 
            outputs_dict = {'classified_nodes':[]}
        
        for step in range(1, self.num_enc_steps + 1):

            # Reattach the initially encoded embeddings before the update
            if self.reattach_initial_edges: # True 
                latent_edge_feats = torch.cat((initial_edge_feats, latent_edge_feats), dim=1)
            if self.reattach_initial_nodes: # True 
                latent_node_feats = torch.cat((initial_node_feats, latent_node_feats), dim=1)
 
            # Message Passing Step                            (N X 32)           (2 X E)      (E X 32)
            #  (N X 32)        (E X 16)
            latent_node_feats, latent_edge_feats = self.MPNet(latent_node_feats, edge_index, latent_edge_feats)

            if step >= first_class_step:

                if self.edge_classification_task:
                    # Edge Classification Step               (E X 16)
                    dec_edge_feats, _ = self.classifier(latent_edge_feats)
                    outputs_dict['classified_edges'].append(dec_edge_feats)

                if self.node_classification_task:
                    # Node Classification Step
                    dec_node_feats, _ = self.node_classifier(latent_node_feats)
                    outputs_dict['classified_nodes'].append(dec_node_feats)

        if self.num_enc_steps == 0:

            if self.edge_classification_task:
                dec_edge_feats, _ = self.classifier(latent_edge_feats)
                outputs_dict['classified_edges'].append(dec_edge_feats)

            if self.node_classification_task:
                dec_node_feats, _ = self.node_classifier(latent_node_feats)
                outputs_dict['classified_nodes'].append(dec_node_feats)

        return outputs_dict


################################################################################################################
################################################################################################################
class EdgeModelBP(nn.Module):
    """
    Class used to peform the edge update during Neural message passing
    """
    def __init__(self, edge_mlp):
        super(EdgeModelBP, self).__init__()
        self.edge_mlp = edge_mlp

    def forward(self, source, target, edge_attr):
        out = torch.cat([source, target, edge_attr], dim=1)
        
        # return      2E X self.number_state_variables*2 + self.NUM_MEAS*2
        return self.edge_mlp(out)
    
class NodeModelBP(nn.Module):
    """
    Class used to peform the node update during Neural mwssage passing
    """
    def __init__(self, node_mlp, node_agg_fn):
        super(NodeModelBP, self).__init__()

        self.node_mlp = node_mlp      
        self.node_agg_fn = node_agg_fn # 'sum'

       
    def forward(self, x, x_lstm, encoded_z_gnss, edge_index, edge_attr):

        row, col = edge_index
        # N X out_edge_mlp: 1
        edge_attr = self.node_agg_fn(edge_attr, row, x.size(0))
        out = torch.cat([x, x_lstm, encoded_z_gnss, edge_attr], dim=1)
        out = self.node_mlp(out)

               # N X number_state_variables
        return out


class MetaLayerBP(torch.nn.Module):
    """
    Core Message Passing Network Class. Extracted from torch_geometric, with minor modifications.
    (https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html)
    """
    def __init__(self, edge_model=None, node_model=None):
        """
        Args:
            edge_model: Callable Edge Update Model
            node_model: Callable Node Update Model
        """
        super(MetaLayerBP, self).__init__()

        self.edge_model = edge_model
        self.node_model = node_model
        self.reset_parameters()

    def reset_parameters(self):
        for item in [self.node_model, self.edge_model]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()

    def forward(self, x, x_lstm, encoded_z_gnss, edge_index, edge_attr):

        row, col = edge_index  # row = first row, col = second row

        # Edge Update
        if self.edge_model is not None:#(2E X NUM_FEAT) (2E X NUM_FEAT)      (2E X (NUM_FEAT+NUM_FEAT))
            edge_attr = self.edge_model(x[row],        x[col],       edge_attr)

        # Node Update
        if self.node_model is not None:
             #                (N X NUM_FEAT)  (N X NUM_FEAT)  (N X NUM_FEAT)    (2 X 2E)   (2E X out_edge_mlp: 1)
            x = self.node_model(x,             x_lstm,        encoded_z_gnss,   edge_index, edge_attr)

        return x, edge_attr

    def __repr__(self):
        return '{}(edge_model={}, node_model={})'.format(self.__class__.__name__, self.edge_model, self.node_model)



class NetBP(torch.nn.Module):

    def __init__(self, model_params= None):
        super(NetBP, self).__init__()
        
        # Assign paramters
        self.model_params = model_params
        if isinstance(self.model_params, dict):
            for key, value in self.model_params.items():
                setattr(self, key, value)
 
        # # Change to encode directly positions of 8 angles
        # # encoder_feats_dict = {'edge_in_dim': 6, 'edge_fc_dims': [18, 18], 'edge_out_dim': 16, 'node_in_dim': 2048, 'node_fc_dims': [128], 'node_out_dim': 32, 'dropout_p': 0, 'use_batchnorm': False}
        encoder_feats_dict = {'edge_in_dim':self.NUM_MEAS, 'edge_fc_dims': [18, 18], 'edge_out_dim': self.num_node_latent_feat, 'node_in_dim': self.NUM_MEAS_GNSS, 'node_fc_dims': [72], 'node_out_dim': self.num_node_latent_feat, 'dropout_p': 0, 'use_batchnorm': False}
        regressor_feats_dict = {'node_in_dim': self.num_node_latent_feat, 'node_fc_dims': [16, 256, 128], 'node_out_dim': self.number_state_variables, 'dropout_p': 0, 'use_batchnorm': False}

        # # 2 encoder MLPs (1 for nodes (only second part), 1 for edges) that provide the initial node and edge embeddings 
        self.encoder = MLPGraphIndependent(**encoder_feats_dict)
        # # final regression for nodes
        self.regressor = MLPGraphIndependent_sparse(**regressor_feats_dict)

        # Define the 'Core' message passing network (i.e. node and edge update models)
        self.MPNet = self._build_core_MPNet(encoder_feats_dict=encoder_feats_dict)

        if (self.model_params is None) or ("T_message_steps" not in self.model_params):  
            self.T_message_steps = 1
            self.num_class_steps = 1


    # Building core network, called in __init__
    # - 4 update MLPs (3 for nodes, 1 per edges) used in the 'core' Message Passing Network
    def _build_core_MPNet(self, encoder_feats_dict):
        """
        Builds the core part of the Message Passing Network: Node Update and Edge Update models.
        Args:
            encoder_feats_dict: dictionary containing the hyperparameters for the initial node/edge encoder
        """

        # Define an aggregation operator for nodes to 'gather' messages from incident edges
        node_agg_fn = 'sum'
        assert node_agg_fn.lower() in ('mean', 'max', 'sum'), "node_agg_fn can only be 'max', 'mean' or 'sum'."

        if node_agg_fn == 'mean':
            node_agg_fn = lambda out, row, x_size: scatter_mean(out, row, dim=0, dim_size=x_size)

        elif node_agg_fn == 'max':
            node_agg_fn = lambda out, row, x_size: scatter_max(out, row, dim=0, dim_size=x_size)[0]

        elif node_agg_fn == 'sum':
            node_agg_fn = lambda out, row, x_size: scatter_add(out, row, dim=0, dim_size=x_size)

        # Define all multi-layer perceptrons (MLPs) involved in the graph network
        # For both nodes and edges, the initial encoded features (i.e. output of self.encoder) can either be
        # reattached or not after each Message Passing Step. This affects MLPs input dimensions
        self.reattach_initial_nodes = False # Determines whether initially encoded node feats are used during node updates
        self.reattach_initial_edges = True  # Determines whether initially encoded edge feats are used during node updates

        edge_factor = 2 if self.reattach_initial_edges else 1  # 2
        node_factor = 2 if self.reattach_initial_nodes else 1  # 1
                          # 1 * 2 * 32 + 2 * 16

        # Define all MLPs used within the MPN
        edge_mlp = MLP(input_dim=self.num_node_latent_feat*4,
                       fc_dims=[80, 16, self.num_node_latent_feat],
                       dropout_p=0,
                       use_batchnorm=0)

        node_mlp = nn.Sequential(*[nn.Linear(2 * self.num_node_latent_feat + self.num_node_latent_feat + + self.num_node_latent_feat, # x + x_lstm + out_edge_mlp: self.number_state_variables
                                 self.num_node_latent_feat), # 32
                                   nn.GELU()])
                                #  nn.ReLU(inplace=True)])

        # Define all MLPs used within the MPN
        return MetaLayerBP(edge_model=EdgeModelBP(edge_mlp = edge_mlp), # edge update model
                                                                     # 2 node update (fut, past) + 1 node update model
                        node_model = NodeModelBP(node_mlp, node_agg_fn=node_agg_fn)) 
        

    def forward(self, x_lstm, z_gnss, edge_index, z_inter_meas):

        # Encoding features step
        # Encoder of measurements
        encoded_z_inter_meas, encoded_z_gnss = self.encoder(z_inter_meas, z_gnss)

        latent_node_feats = x_lstm
        latent_edge_feats = encoded_z_inter_meas

        # During training, the feature vectors that the MPNetwork outputs for the  last self.num_class_steps message
        # passing steps are classified in order to compute the loss.
        first_class_step = self.T_message_steps - self.num_regress_steps + 1 # total 2 = 12  Number of message passing steps - # 11  Number of message passing steps during feature vectors are classified (after Message Passing) + 1
        outputs_dict = {'x_hat': []}
        for step in range(1, self.T_message_steps + 1):

            # Reattach the initially encoded embeddings before the update
            if self.reattach_initial_edges: # True 
                latent_edge_feats = torch.cat((encoded_z_inter_meas, latent_edge_feats), dim=1)
            if self.reattach_initial_nodes: # False 
                latent_node_feats = torch.cat((x_lstm, latent_node_feats), dim=1)
 
            # Message Passing Step                           
            # (N X number_state_variables) (2E X out_edge_mlp)                           
            latent_node_feats, latent_edge_feats = self.MPNet(latent_node_feats, x_lstm, encoded_z_gnss, edge_index, latent_edge_feats)

            if step >= first_class_step:
                # Regression Step              
                _, dec_node_feats = self.regressor(nodes_feats = latent_node_feats)
                outputs_dict['x_hat'].append(dec_node_feats)

        if self.T_message_steps == 0:
            # Regression Step   
            _, dec_node_feats = self.regressor(nodes_feats = latent_node_feats)
            outputs_dict['x_hat'].append(dec_node_feats)

        return outputs_dict
        


class NetBP_LSTM(torch.nn.Module):

    def __init__(self, model_params= None, training_validation_dataset_params = None, dataset_params = None):
        super(NetBP_LSTM, self).__init__()
        
        # Assign paramters
        self.model_params = model_params
        if isinstance(self.model_params, dict):
            for key, value in self.model_params.items():
                setattr(self, key, value)
        self.training_validation_dataset_params = training_validation_dataset_params
        if isinstance(self.training_validation_dataset_params, dict):
            for key, value in self.training_validation_dataset_params.items():
                setattr(self, key, value)
        self.dataset_params = dataset_params
        if isinstance(self.dataset_params, dict):
            for key, value in self.dataset_params.items():
                setattr(self, key, value)          

        self.LSTM_bidirectional_dimension = 2 if self.LSTM_bidirectional else 1
        self.lstm_layer1 = nn.LSTM(input_size=self.model_params['number_state_variables'],hidden_size=int(128/self.LSTM_bidirectional_dimension), num_layers=self.LSTM_num_layers, bidirectional = bool(self.LSTM_bidirectional), batch_first=True)
        self.lstm_layer1_act = nn.ReLU(inplace=False)       
        self.lstm_layer2 = nn.LSTM(input_size=128,hidden_size=int(256/self.LSTM_bidirectional_dimension), num_layers=self.LSTM_num_layers, bidirectional = bool(self.LSTM_bidirectional), batch_first=True)
        self.lstm_layer2_act = nn.ReLU(inplace=False) 
        self.lstm_maxout1 = Maxout(256, 128, 2)
        self.lstm_linear1 = nn.Linear(128, 64)
        self.lstm_linear2 = nn.Linear(64, self.model_params['num_node_latent_feat'])
        
        self.NetBP = NetBP(model_params)
        
    # BATCH-WISE
    def forward(self, data, testing = 0):

        if type(data) is not list:
            data = [data]
            self.batch_size_real = 1

        # initialize the hidden state.
        # always in the shape (2/1 (mono-bi directional) * num_layers, B, H_out/H_hidden) 
        # hidden = (torch.randn(1, self.num_agents, self.model_params['number_state_variables']),
        #         torch.randn(1, self.num_agents, self.model_params['number_state_variables']))
        hidden_layer1 = (torch.randn(self.LSTM_num_layers*self.LSTM_bidirectional_dimension, self.num_agents*self.batch_size_real, int(128/self.LSTM_bidirectional_dimension)).to(self.device),
                torch.randn(self.LSTM_num_layers*self.LSTM_bidirectional_dimension, self.num_agents*self.batch_size_real, int(128/self.LSTM_bidirectional_dimension)).to(self.device))
        hidden_layer2 = (torch.randn(self.LSTM_num_layers*self.LSTM_bidirectional_dimension, self.num_agents*self.batch_size_real, int(256/self.LSTM_bidirectional_dimension)).to(self.device),
                torch.randn(self.LSTM_num_layers*self.LSTM_bidirectional_dimension, self.num_agents*self.batch_size_real, int(256/self.LSTM_bidirectional_dimension)).to(self.device))
        
        # output
        outputs_dict = {'output_lstm': [],
                    'outputs_dict_MPNN': []}
        

        for n in range(self.batch_size_length_seq):

            # reshape such that we have a batch size composed of the number of instances (batch_size_real) * num_agents
            x = torch.cat(([data[b][n].x for b in range(self.batch_size_real)]), 0)
            node_attr = torch.cat(([data[b][n].node_attr for b in range(self.batch_size_real)]), 0)
            num_nodes = [data[b][n].x.shape[0] for b in range(self.batch_size_real)]
            # adjust indexes of edges for batch procedure
            num_nodes[0] = 0
            num_nodes = np.cumsum(num_nodes)
            edge_index = torch.cat(([data[b][n].edge_index+num_nodes[b] for b in range(self.batch_size_real)]), 1)
            edge_attr = torch.cat(([data[b][n].edge_attr for b in range(self.batch_size_real)]), 0)
            
            # INPUT: (B: batch_size, L: length seq, Hin: num input features)
            # B = self.num_agents
            # L = 1 (manually step through the sequence one element at a time.)
            # Hin = 6 (x, y, vx, vy, ax, ay)
            x_lstm, hidden_layer1 = self.lstm_layer1(x.view(self.num_agents*self.batch_size_real, 1, self.model_params['number_state_variables']), hidden_layer1)
            x_lstm = self.lstm_layer1_act(x_lstm)
            x_lstm, hidden_layer2 = self.lstm_layer2(x_lstm, hidden_layer2)
            x_lstm = self.lstm_layer2_act(x_lstm)
            x_lstm = self.lstm_maxout1(x_lstm)
            x_lstm = self.lstm_linear1(x_lstm)
            x_lstm = self.lstm_linear2(x_lstm)

            # MPNN-BP
            outputs_dict_MPNN = self.NetBP(x_lstm.view(self.num_agents*self.batch_size_real, self.model_params['num_node_latent_feat']), node_attr, edge_index, edge_attr)

            # store output
            outputs_dict['output_lstm'].append(x_lstm)
            # outputs_dict['hidden_lstm'].append(hidden)
            outputs_dict['outputs_dict_MPNN'].append(outputs_dict_MPNN)

        return outputs_dict


################################################################################################################
################################################################################################################
class EdgeModelBP_single(nn.Module):
    def __init__(self, edge_mlp):
        super(EdgeModelBP_single, self).__init__()
        self.edge_mlp = edge_mlp

    def forward(self, source, target, edge_attr):
        out = torch.cat([source, target, edge_attr], dim=1)
        
        return self.edge_mlp(out)
    
class NodeModelBP_single(nn.Module):
    def __init__(self, node_mlp, node_agg_fn):
        super(NodeModelBP_single, self).__init__()

        self.node_mlp = node_mlp      
        self.node_agg_fn = node_agg_fn # 'sum'

       
    def forward(self, x, x_lstm, encoded_z_gnss, edge_index, edge_attr, node_indexes_related_to_agent, device):

        row, col = edge_index
        # edge_attr = self.node_agg_fn(edge_attr, torch.tensor(LabelEncoder().fit_transform(col)).to(device), x[node_indexes_related_to_agent].size(0))
        edge_attr = self.node_agg_fn(edge_attr, row, x.size(0))
        out = torch.cat([x[node_indexes_related_to_agent], x_lstm, encoded_z_gnss, edge_attr[node_indexes_related_to_agent,:]], dim=1)
        out = self.node_mlp(out)
        return out

class MetaLayerBP_single(torch.nn.Module):
    """
    Core Message Passing Network Class. Extracted from torch_geometric, with minor modifications.
    (https://rusty1s.github.io/pytorch_geometric/build/html/modules/nn.html)
    """
    def __init__(self, edge_model=None, node_model=None, device = 'cpu'):
        """
        Args:
            edge_model: Callable Edge Update Model
            node_model: Callable Node Update Model
        """
        super(MetaLayerBP_single, self).__init__()

        self.edge_model = edge_model
        self.node_model = node_model
        self.device = device
        self.reset_parameters()

    def reset_parameters(self):
        for item in [self.node_model, self.edge_model]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()

    # node_indexes_related_to_agent: to find the x corresponding to the specific agent. Same timestamp n, same agent, different instances (batch)
    # edge_index has been already selected
    # x: all the node_features for each agent (exchanged)
    def forward(self, x, x_lstm, encoded_z_gnss, edge_index, edge_attr, node_indexes_related_to_agent, edge_indexes_related_to_agent):

        row, col = edge_index  # row = first row, col = second row

        # Edge Update
        if self.edge_model is not None:#(2E X NUM_FEAT) (2E X NUM_FEAT)      (2E X (NUM_FEAT+NUM_FEAT))
            edge_attr = self.edge_model(x[row],        x[col],       edge_attr)

        # Node Update
        if self.node_model is not None:
             #                (N X NUM_FEAT)  (N X NUM_FEAT)  (N X NUM_FEAT)    (2 X 2E)   (2E X out_edge_mlp: 1)
            x = self.node_model(x,             x_lstm,        encoded_z_gnss,   edge_index, edge_attr, node_indexes_related_to_agent, self.device)

        return x, edge_attr

    def __repr__(self):
        return '{}(edge_model={}, node_model={})'.format(self.__class__.__name__, self.edge_model, self.node_model)


class LSTM_single(torch.nn.Module):
    
    def __init__(self, model_params= None, training_validation_dataset_params = None, dataset_params = None):
        super(LSTM_single, self).__init__()

        # Assign paramters
        self.model_params = model_params
        if isinstance(self.model_params, dict):
            for key, value in self.model_params.items():
                setattr(self, key, value)
        self.training_validation_dataset_params = training_validation_dataset_params
        if isinstance(self.training_validation_dataset_params, dict):
            for key, value in self.training_validation_dataset_params.items():
                setattr(self, key, value)
        self.dataset_params = dataset_params
        if isinstance(self.dataset_params, dict):
            for key, value in self.dataset_params.items():
                setattr(self, key, value)

        self.LSTM_bidirectional_dimension = 2 if self.LSTM_bidirectional else 1

        self.hidden_layer1 = (torch.randn(self.LSTM_num_layers*self.LSTM_bidirectional_dimension, self.num_agents*self.batch_size_real, int(128/self.LSTM_bidirectional_dimension)).to(self.device),
                torch.randn(self.LSTM_num_layers*self.LSTM_bidirectional_dimension, self.num_agents*self.batch_size_real, int(128/self.LSTM_bidirectional_dimension)).to(self.device))
        self.hidden_layer2 = (torch.randn(self.LSTM_num_layers*self.LSTM_bidirectional_dimension, self.num_agents*self.batch_size_real, int(256/self.LSTM_bidirectional_dimension)).to(self.device),
                torch.randn(self.LSTM_num_layers*self.LSTM_bidirectional_dimension, self.num_agents*self.batch_size_real, int(256/self.LSTM_bidirectional_dimension)).to(self.device))
        
        self.lstm_layer1 = nn.LSTM(input_size=self.model_params['number_state_variables'],hidden_size=int(128/self.LSTM_bidirectional_dimension), num_layers=self.LSTM_num_layers, bidirectional = bool(self.LSTM_bidirectional), batch_first=True)
        self.lstm_layer1_act = nn.ReLU(inplace=False)       
        self.lstm_layer2 = nn.LSTM(input_size=128,hidden_size=int(256/self.LSTM_bidirectional_dimension), num_layers=self.LSTM_num_layers, bidirectional = bool(self.LSTM_bidirectional), batch_first=True)
        self.lstm_layer2_act = nn.ReLU(inplace=False) 
        self.lstm_maxout1 = Maxout(256, 128, 2)
        self.lstm_linear1 = nn.Linear(128, 64)
        self.lstm_linear2 = nn.Linear(64, self.model_params['num_node_latent_feat'])    

    def forward(self, x, n, testing = 0):

        if n == 0:
            self.hidden_layer1 = (torch.randn(self.LSTM_num_layers*self.LSTM_bidirectional_dimension, self.num_agents*self.batch_size_real, int(128/self.LSTM_bidirectional_dimension)).to(self.device),
                    torch.randn(self.LSTM_num_layers*self.LSTM_bidirectional_dimension, self.num_agents*self.batch_size_real, int(128/self.LSTM_bidirectional_dimension)).to(self.device))
            self.hidden_layer2 = (torch.randn(self.LSTM_num_layers*self.LSTM_bidirectional_dimension, self.num_agents*self.batch_size_real, int(256/self.LSTM_bidirectional_dimension)).to(self.device),
                    torch.randn(self.LSTM_num_layers*self.LSTM_bidirectional_dimension, self.num_agents*self.batch_size_real, int(256/self.LSTM_bidirectional_dimension)).to(self.device))
            
        else:
            if not testing:
                self.hidden_layer1 = (self.hidden_layer1[0].clone().detach().requires_grad_(True), self.hidden_layer1[1].clone().detach().requires_grad_(True))
                self.hidden_layer2 = (self.hidden_layer2[0].clone().detach().requires_grad_(True), self.hidden_layer2[1].clone().detach().requires_grad_(True))

        x_lstm, self.hidden_layer1 = self.lstm_layer1(x.view(self.num_agents*self.batch_size_real, 1, self.model_params['number_state_variables']), self.hidden_layer1)
        x_lstm = self.lstm_layer1_act(x_lstm)
        x_lstm, self.hidden_layer2 = self.lstm_layer2(x_lstm, self.hidden_layer2)
        x_lstm = self.lstm_layer2_act(x_lstm)
        x_lstm = self.lstm_maxout1(x_lstm)
        x_lstm = self.lstm_linear1(x_lstm)
        x_lstm = self.lstm_linear2(x_lstm)

        return x_lstm
        
        

class NetBP_LSTM_single(torch.nn.Module):
    
    def __init__(self, model_params= None, training_validation_dataset_params = None, dataset_params = None):
        super(NetBP_LSTM_single, self).__init__()
        
        # Assign paramters
        self.model_params = model_params
        if isinstance(self.model_params, dict):
            for key, value in self.model_params.items():
                setattr(self, key, value)
        self.training_validation_dataset_params = training_validation_dataset_params
        if isinstance(self.training_validation_dataset_params, dict):
            for key, value in self.training_validation_dataset_params.items():
                setattr(self, key, value)
        self.dataset_params = dataset_params
        if isinstance(self.dataset_params, dict):
            for key, value in self.dataset_params.items():
                setattr(self, key, value)       

        self.lstm = LSTM_single(model_params, training_validation_dataset_params, dataset_params)

        # # Change to encode directly positions of 8 angles
        # # encoder_feats_dict = {'edge_in_dim': 6, 'edge_fc_dims': [18, 18], 'edge_out_dim': 16, 'node_in_dim': 2048, 'node_fc_dims': [128], 'node_out_dim': 32, 'dropout_p': 0, 'use_batchnorm': False}
        encoder_feats_dict = {'edge_in_dim':self.NUM_MEAS, 'edge_fc_dims': [18, 18], 'edge_out_dim': self.num_node_latent_feat, 'node_in_dim': self.NUM_MEAS_GNSS, 'node_fc_dims': [72], 'node_out_dim': self.num_node_latent_feat, 'dropout_p': 0, 'use_batchnorm': False}
        regressor_feats_dict = {'node_in_dim': self.num_node_latent_feat, 'node_fc_dims': [16, 256, 128], 'node_out_dim': self.number_state_variables, 'dropout_p': 0, 'use_batchnorm': False}

        # # 2 encoder MLPs (1 for nodes (only second part), 1 for edges) that provide the initial node and edge embeddings 
        self.encoder = MLPGraphIndependent(**encoder_feats_dict)
        # # final regression for nodes
        self.regressor = MLPGraphIndependent_sparse(**regressor_feats_dict)

        self.MPNet = self._build_core_MPNet_single(encoder_feats_dict=encoder_feats_dict)

    def _build_core_MPNet_single(self, encoder_feats_dict):

        # Define an aggregation operator for nodes to 'gather' messages from incident edges
        node_agg_fn = 'sum'
        assert node_agg_fn.lower() in ('mean', 'max', 'sum'), "node_agg_fn can only be 'max', 'mean' or 'sum'."

        if node_agg_fn == 'mean':
            node_agg_fn = lambda out, row, x_size: scatter_mean(out, row, dim=0, dim_size=x_size)

        elif node_agg_fn == 'max':
            node_agg_fn = lambda out, row, x_size: scatter_max(out, row, dim=0, dim_size=x_size)[0]

        elif node_agg_fn == 'sum':
            node_agg_fn = lambda out, row, x_size: scatter_add(out, row, dim=0, dim_size=x_size)

        # Define all multi-layer perceptrons (MLPs) involved in the graph network
        # For both nodes and edges, the initial encoded features (i.e. output of self.encoder) can either be
        # reattached or not after each Message Passing Step. This affects MLPs input dimensions
        self.reattach_initial_nodes = False # Determines whether initially encoded node feats are used during node updates
        self.reattach_initial_edges = True  # Determines whether initially encoded edge feats are used during node updates

        edge_factor = 2 if self.reattach_initial_edges else 1  # 2
        node_factor = 2 if self.reattach_initial_nodes else 1  # 1


        # Define all MLPs used within the MPN
        edge_mlp = MLP(input_dim=self.num_node_latent_feat*4,
                       fc_dims=[80, 16, self.num_node_latent_feat],
                       dropout_p=0,
                       use_batchnorm=0)

        node_mlp = nn.Sequential(*[nn.Linear(2 * self.num_node_latent_feat + self.num_node_latent_feat + + self.num_node_latent_feat, # x + x_lstm + out_edge_mlp: self.number_state_variables
                                 self.num_node_latent_feat), # 32
                                   nn.GELU()])

        # Define all MLPs used within the MPN
        return MetaLayerBP_single(edge_model=EdgeModelBP_single(edge_mlp = edge_mlp), # edge update model
                            node_model = NodeModelBP_single(node_mlp, node_agg_fn=node_agg_fn), 
                            device = self.device)

################################################################################################################
################################################################################################################

class BP():
    # x_n = F x_n-1 + W w_n-1
    # x: pos, vel, acc. w: pos_noise, vel_noise, acc_noise
    def __init__(self, dataset_instance, dataset_params, model_params, F, Q, motion_model, inter_agent_meas_model = None, gnss_meas_model = None):
        self.dataset_params = dataset_params
        self.model_params = model_params
        self.F = F
        self.F_total = np.kron(np.eye(self.model_params['num_particles']), F)
        self.Q = Q           # var of pos_x, pos_y, vel_x, vel_t, acc_x, acc_y
        self.Gate = np.array([1, 1, 0, 0, 0, 0])

        self.create_beliefs(dataset_instance) # mean = beliefs[agent][0], var = beliefs[agent][1] of pos, vel, acc
        self.motion_model = motion_model

        self.T_message_steps = self.model_params['T_message_steps']

        if inter_agent_meas_model is not None:
            self.inter_agent_meas_model = inter_agent_meas_model
        else:
            self.inter_agent_meas_model = dataset_params['std_noise_measurements']**2
        
        if gnss_meas_model is not None:
            self.num_measurements_gnss = min(model_params['NUM_MEAS_GNSS'], model_params['number_state_variables'])
            self.diag_var_gnss = gnss_meas_model
            self.diag_sigma_gnss = np.sqrt(self.diag_var_gnss)
            self.Cov_gnss = np.diag(self.diag_var_gnss)   
        else:
            self.num_measurements_gnss = min(model_params['NUM_MEAS_GNSS'], model_params['number_state_variables'])
            self.diag_var_gnss = np.array([dataset_params['std_noise_gnss_position']**2, 
                                    dataset_params['std_noise_gnss_position']**2,
                                    dataset_params['std_noise_gnss_velocity']**2,
                                    dataset_params['std_noise_gnss_velocity']**2,
                                    dataset_params['std_noise_gnss_acceleration']**2,
                                    dataset_params['std_noise_gnss_acceleration']**2][:self.num_measurements_gnss]).astype(float)
            self.diag_sigma_gnss = np.sqrt(self.diag_var_gnss)
            self.Cov_gnss = np.diag(self.diag_var_gnss)   


        # mu_i
        self.prediction_message = {agent:0 for agent in range(self.dataset_params['num_agents'])}
        # mu_ij
        self.measurement_message = [ [ 0 for j in range(self.dataset_params['num_agents'])] for i in range(self.dataset_params['num_agents'])]
        # mu_i_gnss
        self.measurement_gnss_message = {agent:np.ones((1,self.model_params['num_particles'])) for agent in range(self.dataset_params['num_agents'])}

        # predicted x
        self.predicted_x = {agent:[] for agent in range(self.dataset_params['num_agents'])}
        self.predicted_C = {agent:[] for agent in range(self.dataset_params['num_agents'])}

        self.start_cooperation = 0
        

    def converge(self):
        pass


    def prediction(self, time_n):
        if not self.model_params['Particle_filter']:
            self.prediction_no_particle(time_n)
        else:
            self.prediction_particle(time_n)

    @lru_cache(maxsize=None)
    def update(self, dataset_instance, time_n):
        if not self.model_params['Particle_filter']:
            self.update_no_particle(dataset_instance, time_n)
        else:
            # non-coop
            # self.update_particle(dataset_instance, time_n)
            # coop
            self.update_particle3(dataset_instance, time_n)

    def estimate_position(self, time_n):
        if not self.model_params['Particle_filter']:
            self.estimate_position_no_particle(time_n)
        else:
            self.estimate_position_particle(time_n)

    def prediction_particle(self, time_n):
        if time_n == 0:
            # visualize prior
            # agent = 0
            # plot_particles(self.beliefs[agent][0][0,:].reshape([-1,]), self.beliefs[agent][0][1,:].reshape([-1,]), self.beliefs[agent][1].reshape([-1,]))
            return
        for agent in range(self.dataset_params['num_agents']):

            # add noise
            for i in range(self.model_params['number_state_variables']):
                self.beliefs[agent][0][i] = self.beliefs[agent][0][i] + np.sqrt(self.motion_model[i][i])*np.random.normal(size=(1, self.model_params['num_particles'])) 
            
            # motion prediction
            self.beliefs[agent][0] = self.F@self.beliefs[agent][0]

            self.beliefs[agent][1] = (np.ones((1,self.model_params['num_particles']))/self.model_params['num_particles'])

            if agent == 0 and self.model_params['log_BP']:
                weights = (self.beliefs[agent][1]/np.sum(self.beliefs[agent][1])).transpose()
                prod_particles_weights = np.repeat(weights, self.model_params['number_state_variables'], 1).transpose()*self.beliefs[agent][0]
                x_state_mean = np.sum(prod_particles_weights,1)/np.sum(self.beliefs[agent][1])
                mean = x_state_mean[:self.model_params['number_state_variables']]
                print('Prediction mean: a {} {}'.format(agent, mean.squeeze().tolist()))

    def update_particle(self, dataset_instance, time_n):
        measurements = dataset_instance.measurements[time_n]
        mutual_distances = dataset_instance.mutual_distances[time_n]
        connectivity_matrix = dataset_instance.connectivity_matrix[time_n]

        for t in range(self.T_message_steps):
            for i in range(self.dataset_params['num_agents']): # rx

                # evaluate likelihood gnss measurements
                gnss = np.array(dataset_instance.gnss[i][time_n][:self.num_measurements_gnss]).reshape([-1,1])
                likelihood_gnss = self.compute_likelihood('GNSS', gnss - self.beliefs[i][0][:self.num_measurements_gnss])
                if np.sum(likelihood_gnss) < 10^-20:
                    likelihood_gnss = np.ones((1,self.model_params['num_particles']))
                    raise('State meas not consistent')
                self.measurement_gnss_message[i] = likelihood_gnss

                for j in range(self.dataset_params['num_agents']): # tx
                    if connectivity_matrix[i][j]:

                        # evaluate the range between corresponding agents' particles
                        delta_x = self.beliefs[i][0][0] - self.beliefs[j][0][0]
                        delta_y = self.beliefs[i][0][1] - self.beliefs[j][0][1]
                        ranges_ij = np.sqrt(delta_x**2 + delta_y**2)
                        likelihood_range = self.compute_likelihood('TOA', ranges_ij-measurements[i][j])
                        if np.sum(likelihood_range) < 10^-50:
                            likelihood_range = np.ones((1,self.model_params['num_particles']))
                            raise('Direct RANGE not consistent')
                        self.measurement_message[i][j] = likelihood_range.reshape([1,-1])

                # Strategy: inter-meas + gnss
                self.beliefs[i][1] = self.beliefs[i][1] * np.prod([self.measurement_message[i][j] for j in range(self.dataset_params['num_agents']) if connectivity_matrix[i][j]],0) * self.measurement_gnss_message[i]
                # Strategy: only gnss
                # self.beliefs[i][1] = self.beliefs[i][1] * self.measurement_gnss_message[i]

                self.beliefs[i][1] = self.beliefs[i][1]/np.sum(self.beliefs[i][1])

                if i == 0 and self.model_params['log_BP']:
                    weights = (self.beliefs[i][1]/np.sum(self.beliefs[i][1])).transpose()
                    prod_particles_weights = np.repeat(weights, self.model_params['number_state_variables'], 1).transpose()*self.beliefs[i][0]
                    x_state_mean = np.sum(prod_particles_weights,1)/np.sum(self.beliefs[i][1])
                    mean = x_state_mean[:self.model_params['number_state_variables']]
                    print('Update1 mean: t {} a {} {}'.format(t, i, mean.squeeze().tolist()))

            # resampling
            for i in range(self.dataset_params['num_agents']):
                if t == 0:
                    self.beliefs[i][0][:2,:] = self.beliefs[i][0][:2,self.resampleSystematic(self.beliefs[i][1], self.model_params['num_particles']).astype(int).squeeze()]
                else:
                    self.beliefs[i][0][:,:] = self.beliefs[i][0][:,self.resampleSystematic(self.beliefs[i][1], self.model_params['num_particles']).astype(int).squeeze()]
                self.beliefs[i][1] = np.array(np.ones((1,self.model_params['num_particles']))/self.model_params['num_particles'])

                if i == 0 and self.model_params['log_BP'] and self.model_params['log_BP']:
                    weights = (self.beliefs[i][1]/np.sum(self.beliefs[i][1])).transpose()
                    prod_particles_weights = np.repeat(weights, self.model_params['number_state_variables'], 1).transpose()*self.beliefs[i][0]
                    x_state_mean = np.sum(prod_particles_weights,1)/np.sum(self.beliefs[i][1])
                    mean = x_state_mean[:self.model_params['number_state_variables']]
                    print('Update2 mean: t {} a {} {}'.format(t, i, mean.squeeze().tolist()))

            # regularize
            for i in range(self.dataset_params['num_agents']):
                if t == 0:
                    self.beliefs[i][0][:2,:] = self.regularizeAgentParticles(self.beliefs[i][0][:2,:])
                else:
                    self.beliefs[i][0][:,:] = self.regularizeAgentParticles(self.beliefs[i][0][:,:])

                if i == 0 and self.model_params['log_BP']:
                    weights = (self.beliefs[i][1]/np.sum(self.beliefs[i][1])).transpose()
                    prod_particles_weights = np.repeat(weights, self.model_params['number_state_variables'], 1).transpose()*self.beliefs[i][0]
                    x_state_mean = np.sum(prod_particles_weights,1)/np.sum(self.beliefs[i][1])
                    mean = x_state_mean[:self.model_params['number_state_variables']]
                    print('Update3 mean: t {} a {} {}\n'.format(t, i, mean.squeeze().tolist()))

        for i in range(self.dataset_params['num_agents']): # rx
            self.beliefs[i][1] = self.beliefs[i][1]/np.sum(self.beliefs[i][1])


    def update_particle2(self, dataset_instance, time_n):
        measurements = dataset_instance.measurements[time_n]
        mutual_distances = dataset_instance.mutual_distances[time_n]
        connectivity_matrix = dataset_instance.connectivity_matrix[time_n]

        # Noise gnss
        num_measurements = min(self.model_params['NUM_MEAS_GNSS'], self.model_params['number_state_variables'])
        diag_var_gnss = np.array([self.dataset_params['std_noise_gnss_position']**2, 
                        self.dataset_params['std_noise_gnss_position']**2,
                        self.dataset_params['std_noise_gnss_velocity']**2,
                        self.dataset_params['std_noise_gnss_velocity']**2,
                        self.dataset_params['std_noise_gnss_acceleration']**2,
                        self.dataset_params['std_noise_gnss_acceleration']**2][:num_measurements]).astype(float)
        diag_sigma_gnss = np.sqrt(diag_var_gnss)
        Cov_gnss = np.diag(diag_var_gnss)

        for t in range(self.T_message_steps):
            if t == 0:
                self.temp_beliefs = deepcopy(self.beliefs)

            for i in range(self.dataset_params['num_agents']): # rx

                # evaluate likelihood gnss measurements
                gnss = np.array(dataset_instance.gnss[i][time_n][:num_measurements]).reshape([-1,1])
                likelihood_gnss = (1 / np.sqrt((2 * np.pi) ** Cov_gnss.shape[0] * np.linalg.det(Cov_gnss))) * \
                        np.exp(-0.5 * np.sum((gnss - self.beliefs[i][0][:num_measurements]) ** 2 * np.tile(diag_sigma_gnss.T ** (-2), (self.model_params['num_particles'], 1)).T, axis=0))
                if np.sum(likelihood_gnss) < 10^-20:
                    likelihood_gnss = np.ones((1,self.model_params['num_particles']))
                    raise('State meas not consistent')
                self.measurement_gnss_message[i] = likelihood_gnss

                for j in range(self.dataset_params['num_agents']): # tx
                    if connectivity_matrix[i][j]:

                        # evaluate the range between corresponding agents' particles
                        delta_x = self.beliefs[i][0][0] - self.beliefs[j][0][0]
                        delta_y = self.beliefs[i][0][1] - self.beliefs[j][0][1]
                        ranges_ij = np.sqrt(delta_x**2 + delta_y**2)
                        likelihood_range = self.compute_likelihood('TOA', ranges_ij-measurements[i][j])
                        if np.sum(likelihood_range) < 10^-50:
                            likelihood_range = np.ones((1,self.model_params['num_particles']))
                            raise('Direct RANGE not consistent')
                        self.measurement_message[i][j] = likelihood_range.reshape([1,-1])

                # Strategy: inter-meas + gnss
                self.temp_beliefs[i][1] = self.beliefs[i][1] * np.prod([self.measurement_message[i][j] for j in range(self.dataset_params['num_agents']) if connectivity_matrix[i][j]],0) * self.measurement_gnss_message[i]
                # Strategy: only gnss
                # self.temp_beliefs[i][1] = self.beliefs[i][1] * self.measurement_gnss_message[i]

                self.temp_beliefs[i][1] = np.nan_to_num(self.temp_beliefs[i][1]/np.sum(self.temp_beliefs[i][1]))

                if i == 0 and self.model_params['log_BP']:
                    weights = (self.beliefs[i][1]/np.sum(self.beliefs[i][1])).transpose()
                    prod_particles_weights = np.repeat(weights, self.model_params['number_state_variables'], 1).transpose()*self.beliefs[i][0]
                    x_state_mean = np.sum(prod_particles_weights,1)/np.sum(self.beliefs[i][1])
                    mean = x_state_mean[:self.model_params['number_state_variables']]
                    print('Update1 mean: t {} a {} {}'.format(t, i, mean.squeeze().tolist()))

        for i in range(self.dataset_params['num_agents']): # rx
            self.beliefs[i][1] = np.nan_to_num(self.temp_beliefs[i][1]/np.sum(self.temp_beliefs[i][1]))

        # resampling
        for i in range(self.dataset_params['num_agents']):
            if t == 0:
                self.beliefs[i][0][:2,:] = self.beliefs[i][0][:2,self.resampleSystematic(self.beliefs[i][1], self.model_params['num_particles']).astype(int).squeeze()]
            else:
                self.beliefs[i][0][:,:] = self.beliefs[i][0][:,self.resampleSystematic(self.beliefs[i][1], self.model_params['num_particles']).astype(int).squeeze()]
            self.beliefs[i][1] = np.array(np.ones((1,self.model_params['num_particles']))/self.model_params['num_particles'])

            if i == 0 and self.model_params['log_BP'] and self.model_params['log_BP']:
                weights = (self.beliefs[i][1]/np.sum(self.beliefs[i][1])).transpose()
                prod_particles_weights = np.repeat(weights, self.model_params['number_state_variables'], 1).transpose()*self.beliefs[i][0]
                x_state_mean = np.sum(prod_particles_weights,1)/np.sum(self.beliefs[i][1])
                mean = x_state_mean[:self.model_params['number_state_variables']]
                print('Update2 mean: t {} a {} {}'.format(t, i, mean.squeeze().tolist()))

        # regularize
        for i in range(self.dataset_params['num_agents']):
            if t == 0:
                self.beliefs[i][0][:2,:] = self.regularizeAgentParticles(self.beliefs[i][0][:2,:])
            else:
                self.beliefs[i][0][:,:] = self.regularizeAgentParticles(self.beliefs[i][0][:,:])

            if i == 0 and self.model_params['log_BP']:
                weights = (self.beliefs[i][1]/np.sum(self.beliefs[i][1])).transpose()
                prod_particles_weights = np.repeat(weights, self.model_params['number_state_variables'], 1).transpose()*self.beliefs[i][0]
                x_state_mean = np.sum(prod_particles_weights,1)/np.sum(self.beliefs[i][1])
                mean = x_state_mean[:self.model_params['number_state_variables']]
                print('Update3 mean: t {} a {} {}\n'.format(t, i, mean.squeeze().tolist()))

        for i in range(self.dataset_params['num_agents']): # rx
            self.beliefs[i][1] = self.beliefs[i][1]/np.sum(self.beliefs[i][1])


    # TEMP_BELIEFS are in LOG 
    def update_particle3(self, dataset_instance, time_n):
        measurements = dataset_instance.measurements[time_n]
        mutual_distances = dataset_instance.mutual_distances[time_n]
        connectivity_matrix = dataset_instance.connectivity_matrix[time_n]

        for t in range(self.T_message_steps):
            if t == 0:
                self.temp_beliefs = deepcopy(self.beliefs)

            for i in range(self.dataset_params['num_agents']): # rx

                real_pos = dataset_instance.positions[i][time_n]

                # evaluate likelihood gnss measurements
                gnss = np.array(dataset_instance.gnss[i][time_n][:self.num_measurements_gnss]).reshape([-1,1])
                likelihood_gnss = self.compute_likelihood('GNSS', gnss - self.beliefs[i][0][:self.num_measurements_gnss])
                if np.sum(likelihood_gnss) < 10^-20:
                    likelihood_gnss = np.ones((1,self.model_params['num_particles']))
                    raise('State meas not consistent')
                self.measurement_gnss_message[i] = likelihood_gnss   

        
                for j in range(self.dataset_params['num_agents']): # tx
                    if connectivity_matrix[i][j]:

                        # evaluate the range between corresponding agents' particles
                        delta_x = self.beliefs[i][0][0] - self.beliefs[j][0][0]
                        delta_y = self.beliefs[i][0][1] - self.beliefs[j][0][1]
                        ranges_ij = np.sqrt(delta_x**2 + delta_y**2)
                        likelihood_range = self.compute_likelihood('TOA', ranges_ij-measurements[i][j])
                        if np.sum(likelihood_range) < 10^-50:
                            likelihood_range = np.ones((1,self.model_params['num_particles']))
                            raise('Direct RANGE not consistent')
                        self.measurement_message[i][j] = likelihood_range.reshape([1,-1])


                # Start with inter-agent measurements after few seconds
                # Strategy: inter-meas + gnss
                if self.start_cooperation or (time_n > 0):
                    self.temp_beliefs[i][1] = np.log(self.beliefs[i][1]) + np.sum([np.log(self.measurement_message[i][j]) for j in range(self.dataset_params['num_agents']) if connectivity_matrix[i][j]],0) + np.log(self.measurement_gnss_message[i])
                    self.start_cooperation = 1
                # Strategy: only gnss
                else:
                    self.temp_beliefs[i][1] = np.log(self.beliefs[i][1]) + np.log(self.measurement_gnss_message[i])

                # scale first
                self.temp_beliefs[i][1] = self.temp_beliefs[i][1] - np.max(self.temp_beliefs[i][1])
                self.temp_beliefs[i][1] = self.temp_beliefs[i][1] - np.log(np.sum(np.exp(self.temp_beliefs[i][1])))

                if i == 0 and self.model_params['log_BP']:
                    weights = (self.beliefs[i][1]/np.sum(self.beliefs[i][1])).transpose()
                    prod_particles_weights = np.repeat(weights, self.model_params['number_state_variables'], 1).transpose()*self.beliefs[i][0]
                    x_state_mean = np.sum(prod_particles_weights,1)/np.sum(self.beliefs[i][1])
                    mean = x_state_mean[:self.model_params['number_state_variables']]
                    print('Update1 mean: t {} a {} {}'.format(t, i, mean.squeeze().tolist()))

        for i in range(self.dataset_params['num_agents']): # rx
            # print('Time n: {}, a: {}, beliefs_max {}, beliefs_min {}'.format(time_n, i, np.max(self.temp_beliefs[i][1]), np.min(self.temp_beliefs[i][1])))
            self.temp_beliefs[i][1] = self.temp_beliefs[i][1] - np.max(self.temp_beliefs[i][1])
            self.beliefs[i][1] = np.exp(self.temp_beliefs[i][1] - np.log(np.sum(np.exp(self.temp_beliefs[i][1]))))

        # resampling
        for i in range(self.dataset_params['num_agents']):
            if t == 0:
                self.beliefs[i][0][:2,:] = self.beliefs[i][0][:2,self.resampleSystematic(self.beliefs[i][1], self.model_params['num_particles']).astype(int).squeeze()]
            else:
                self.beliefs[i][0][:,:] = self.beliefs[i][0][:,self.resampleSystematic(self.beliefs[i][1], self.model_params['num_particles']).astype(int).squeeze()]
            self.beliefs[i][1] = np.array(np.ones((1,self.model_params['num_particles']))/self.model_params['num_particles'])

            if i == 0 and self.model_params['log_BP'] and self.model_params['log_BP']:
                weights = (self.beliefs[i][1]/np.sum(self.beliefs[i][1])).transpose()
                prod_particles_weights = np.repeat(weights, self.model_params['number_state_variables'], 1).transpose()*self.beliefs[i][0]
                x_state_mean = np.sum(prod_particles_weights,1)/np.sum(self.beliefs[i][1])
                mean = x_state_mean[:self.model_params['number_state_variables']]
                print('Update2 mean: t {} a {} {}'.format(t, i, mean.squeeze().tolist()))

        # regularize
        for i in range(self.dataset_params['num_agents']):
            if 0: # t == 0:
                self.beliefs[i][0][:2,:] = self.regularizeAgentParticles(self.beliefs[i][0][:2,:])
            else:
                self.beliefs[i][0][:,:] = self.regularizeAgentParticles(self.beliefs[i][0][:,:])

            if i == 0 and self.model_params['log_BP']:
                weights = (self.beliefs[i][1]/np.sum(self.beliefs[i][1])).transpose()
                prod_particles_weights = np.repeat(weights, self.model_params['number_state_variables'], 1).transpose()*self.beliefs[i][0]
                x_state_mean = np.sum(prod_particles_weights,1)/np.sum(self.beliefs[i][1])
                mean = x_state_mean[:self.model_params['number_state_variables']]
                print('Update3 mean: t {} a {} {}\n'.format(t, i, mean.squeeze().tolist()))

        for i in range(self.dataset_params['num_agents']): # rx
            self.beliefs[i][1] = self.beliefs[i][1]/np.sum(self.beliefs[i][1])



    def estimate_position_particle(self, time_n):
        for agent in range(self.dataset_params['num_agents']):
            weights = (self.beliefs[agent][1]/np.sum(self.beliefs[agent][1])).transpose()
            prod_particles_weights = np.repeat(weights, self.model_params['number_state_variables'], 1).transpose()*self.beliefs[agent][0]
            x_state_mean = np.sum(prod_particles_weights,1)/np.sum(self.beliefs[agent][1])

            C = np.cov(self.beliefs[agent][0]-x_state_mean.reshape([-1,1]))

            self.predicted_x[agent].append(x_state_mean.tolist())
            self.predicted_C[agent].append(C)

    def estimate_position_no_particle(self, time_n):
        for agent in range(self.dataset_params['num_agents']):
            x_state_mean = self.beliefs[agent][0]
            C = self.beliefs[agent][1]
            self.predicted_x[agent].append(x_state_mean.tolist())
            self.predicted_C[agent].append(C)

    def prediction_no_particle(self, time_n):
        for agent in range(self.dataset_params['num_agents']):
            self.prediction_message[agent] = [self.F@self.beliefs[agent][0], 
                                              self.Q + self.F@self.beliefs[agent][1]@self.F.transpose()]
            if agent == 0:
                print('Prediction mean: a {} {}'.format(agent, self.prediction_message[agent][0].squeeze().tolist()))
                print('Prediction var: a {}\n{}'.format(agent, self.prediction_message[agent][1]))

    def update_no_particle(self, dataset_instance, time_n):
        measurements = dataset_instance.measurements[time_n]
        mutual_distances = dataset_instance.mutual_distances[time_n]
        connectivity_matrix = dataset_instance.connectivity_matrix[time_n]

        # Noise gnss
        num_measurements = min(self.model_params['NUM_MEAS_GNSS'], self.model_params['number_state_variables'])
        diag_var_gnss = np.array([self.dataset_params['std_noise_gnss_position']**2, 
                        self.dataset_params['std_noise_gnss_position']**2,
                        self.dataset_params['std_noise_gnss_velocity']**2,
                        self.dataset_params['std_noise_gnss_velocity']**2,
                        self.dataset_params['std_noise_gnss_acceleration']**2,
                        self.dataset_params['std_noise_gnss_acceleration']**2][:num_measurements]).astype(float)
        diag_sigma_gnss = np.sqrt(diag_var_gnss)
        Cov_gnss = np.diag(diag_var_gnss)

        # No message passing, only gnss
        for i in range(self.dataset_params['num_agents']):
            self.beliefs[i] = copy(self.prediction_message[i])

            # evaluate likelihood gnss measurements
            gnss = np.array(dataset_instance.gnss[i][time_n][:num_measurements]).reshape([-1,1])
            H = np.eye(num_measurements)
            G = self.beliefs[i][1]@H.transpose()@np.linalg.inv(H@self.beliefs[i][1]@H.transpose() + Cov_gnss)
            self.measurement_gnss_message[i] = gnss

            self.beliefs[i][0] = self.beliefs[i][0] + G@(self.measurement_gnss_message[i] - H@self.beliefs[i][0])
            self.beliefs[i][1] = self.beliefs[i][1] - G@H@self.beliefs[i][1]   

    def compute_product_Gaussian_scalar(self, means_vars):
        num_Gaussian = len(means_vars)
        new_var = 0
        for g in range(num_Gaussian):
            new_var += 1./np.array(means_vars[g][1])
        new_var = np.reciprocal(new_var)

        new_mean = 0
        for g in range(num_Gaussian):
            new_mean += np.array(means_vars[g][0])/np.array(means_vars[g][1])
        new_mean = new_var*new_mean
        return [new_mean, new_var]

    def compute_product_Gaussian_vector(self, means_vars):
        num_Gaussian = len(means_vars)
        inv_covariances = []
        for g in range(num_Gaussian):
            inv_covariances.append(np.linalg.inv(np.array(means_vars[g][1])))
        inv_covariances = np.array(inv_covariances)
        new_var = np.linalg.inv(np.sum(inv_covariances, 0))

        new_mean = []
        for g in range(num_Gaussian):
            new_mean.append(inv_covariances[g]@np.array(means_vars[g][0]))
        new_mean = np.array(new_mean)
        new_mean = new_var@np.sum(new_mean, 0)
        return [new_mean, new_var]

    def compute_H(self, b1, b2):

        H = np.zeros((1, len(b1)))
        b1 = np.array(b1)
        b2 = np.array(b2)
        d = np.linalg.norm(b1[0:2]-b2[0:2])
        a = (b1[0:2]-b2[0:2])/d

        H[0,0] = -a[0]
        H[0,1] = -a[1]
        return H
    
    def compute_h(self, b1, b2):

        b1 = np.array(b1)
        b2 = np.array(b2)
        h = np.linalg.norm(b1[0:2]-b2[0:2])
        return h

    def compute_likelihood(self, type_, argument):
        if type_ == 'TOA':
            return np.exp(-0.5*(argument/np.sqrt(self.inter_agent_meas_model))**2)/np.sqrt(2*np.pi*self.inter_agent_meas_model)
        elif type_ == 'GNSS':
            return (1 / np.sqrt((2 * np.pi) ** self.Cov_gnss.shape[0] * np.linalg.det(self.Cov_gnss))) * \
                        np.exp(-0.5 * np.sum((argument) ** 2 * np.tile(self.diag_sigma_gnss.T ** (-2), (self.model_params['num_particles'], 1)).T, axis=0))

    def resampleSystematic(self,w,N):
        indx = np.zeros((N,1))
        Q = np.cumsum(w)
        T = np.linspace(0,1-1/N,N) + np.random.rand(1)/N
        T = np.append(T, 1)
        i=0
        j=0
        while i<N :
            if T[i]<Q[j]:
                indx[i]=j
                i=i+1
            else:
                j=j+1
        return indx
    
    def resampleSystematic2(self,w,particles):
        """
        Resamples particles based on their weights.
        
        Args:
        - particles: a numpy array of shape (n_particles, n_dims)
        - w: a numpy array of shape (n_particles,)
        
        Returns:
        - resampled_particles: a numpy array of shape (n_particles, n_dims)
        """
        w = w / np.sum(w)
    
        # Compute the cumulative sum of the weights
        cumulative_sum = np.cumsum(w)
        
        # Normalize the cumulative sum
        normalized_cumulative_sum = cumulative_sum / cumulative_sum[-1]
        
        # Initialize the resampled particles array
        resampled_particles = np.empty_like(particles)
        
        # Draw samples with replacement based on the weights
        for i in range(len(particles)):
            index = np.searchsorted(normalized_cumulative_sum, np.random.rand())
            resampled_particles[i] = particles[index]
        
        return resampled_particles

    def regularizeAgentParticles(self, samples):
        # regularizationVAs = parameters.regularizationVAs
        numParticles = samples.shape[1]
        uniqueParticles = len(np.unique(samples))
        covarianceMatrix = np.cov(samples) / uniqueParticles ** (1 / 3)
        samples = samples + np.random.multivariate_normal(np.zeros((samples.shape[0],)),covarianceMatrix+1e-8*np.eye(covarianceMatrix.shape[0]),numParticles).transpose()
        return samples
   
    def create_beliefs(self, dataset_instance):

        num_particles = self.model_params['num_particles']
        number_state_variables = self.model_params['number_state_variables']
        beliefs = {}
        if self.model_params['Particle_filter']:
            # beliefs[agent] = [particles, weights], particles = [x, y, vx, vy, ax, ay]

            for agent in range(self.dataset_params['num_agents']):
                beliefs[agent] = [[[] for i in range(number_state_variables)],[[] for i in range(number_state_variables)]]
            
                noise_angle = (np.random.uniform(-0.5, 0.5 ,(1, num_particles)))*2.*np.pi
                noise_range = np.random.uniform(0,1,(1, num_particles)) 
                try:
                    pos = np.array(dataset_instance.positions[agent][0]).reshape(-1,1) + 5*np.random.normal(size=(2, num_particles))
                except:
                    pass
                beliefs[agent][0][0] = pos[0]
                beliefs[agent][0][1] = pos[1]

                if number_state_variables >= 4:    
                    vel = (np.random.uniform(-0.5, 0.5 ,(2, num_particles)))*2*5
                    beliefs[agent][0][2] = vel[0]
                    beliefs[agent][0][3] = vel[1]

                    if number_state_variables >= 6:    
                        acc = (np.random.uniform(-0.5, 0.5 ,(2, num_particles)))*2*10
                        beliefs[agent][0][4] = acc[0]
                        beliefs[agent][0][5] = acc[1]

                beliefs[agent][0] = np.array(beliefs[agent][0])
                beliefs[agent][1] = np.array(np.ones((1,num_particles))/num_particles)
        else:
            beliefs = {agent:[np.pad(np.random.uniform(-100, 100, (2,1)), [(0,4), (0,0)]),np.diag([99**2,100**2,98,97,9,8])] for agent in range(self.dataset_params['num_agents'])}
            beliefs = {agent:[beliefs[agent][0][:number_state_variables],beliefs[agent][1][:number_state_variables, :number_state_variables]]   for agent in range(self.dataset_params['num_agents'])}

        self.beliefs = beliefs

