import torch
import random
from copy import deepcopy

NUM_CLIENTS = 3

def generate_indices_first(server_net, prop_to_keep):
    indices = [[] for _ in range(NUM_CLIENTS)]
    num_layers = len(list(server_net.parameters()))//2
    for (name, param) in server_net.named_parameters():
        param_type = name.split('.')[1]
        if param_type == 'bias' and len(indices[0]) < num_layers - 1:
            num_neurons = param.data.size(0)
            for client_num in range(NUM_CLIENTS):
                indices[client_num].append(list(range(int(num_neurons * prop_to_keep))))
    return indices

def generate_indices_random(server_net, prop_to_keep):
    indices = [[] for _ in range(NUM_CLIENTS)]
    num_layers = len(list(server_net.parameters()))//2
    for (name, param) in server_net.named_parameters():
        param_type = name.split('.')[1]
        if param_type == 'bias' and len(indices[0]) < num_layers - 1:
            num_neurons = param.data.size(0)
            for client_num in range(NUM_CLIENTS):
                indices[client_num].append(random.sample(range(num_neurons), int(num_neurons * prop_to_keep)))
    return indices

def create_clients(server_net, indices):
    client_nets = [deepcopy(server_net) for _ in range(NUM_CLIENTS)]

    for (client_num, client_net) in enumerate(client_nets):
        for (param_num, (name, client_param)) in enumerate(client_net.named_parameters()):
            
            if param_num < 2 * len(indices[client_num]): # filter output weights
                client_param.data = client_param.data[indices[client_num][param_num//2]]
                
            if name.split('.')[1] == 'weight' and param_num != 0: # filter input weights
                new_param = []
                for output_weight in client_param.data:
                    new_param.append(output_weight[indices[client_num][(param_num//2)-1]])
                # print(new_param)
                new_param = torch.stack(new_param, 0)
                client_param.data = new_param
                
    return client_nets