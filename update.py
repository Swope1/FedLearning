import torch
import torch.nn as nn
import math
from copy import deepcopy

from clients import NUM_CLIENTS
from model import DEVICE

@torch.no_grad()
def fed_avg(server_net, client_nets):
    client_params = [list(client_net.parameters()) for client_net in client_nets]
    for (param_num, server_param) in enumerate(server_net.parameters()):
        new_param = torch.zeros(server_param.data.size()).to(DEVICE)
        for client in client_params:
            new_param += client[param_num].data
        server_param.data = new_param/NUM_CLIENTS
        
@torch.no_grad()
def fed_avg_drouput(server_net, client_nets, indices):
    
    # server_layers = list(server_net.children())
    new_server_layers = deepcopy(list(server_net.children()))
    client_layers = [list(client_net.children()) for client_net in client_nets]
    
    num_weighted_layers = len(indices[0]) + 1
    weighted_layer_num = 0
    layer_num = 0
    
    for layer in server_net.children():
        if isinstance(layer, nn.Linear):
            
            # set new_server_layer weights (and biases) to zero
            new_server_layers[layer_num].weight.data = torch.zeros(new_server_layers[layer_num].weight.size()).to(DEVICE)
            new_server_layers[layer_num].bias.data = torch.zeros(new_server_layers[layer_num].bias.size()).to(DEVICE)
            
            # set num_clients_per_weight to zero
            client_counts_weights = torch.zeros(layer.weight.size()).to(DEVICE)
            client_counts_bias = torch.zeros(layer.bias.size()).to(DEVICE)
            
            for client in range(NUM_CLIENTS):
                
                if weighted_layer_num == 0: # first layer
                    new_server_layers[layer_num].weight.data[indices[client][0]] += client_layers[client][layer_num].weight.data
                    client_counts_weights[indices[client][0]] += 1
                    
                    new_server_layers[layer_num].bias.data[indices[client][0]] += client_layers[client][layer_num].bias.data
                    client_counts_bias[indices[client][0]] += 1
                    
                elif weighted_layer_num == num_weighted_layers - 1: # last layer
                    for output in range(len(new_server_layers[layer_num].weight.data)):
                        new_server_layers[layer_num].weight.data[output][indices[client][weighted_layer_num - 1]] += client_layers[client][layer_num].weight.data[output]
                        client_counts_weights[output][indices[client][weighted_layer_num - 1]] += 1

                    new_server_layers[layer_num].bias.data += client_layers[client][layer_num].bias.data / NUM_CLIENTS
                    client_counts_bias = None
                    
                else: # every other layer
                    for (index, weights) in zip(indices[client][weighted_layer_num], range(len(client_layers[client][layer_num].weight.data))):
                        new_server_layers[layer_num].weight.data[index][indices[client][weighted_layer_num - 1]] += client_layers[client][layer_num].weight.data[weights]
                        client_counts_weights[index][indices[client][weighted_layer_num - 1]] += 1
                    
                    new_server_layers[layer_num].bias.data[indices[client][weighted_layer_num]] += client_layers[client][layer_num].bias.data
                    client_counts_bias[indices[client][weighted_layer_num]] += 1

            # check for and correct zero weights
            zero_indices_weights = (client_counts_weights == 0).nonzero()
            for index in zero_indices_weights:
                new_server_layers[layer_num].weight.data[index[0]][index[1]] = layer.weight.data[index[0]][index[1]]
                client_counts_weights[index[0]][index[1]] = 1
                
            if weighted_layer_num != num_weighted_layers - 1:
                zero_indices_bias = (client_counts_bias == 0).nonzero()
                new_server_layers[layer_num].bias.data[zero_indices_bias] = layer.bias.data[zero_indices_bias]
                client_counts_bias[zero_indices_bias] = 1
            
            # update weights
            new_server_layers[layer_num].weight.data /= client_counts_weights
            if weighted_layer_num != num_weighted_layers - 1:
                new_server_layers[layer_num].bias.data /= client_counts_bias
            
            layer.weight.data = new_server_layers[layer_num].weight.data
            layer.bias.data = new_server_layers[layer_num].bias.data

            weighted_layer_num += 1
        layer_num += 1
        

@torch.no_grad()
def fed_avg_drouput_with_server(server_net, client_nets, indices):
    
    server_params = list(server_net.parameters())
    num_clients = [torch.ones(server_param.data.size()).to(DEVICE) for server_param in server_params]
    client_nets_params = [list(client_net.parameters()) for client_net in client_nets]
    param_names = [named_param[0] for named_param in server_net.named_parameters()]
    
    for param_num in range(len(server_params)):
        for client_num in range(NUM_CLIENTS):
            if param_names[param_num].split('.')[1] == 'weight':
                if param_num == 0: # first layer
                    for i in range(len(indices[client_num][0])):
                        server_params[param_num][indices[client_num][0][i]] += client_nets_params[client_num][param_num][i]
                        num_clients[param_num][indices[client_num][0][i]] += 1
                elif param_num == len(param_names) - 2: # last layer
                    for i in range(len(server_params[param_num])):
                        server_params[param_num][i][indices[client_num][-1]] += client_nets_params[client_num][param_num][i]
                        num_clients[param_num][i][indices[client_num][-1]] += 1
                else: # middle layers
                    for (i,j) in zip(indices[client_num][param_num // 2], range(len(client_nets_params[client_num][param_num]))):
                        server_params[param_num][i][indices[client_num][(param_num // 2) - 1]] += client_nets_params[client_num][param_num][j]
                        num_clients[param_num][i][indices[client_num][(param_num // 2) - 1]] += 1
            
            else:
                if param_num == len(param_names) - 1: # last layer
                    server_params[param_num] += client_nets_params[client_num][param_num]
                    num_clients[param_num] += 1
                else: # first and middle layers
                    server_params[param_num][indices[client_num][param_num // 2]] += client_nets_params[client_num][param_num]
                    num_clients[param_num][indices[client_num][param_num // 2]] += 1

    for (server_param_num, server_param) in enumerate(server_net.parameters()):
        server_param.data = torch.div(server_params[server_param_num], num_clients[server_param_num])
     
@torch.no_grad()   
def fed_avg_drouput_with_server_weights(server_net, client_nets, indices, client_weights):
    
    server_params = list(server_net.parameters())
    num_clients = [torch.ones(server_param.data.size()).to(DEVICE) for server_param in server_params]
    client_nets_params = [list(client_net.parameters()) for client_net in client_nets]
    param_names = [named_param[0] for named_param in server_net.named_parameters()]
    
    for param_num in range(len(server_params)):
        for client_num in range(NUM_CLIENTS):
            if param_names[param_num].split('.')[1] == 'weight':
                if param_num == 0: # first layer
                    for i in range(len(indices[client_num][0])):
                        server_params[param_num][indices[client_num][0][i]] += client_nets_params[client_num][param_num][i] * client_weights[client_num]
                        num_clients[param_num][indices[client_num][0][i]] += client_weights[client_num]
                elif param_num == len(param_names) - 2: # last layer
                    for i in range(len(server_params[param_num])):
                        server_params[param_num][i][indices[client_num][-1]] += client_nets_params[client_num][param_num][i] * client_weights[client_num]
                        num_clients[param_num][i][indices[client_num][-1]] += client_weights[client_num]
                else: # middle layers
                    for (i,j) in zip(indices[client_num][param_num // 2], range(len(client_nets_params[client_num][param_num]))):
                        server_params[param_num][i][indices[client_num][(param_num // 2) - 1]] += client_nets_params[client_num][param_num][j] * client_weights[client_num]
                        num_clients[param_num][i][indices[client_num][(param_num // 2) - 1]] += client_weights[client_num]
            
            else:
                if param_num == len(param_names) - 1: # last layer
                    server_params[param_num] += client_nets_params[client_num][param_num]
                    num_clients[param_num] += 1
                else: # first and middle layers
                    server_params[param_num][indices[client_num][param_num // 2]] += client_nets_params[client_num][param_num]
                    num_clients[param_num][indices[client_num][param_num // 2]] += 1

    for (server_param_num, server_param) in enumerate(server_net.parameters()):
        server_param.data = torch.div(server_params[server_param_num], num_clients[server_param_num])