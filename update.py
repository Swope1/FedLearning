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
    
    client_layers = [list(client_net.children()) for client_net in client_nets]
    
    num_weighted_layers = len(indices[0]) + 1
    weighted_layer_num = 0
    layer_num = 0
    
    for layer in server_net.children():
        if isinstance(layer, nn.Linear):
                        
            # set num_clients_per_weight to zero
            client_counts_weights = torch.ones(layer.weight.size()).to(DEVICE)
            client_counts_bias = torch.ones(layer.bias.size()).to(DEVICE)
            
            for client in range(NUM_CLIENTS):
                
                if weighted_layer_num == 0: # first layer
                    layer.weight.data[indices[client][0]] += client_layers[client][layer_num].weight.data
                    client_counts_weights[indices[client][0]] += 1
                    
                    layer.bias.data[indices[client][0]] += client_layers[client][layer_num].bias.data
                    client_counts_bias[indices[client][0]] += 1
                    
                elif weighted_layer_num == num_weighted_layers - 1: # last layer
                    for output in range(len(layer.weight.data)):
                        layer.weight.data[output][indices[client][weighted_layer_num - 1]] += client_layers[client][layer_num].weight.data[output]
                        client_counts_weights[output][indices[client][weighted_layer_num - 1]] += 1

                    layer.bias.data += client_layers[client][layer_num].bias.data / NUM_CLIENTS
                    client_counts_bias = None
                    
                else: # every other layer
                    for (index, weights) in zip(indices[client][weighted_layer_num], range(len(client_layers[client][layer_num].weight.data))):
                        layer.weight.data[index][indices[client][weighted_layer_num - 1]] += client_layers[client][layer_num].weight.data[weights]
                        client_counts_weights[index][indices[client][weighted_layer_num - 1]] += 1
                    
                    layer.bias.data[indices[client][weighted_layer_num]] += client_layers[client][layer_num].bias.data
                    client_counts_bias[indices[client][weighted_layer_num]] += 1
            
            # update weights
            layer.weight.data /= client_counts_weights
            if weighted_layer_num != num_weighted_layers - 1:
                layer.bias.data /= client_counts_bias

            weighted_layer_num += 1
        layer_num += 1
     
@torch.no_grad()   
def fed_avg_drouput_with_server_weights(server_net, client_nets, indices, client_weights):
    
    client_layers = [list(client_net.children()) for client_net in client_nets]
    
    num_weighted_layers = len(indices[0]) + 1
    weighted_layer_num = 0
    layer_num = 0
    
    for layer in server_net.children():
        if isinstance(layer, nn.Linear):
                        
            # set num_clients_per_weight to zero
            client_counts_weights = torch.ones(layer.weight.size()).to(DEVICE)
            client_counts_bias = torch.ones(layer.bias.size()).to(DEVICE)
            
            for client in range(NUM_CLIENTS):
                
                if weighted_layer_num == 0: # first layer
                    layer.weight.data[indices[client][0]] += client_layers[client][layer_num].weight.data * client_weights[client]
                    client_counts_weights[indices[client][0]] += client_weights[client]
                    
                    layer.bias.data[indices[client][0]] += client_layers[client][layer_num].bias.data * client_weights[client]
                    client_counts_bias[indices[client][0]] += client_weights[client]
                    
                elif weighted_layer_num == num_weighted_layers - 1: # last layer
                    for output in range(len(layer.weight.data)):
                        layer.weight.data[output][indices[client][weighted_layer_num - 1]] += client_layers[client][layer_num].weight.data[output] * client_weights[client]
                        client_counts_weights[output][indices[client][weighted_layer_num - 1]] += client_weights[client]

                    layer.bias.data += client_layers[client][layer_num].bias.data * client_weights[client]
                    client_counts_bias += client_weights[client]
                    
                else: # every other layer
                    for (index, weights) in zip(indices[client][weighted_layer_num], range(len(client_layers[client][layer_num].weight.data))):
                        layer.weight.data[index][indices[client][weighted_layer_num - 1]] += client_layers[client][layer_num].weight.data[weights] * client_weights[client]
                        client_counts_weights[index][indices[client][weighted_layer_num - 1]] += client_weights[client]
                    
                    layer.bias.data[indices[client][weighted_layer_num]] += client_layers[client][layer_num].bias.data * client_weights[client]
                    client_counts_bias[indices[client][weighted_layer_num]] += client_weights[client]
            
            # update weights
            layer.weight.data /= client_counts_weights
            layer.bias.data /= client_counts_bias

            weighted_layer_num += 1
        layer_num += 1