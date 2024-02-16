import torch
import math

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
    
    server_params = list(server_net.parameters())
    new_server_params = [torch.zeros(server_param.data.size()).to(DEVICE) for server_param in server_params]
    num_clients = [torch.zeros(server_param.data.size()).to(DEVICE) for server_param in server_params]
    client_nets_params = [list(client_net.parameters()) for client_net in client_nets]
    param_names = [named_param[0] for named_param in server_net.named_parameters()]
    
    for param_num in range(len(server_params)):
        for client_num in range(NUM_CLIENTS):
            if param_names[param_num].split('.')[1] == 'weight':
                if param_num == 0: # first layer
                    for i in range(len(indices[client_num][0])):
                        new_server_params[param_num][indices[client_num][0][i]] += client_nets_params[client_num][param_num][i]
                        num_clients[param_num][indices[client_num][0][i]] += torch.ones(client_nets_params[client_num][param_num][i].size()).to(DEVICE)
                elif param_num == len(param_names) - 2: # last layer
                    for i in range(len(new_server_params[param_num])):
                        new_server_params[param_num][i][indices[client_num][-1]] += client_nets_params[client_num][param_num][i]
                        num_clients[param_num][i][indices[client_num][-1]] += 1
                else: # middle layers
                    for (i,j) in zip(indices[client_num][param_num // 2], range(len(client_nets_params[client_num][param_num]))):
                        new_server_params[param_num][i][indices[client_num][(param_num // 2) - 1]] += client_nets_params[client_num][param_num][j]
                        num_clients[param_num][i][indices[client_num][(param_num // 2) - 1]] += 1
            
            else:
                if param_num == len(param_names) - 1: # last layer
                    new_server_params[param_num] += client_nets_params[client_num][param_num]
                    num_clients[param_num] += 1
                else: # first and middle layers
                    new_server_params[param_num][indices[client_num][param_num // 2]] += client_nets_params[client_num][param_num]
                    num_clients[param_num][indices[client_num][param_num // 2]] += 1

    for param_num in range(len(new_server_params)):
        if param_names[param_num].split('.')[1] == 'weight':
            zero_indices = (num_clients[param_num] == 0).nonzero()
            for i in zero_indices:
                new_server_params[param_num][i[0]][i[1]] = server_params[param_num][i[0]][i[1]]
                num_clients[param_num][i[0]][i[1]] = 1
        else: 
            zero_indices = (num_clients[param_num] == 0).nonzero()
            new_server_params[param_num][zero_indices] = server_params[param_num][zero_indices]
            num_clients[param_num][zero_indices] = 1
        new_server_params[param_num] = torch.div(new_server_params[param_num], num_clients[param_num])

    for (server_param_num, server_param) in enumerate(server_net.parameters()):
        server_param.data = new_server_params[server_param_num]

@torch.no_grad()
def fed_avg_drouput_with_server(server_net, client_nets, indices):
    
    server_params = list(server_net.parameters())
    # new_server_params = [torch.zeros(server_param.data.size()).to(DEVICE) for server_param in server_params]
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
    # new_server_params = [torch.zeros(server_param.data.size()).to(DEVICE) for server_param in server_params]
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