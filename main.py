import torch
from copy import deepcopy
import scipy.stats as stats

from model import Net, Net2, ConvNet, ConvNet2, train, test, DEVICE
from clients import create_clients, generate_indices_first, generate_indices_random, NUM_CLIENTS
from data import load_data, load_data_server, load_data_flwr, load_data_server_flwr
from update import fed_avg, fed_avg_drouput, fed_avg_drouput_with_server, fed_avg_drouput_with_server_weights

NUM_ROUNDS = 80
NUM_EPOCHS_PER_ROUND = 5
LR_START = 0.001
LR_DISCOUT = 1
CLIENT_VERBOSE = False
SERVER_VERBOSE = True

loaders = load_data()

if SERVER_VERBOSE:
    server_loader = load_data_server()

server_net = Net().to(DEVICE)
# server_net = Net2().to(DEVICE)
# server_net = ConvNet().to(DEVICE)
# server_net = ConvNet2().to(DEVICE)

# torch.set_printoptions(threshold=10)

for round in range(NUM_ROUNDS):
    print("Round: " + str(round))
    
    # indices = generate_indices_first(server_net, 0.5)
    indices = generate_indices_random(server_net, 0.5)
    
    # client_nets = [deepcopy(server_net) for _ in range(NUM_CLIENTS)]
    client_nets = create_clients(server_net, indices)
    
    optimizers = [torch.optim.SGD(client_net.parameters(), lr=LR_START * (LR_DISCOUT ** round)) for client_net in client_nets]
    
    for epoch in range(NUM_EPOCHS_PER_ROUND):
        client_losses = []
        if CLIENT_VERBOSE:
            print("Epoch: " + str(epoch))
        for client_num, client_net in enumerate(client_nets):
            if CLIENT_VERBOSE:
                print("Client: " + str(client_num))
                
            train(client_net, loaders[client_num][0], optimizers[client_num], torch.nn.CrossEntropyLoss())

            if CLIENT_VERBOSE:
                loss, accuracy = test(client_net, loaders[client_num][1], torch.nn.CrossEntropyLoss())
                client_losses.append(loss)
                print('Loss: ' + str(loss) + ' Accuracy: ' + str(accuracy))
    
    # fed_avg(server_net, client_nets)
    # fed_avg_drouput(server_net, client_nets, indices)
    fed_avg_drouput_with_server(server_net, client_nets, indices)
    
    # client_weights = stats.zscore(client_losses)
    # client_weights += abs(min(client_weights)) + 1
    # fed_avg_drouput_with_server_weights(server_net, client_nets, indices, client_weights)
    
    if SERVER_VERBOSE:
        loss, accuracy = test(server_net, server_loader[0], torch.nn.CrossEntropyLoss())
        print("Server Train:")
        print('Loss: ' + str(loss) + ' Accuracy: ' + str(accuracy))
        loss, accuracy = test(server_net, server_loader[1], torch.nn.CrossEntropyLoss())
        print("Server:")
        print('Loss: ' + str(loss) + ' Accuracy: ' + str(accuracy))