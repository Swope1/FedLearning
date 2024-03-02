import torch
from copy import deepcopy
import scipy.stats as stats
import json
import math

from model import MNIST_Net, CIFAR10_Net, CIFAR10_Net2, ConvNet, ConvNet2, plot_pr_curves, train, test, DEVICE
from clients import create_clients, generate_indices_first, generate_indices_random, generate_indices_rolex, NUM_CLIENTS
from data import load_data_MNIST, load_data_non_iid_MNIST, load_pr_data_MNIST, load_data_CIFAR10, load_data_non_iid_CIFAR10, load_data_server_CIFAR10, load_pr_data_CIFAR10, load_data_flwr, load_data_server_flwr, NUM_CLASSES
from update import fed_avg, fed_avg_drouput, fed_avg_drouput_with_server, fed_avg_drouput_with_server_weights

NUM_ROUNDS = 40
NUM_EPOCHS_PER_ROUND = 5

NUM_CLASSES_PER_CLIENT = 10

DROPOUT_METHOD = 'FedRolex'
DROPOUT_PROPORTION = .25
CLIENT_STEP = .2
ROUND_STEP = .2

LR_START = 1e-5
LR_DISCOUT = .95

EARLY_STOPPING = False
EARLY_STOPPING_METRIC = 'acc'
EARLY_STOPPING_PATIENCE = 5

BEST_MODEL_METRIC = 'acc'

CLIENT_VERBOSE = False
SERVER_VERBOSE = True

SEED = 1

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# loaders, server_loader = load_data_MNIST()
loaders, server_loader = load_data_non_iid_MNIST(NUM_CLASSES_PER_CLIENT)

server_net = MNIST_Net().to(DEVICE)

client_train_losses = []
client_train_accuracies = []
client_test_losses = []
client_test_accuracies = []
server_test_losses = []
server_test_accuracies = []

min_loss = math.inf
max_accuracy = -math.inf
best_server_net = None

# torch.set_printoptions(threshold=10)

for round in range(NUM_ROUNDS):
    print('Round: ' + str(round))
    
    if DROPOUT_METHOD == 'FedAvg':
        client_nets = [deepcopy(server_net) for _ in range(NUM_CLIENTS)]
    if DROPOUT_METHOD == 'FedDropout':
        indices = generate_indices_random(server_net, DROPOUT_PROPORTION)
        client_nets = create_clients(server_net, indices)
    if DROPOUT_METHOD == 'FedRolex':
        indices = generate_indices_rolex(server_net, DROPOUT_PROPORTION, CLIENT_STEP, ROUND_STEP, round)
        client_nets = create_clients(server_net, indices)
    
    optimizers = [torch.optim.Adam(client_net.parameters(), lr=LR_START * (LR_DISCOUT ** round)) for client_net in client_nets]
    
    for epoch in range(NUM_EPOCHS_PER_ROUND):
        epoch_client_train_losses = []
        epoch_client_train_accuracies = []
        epoch_client_test_losses = []
        epoch_client_test_accuracies = []
        if CLIENT_VERBOSE:
            print('Epoch: ' + str(epoch))
        for client_num, client_net in enumerate(client_nets):
            if CLIENT_VERBOSE:
                print('Client: ' + str(client_num))
                
            train_loss, train_accuracy = train(client_net, loaders[client_num][0], optimizers[client_num], torch.nn.CrossEntropyLoss())
            test_loss, test_accuracy = test(client_net, loaders[client_num][1], torch.nn.CrossEntropyLoss())
            
            epoch_client_train_losses.append(train_loss)
            epoch_client_train_accuracies.append(train_accuracy)
            epoch_client_test_losses.append(test_loss)
            epoch_client_test_accuracies.append(test_accuracy)
            
            if CLIENT_VERBOSE:
                print('Train Loss: ' + str(train_loss) + ' Train Accuracy: ' + str(train_accuracy) + ' Test Loss: ' + str(test_loss) + ' Test Accuracy: ' + str(test_accuracy))
        
        client_train_losses.append(epoch_client_train_losses)
        client_train_accuracies.append(epoch_client_train_accuracies)
        client_test_losses.append(epoch_client_test_losses)
        client_test_accuracies.append(epoch_client_test_accuracies)
    
    
    if DROPOUT_METHOD == 'FedAvg':
        fed_avg(server_net, client_nets)
    if DROPOUT_METHOD == 'FedDropout' or DROPOUT_METHOD == 'FedRolex':
        fed_avg_drouput(server_net, client_nets, indices)
    
    server_loss, server_accuracy = test(server_net, server_loader, torch.nn.CrossEntropyLoss())
    
    server_test_losses.append(server_loss)
    server_test_accuracies.append(server_accuracy)
    
    if BEST_MODEL_METRIC == 'loss':
        if server_loss < min_loss:
            min_loss = server_loss
            best_server_net = deepcopy(server_net)
    elif BEST_MODEL_METRIC == 'acc':
        if server_accuracy > max_accuracy:
            max_accuracy = server_accuracy
            best_server_net = deepcopy(server_net)
    
    if SERVER_VERBOSE:
        print('Server:')
        print('Loss: ' + str(server_loss) + ' Accuracy: ' + str(server_accuracy))
    
    # check if we need to stop early
    if EARLY_STOPPING:
        early_stopping_counter = 0
        for patience_value in range(EARLY_STOPPING_PATIENCE + 1):
            if EARLY_STOPPING_METRIC == 'loss':
                if round > EARLY_STOPPING_PATIENCE and server_test_losses[-(patience_value + 2)] < server_test_losses[-(patience_value + 1)]:
                    early_stopping_counter += 1
            elif EARLY_STOPPING_METRIC == 'acc':
                if round > EARLY_STOPPING_PATIENCE and server_test_accuracies[-(patience_value + 2)] > server_test_accuracies[-(patience_value + 1)]:
                    early_stopping_counter += 1
        if EARLY_STOPPING_PATIENCE + 1 == early_stopping_counter:
            break

json_data = {
    'client_train_losses': client_train_losses,
    'client_train_accuracies': client_train_accuracies,
    'client_test_losses': client_test_losses,
    'client_test_accuracies': client_test_accuracies,
    'server_test_losses': server_test_losses,
    'server_test_accuracies': server_test_accuracies,
}

file_name = 'outputs\losses_and_accuracies_' + DROPOUT_METHOD + '2_' + str(DROPOUT_PROPORTION) + '_' + str(NUM_CLASSES_PER_CLIENT) + '_' + str(NUM_EPOCHS_PER_ROUND) + '_' + str(NUM_ROUNDS)

with open(file_name, 'w') as f:
    json.dump(json_data, f)
    
file_name = 'saved_models\model_' + DROPOUT_METHOD + '2_' + str(DROPOUT_PROPORTION) + '_' + str(NUM_CLASSES_PER_CLIENT) + '_' + str(NUM_EPOCHS_PER_ROUND) + '_' + str(NUM_ROUNDS)
    
torch.save(best_server_net, file_name)
    
plot_pr_curves(best_server_net)