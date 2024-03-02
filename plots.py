import json
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import seaborn as sns

NUM_EPOCHS_PER_ROUND = 5
NUM_ROUNDS = 40
Methods = ['FedRolex','FedDropout']
IID_Levels = [2,4,6,8,10]
Dropout_levels = [0.25, 0.5, 0.75, 1]

colors = [['tab:blue','b'], ['tab:orange', 'orange'], ['tab:green','g'], ['tab:red', 'r'], ['tab:purple', 'm']]

# colors_long = [list(mcolors.XKCD_COLORS)[i] for i in np.random.randint(0, len(mcolors.XKCD_COLORS), len(Methods) * len(IID_Levels) * len(Dropout_levels))]
# sns.reset_orig()
# colors_long = sns.color_palette('husl', n_colors=len(Methods) * len(IID_Levels) * len(Dropout_levels))

client_train_losses = []
client_train_accuracies = []
client_test_losses = []
client_test_accuracies = []
server_test_losses = []
server_test_accuracies = []

for method_num, method in enumerate(Methods):
    for iid_num, iid_level in enumerate(IID_Levels):
        for dropout_num, dropout_level in enumerate(Dropout_levels):
            index = (method_num * len(IID_Levels) * len(Dropout_levels)) + (iid_num * len(Dropout_levels)) + dropout_num
            file_name = 'outputs\losses_and_accuracies_' + method + '_' + str(dropout_level) + '_' + str(iid_level) + '_' + str(NUM_EPOCHS_PER_ROUND) + '_' + str(NUM_ROUNDS)

            with open(file_name, 'r') as f:
                data = json.load(f)
            
            client_train_losses.append(data['client_train_losses'])
            client_train_accuracies.append(data['client_train_accuracies'])
            client_test_losses.append(data['client_test_losses'])
            client_test_accuracies.append(data['client_test_accuracies'])
            server_test_losses.append(data['server_test_losses'])
            server_test_accuracies.append(data['server_test_accuracies'])
            
            for epoch in range(len(client_test_accuracies[index])):
                client_train_losses[index][epoch] = sum(client_train_losses[index][epoch])/len(client_train_losses[index][epoch])
                client_train_accuracies[index][epoch] = sum(client_train_accuracies[index][epoch])/len(client_train_accuracies[index][epoch])
                client_test_losses[index][epoch] = sum(client_test_losses[index][epoch])/len(client_test_losses[index][epoch])
                client_test_accuracies[index][epoch] = sum(client_test_accuracies[index][epoch])/len(client_test_accuracies[index][epoch])
        
        fig, axs = plt.subplots(1,2, figsize=(15, 6))
        fig.suptitle(method + ' with ' + str(iid_level) + ' classes per client')
        
        for dropout_num, dropout_level in enumerate(Dropout_levels):
            index = (method_num * len(IID_Levels) * len(Dropout_levels)) + (iid_num * len(Dropout_levels)) + dropout_num
            axs[0].plot(list(range(len(client_train_accuracies[index]))), client_train_accuracies[index], c=colors[dropout_num][1], linewidth=0.25)
            axs[0].plot(list(range(len(client_test_accuracies[index]))), client_test_accuracies[index], c=colors[dropout_num][0], linewidth=0.25)
            axs[0].plot(list(range(0, len(server_test_accuracies[index]) * NUM_EPOCHS_PER_ROUND, NUM_EPOCHS_PER_ROUND)), server_test_accuracies[index], c=colors[dropout_num][0], linewidth=2, label = str(dropout_level))
            
            axs[1].plot(list(range(len(client_train_losses[index]))), client_train_losses[index], c=colors[dropout_num][1], linewidth=0.25)
            axs[1].plot(list(range(len(client_test_losses[index]))), client_test_losses[index], c=colors[dropout_num][0], linewidth=0.25)
            axs[1].plot(list(range(0, len(server_test_losses[index]) * NUM_EPOCHS_PER_ROUND, NUM_EPOCHS_PER_ROUND)), server_test_losses[index], c=colors[dropout_num][0], linewidth=2, label = str(dropout_level))
        
        axs[0].set_ylim(0,1)
        axs[0].set_ylabel('Accuracy')
        axs[0].set_xlabel('Epoch')
        axs[0].set_title('Accuracy vs Epoch')
        axs[0].legend()
        
        axs[1].set_ylabel('Loss')
        axs[1].set_xlabel('Epoch')
        axs[1].set_title('Loss vs Epoch')
        axs[1].legend()
        
        file_name = 'plots\\' + method + '_IID_' + str(iid_level)

        plt.savefig(file_name, bbox_inches='tight')
        plt.clf()
    
    
    for dropout_num, dropout_level in enumerate(Dropout_levels):
        fig, axs = plt.subplots(1,2, figsize=(15, 6))
        fig.suptitle(method + ' with ' + str(int(dropout_level*100)) + ' dropout')
        
        for iid_num, iid_level in enumerate(IID_Levels):
            index = (method_num * len(IID_Levels) * len(Dropout_levels)) + (iid_num * len(Dropout_levels)) + dropout_num
            axs[0].plot(list(range(len(client_train_accuracies[index]))), client_train_accuracies[index], c=colors[iid_num][1], linewidth=0.25)
            axs[0].plot(list(range(len(client_test_accuracies[index]))), client_test_accuracies[index], c=colors[iid_num][0], linewidth=0.25)
            axs[0].plot(list(range(0, len(server_test_accuracies[index]) * NUM_EPOCHS_PER_ROUND, NUM_EPOCHS_PER_ROUND)), server_test_accuracies[index], c=colors[iid_num][0], linewidth=2, label = str(iid_level))
            
            axs[1].plot(list(range(len(client_train_losses[index]))), client_train_losses[index], c=colors[iid_num][1], linewidth=0.25)
            axs[1].plot(list(range(len(client_test_losses[index]))), client_test_losses[index], c=colors[iid_num][0], linewidth=0.25)
            axs[1].plot(list(range(0, len(server_test_losses[index]) * NUM_EPOCHS_PER_ROUND, NUM_EPOCHS_PER_ROUND)), server_test_losses[index], c=colors[iid_num][0], linewidth=2, label = str(iid_level))
            
        axs[0].set_ylim(0,1)
        axs[0].set_ylabel('Accuracy')
        axs[0].set_xlabel('Epoch')
        axs[0].set_title('Accuracy vs Epoch')
        axs[0].legend()
        
        axs[1].set_ylabel('Loss')
        axs[1].set_xlabel('Epoch')
        axs[1].set_title('Loss vs Epoch')
        axs[1].legend()
        
        file_name = 'plots\\' + method + '_dropoutLevel_' + str(int(dropout_level * 100))

        plt.savefig(file_name, bbox_inches='tight')
        plt.clf()
plt.close()

for method_num, method in enumerate(Methods):
    fig, axs = plt.subplots(2,len(IID_Levels), figsize=(30,12), sharey='row')
    fig.suptitle(method)
    for iid_num, iid_level in enumerate(IID_Levels):
        for dropout_num, dropout_level in enumerate(Dropout_levels):
            index = (method_num * len(IID_Levels) * len(Dropout_levels)) + (iid_num * len(Dropout_levels)) + dropout_num
            axs[0][iid_num].plot(list(range(0, len(server_test_accuracies[index]) * NUM_EPOCHS_PER_ROUND, NUM_EPOCHS_PER_ROUND)), server_test_accuracies[index], c=colors[dropout_num][0], linewidth=2, label = dropout_level)
            axs[1][iid_num].plot(list(range(0, len(server_test_losses[index]) * NUM_EPOCHS_PER_ROUND, NUM_EPOCHS_PER_ROUND)), server_test_losses[index], c=colors[dropout_num][0], linewidth=2, label = dropout_level)

        axs[0][iid_num].set_ylim(0,1)
        axs[0][iid_num].set_ylabel('Accuracy')
        axs[0][iid_num].set_xlabel('Epoch')
        axs[0][iid_num].set_title('Accuracy vs Epoch with ' + str(iid_level) + ' classes per client')
        axs[0][iid_num].legend()

        axs[1][iid_num].set_ylabel('Loss')
        axs[1][iid_num].set_xlabel('Epoch')
        axs[1][iid_num].set_title('Loss vs Epoch with ' + str(iid_level) + ' classes per client')
        axs[1][iid_num].legend()

    file_name = 'plots\\' + method + '_byIID'
    plt.savefig(file_name, bbox_inches='tight')
    plt.clf()
    plt.close()
    
    fig, axs = plt.subplots(2,len(Dropout_levels), figsize=(30,12), sharey='row')
    fig.suptitle(method)
    for dropout_num, dropout_level in enumerate(Dropout_levels):
        for iid_num, iid_level in enumerate(IID_Levels):
            index = (method_num * len(IID_Levels) * len(Dropout_levels)) + (iid_num * len(Dropout_levels)) + dropout_num
            axs[0][dropout_num].plot(list(range(0, len(server_test_accuracies[index]) * NUM_EPOCHS_PER_ROUND, NUM_EPOCHS_PER_ROUND)), server_test_accuracies[index], c=colors[iid_num][0], linewidth=2, label = iid_level)
            axs[1][dropout_num].plot(list(range(0, len(server_test_losses[index]) * NUM_EPOCHS_PER_ROUND, NUM_EPOCHS_PER_ROUND)), server_test_losses[index], c=colors[iid_num][0], linewidth=2, label = iid_level)

        axs[0][dropout_num].set_ylim(0,1)
        axs[0][dropout_num].set_ylabel('Accuracy')
        axs[0][dropout_num].set_xlabel('Epoch')
        axs[0][dropout_num].set_title('Accuracy vs Epoch with ' + str(dropout_level) + ' dropout per client')
        axs[0][dropout_num].legend()

        axs[1][dropout_num].set_ylabel('Loss')
        axs[1][dropout_num].set_xlabel('Epoch')
        axs[1][dropout_num].set_title('Loss vs Epoch with ' + str(dropout_level) + ' dropout per client')
        axs[1][dropout_num].legend()

    file_name = 'plots\\' + method + '_byDropoutLevel'
    plt.savefig(file_name, bbox_inches='tight')
    plt.clf()
    plt.close()
    
    
    fig, axs = plt.subplots(2,len(IID_Levels), figsize=(30,12), sharey='row')
    fig.suptitle(method)
    for iid_num, iid_level in enumerate(IID_Levels):
        for dropout_num, dropout_level in enumerate(Dropout_levels):
            index = (method_num * len(IID_Levels) * len(Dropout_levels)) + (iid_num * len(Dropout_levels)) + dropout_num
            axs[0][iid_num].plot(list(range(len(client_train_accuracies[index]))), client_train_accuracies[index], c=colors[dropout_num][1], linewidth=0.25)
            axs[0][iid_num].plot(list(range(len(client_test_accuracies[index]))), client_test_accuracies[index], c=colors[dropout_num][0], linewidth=0.25)
            axs[0][iid_num].plot(list(range(0, len(server_test_accuracies[index]) * NUM_EPOCHS_PER_ROUND, NUM_EPOCHS_PER_ROUND)), server_test_accuracies[index], c=colors[dropout_num][0], linewidth=2, label = dropout_level)
            
            axs[1][iid_num].plot(list(range(len(client_train_losses[index]))), client_train_losses[index], c=colors[dropout_num][1], linewidth=0.25)
            axs[1][iid_num].plot(list(range(len(client_test_losses[index]))), client_test_losses[index], c=colors[dropout_num][0], linewidth=0.25)
            axs[1][iid_num].plot(list(range(0, len(server_test_losses[index]) * NUM_EPOCHS_PER_ROUND, NUM_EPOCHS_PER_ROUND)), server_test_losses[index], c=colors[dropout_num][0], linewidth=2, label = dropout_level)

        axs[0][iid_num].set_ylim(0,1)
        axs[0][iid_num].set_ylabel('Accuracy')
        axs[0][iid_num].set_xlabel('Epoch')
        axs[0][iid_num].set_title('Accuracy vs Epoch with ' + str(iid_level) + ' classes per client')
        axs[0][iid_num].legend()

        axs[1][iid_num].set_ylabel('Loss')
        axs[1][iid_num].set_xlabel('Epoch')
        axs[1][iid_num].set_title('Loss vs Epoch with ' + str(iid_level) + ' classes per client')
        axs[1][iid_num].legend()

    file_name = 'plots\\' + method + '_byIID_all'
    plt.savefig(file_name, bbox_inches='tight')
    plt.clf()
    plt.close()
    
    fig, axs = plt.subplots(2,len(Dropout_levels), figsize=(30,12), sharey='row')
    fig.suptitle(method)
    for dropout_num, dropout_level in enumerate(Dropout_levels):
        for iid_num, iid_level in enumerate(IID_Levels):
            index = (method_num * len(IID_Levels) * len(Dropout_levels)) + (iid_num * len(Dropout_levels)) + dropout_num
            axs[0][dropout_num].plot(list(range(len(client_train_accuracies[index]))), client_train_accuracies[index], c=colors[iid_num][1], linewidth=0.25)
            axs[0][dropout_num].plot(list(range(len(client_test_accuracies[index]))), client_test_accuracies[index], c=colors[iid_num][0], linewidth=0.25)
            axs[0][dropout_num].plot(list(range(0, len(server_test_accuracies[index]) * NUM_EPOCHS_PER_ROUND, NUM_EPOCHS_PER_ROUND)), server_test_accuracies[index], c=colors[iid_num][0], linewidth=2, label = iid_level)
            
            axs[1][dropout_num].plot(list(range(len(client_train_losses[index]))), client_train_losses[index], c=colors[iid_num][1], linewidth=0.25)
            axs[1][dropout_num].plot(list(range(len(client_test_losses[index]))), client_test_losses[index], c=colors[iid_num][0], linewidth=0.25)
            axs[1][dropout_num].plot(list(range(0, len(server_test_losses[index]) * NUM_EPOCHS_PER_ROUND, NUM_EPOCHS_PER_ROUND)), server_test_losses[index], c=colors[iid_num][0], linewidth=2, label = iid_level)

        axs[0][dropout_num].set_ylim(0,1)
        axs[0][dropout_num].set_ylabel('Accuracy')
        axs[0][dropout_num].set_xlabel('Epoch')
        axs[0][dropout_num].set_title('Accuracy vs Epoch with ' + str(dropout_level) + ' dropout per client')
        axs[0][dropout_num].legend()

        axs[1][dropout_num].set_ylabel('Loss')
        axs[1][dropout_num].set_xlabel('Epoch')
        axs[1][dropout_num].set_title('Loss vs Epoch with ' + str(dropout_level) + ' dropout per client')
        axs[1][dropout_num].legend()

    file_name = 'plots\\' + method + '_byDropoutLevel_all'
    plt.savefig(file_name, bbox_inches='tight')
    plt.clf()
    plt.close()