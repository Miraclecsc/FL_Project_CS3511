import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import dataloader
import os
import dill
from tqdm import tqdm

from model import LeNet
import argparse
import numpy as np


def dataset_load_from_disk(num_clients, client_data_path):
    train_dataset = []
    for i in range(num_clients):
        with open(os.path.join(client_data_path, f"Client{i+1}.pkl"), 'rb') as f:
            train_dataset_client = dill.load(f)
            train_dataset.append(train_dataset_client)
            
    with open(os.path.join(client_data_path, "Test.pkl"), 'rb') as f:
        test_dataset = dill.load(f)
    test_loader = dataloader.DataLoader(test_dataset, batch_size=64, shuffle=False)
    return train_dataset, test_loader

def train(num_clients=20):
    total_num_clients = 20
    if(num_clients > total_num_clients or num_clients < 1):
        print("Invalid number of clients")
        return
    client_data_path = "DATA_CIFAR10"
    batch_size = 64
    lr = 0.01
    epoch_num = 100
    single_epoch_num = 3
    best_acc = 0
    
    train_dataset, test_loader = dataset_load_from_disk(total_num_clients, client_data_path)
    server_model = LeNet()
    server_model_path = "stage2_model/server_model.pth"
    torch.save(server_model.state_dict(), server_model_path)
    chosen_time = [0 for i in range(total_num_clients)]
    for epoch in range(epoch_num):
        print("Epoch:", epoch+1)
        idx = np.random.choice(total_num_clients, num_clients, replace=False)
        for i in idx:
            chosen_time[i] += 1
        for i in tqdm(idx, desc="Training Clients"):
            client_model = LeNet()
            checkpoint = torch.load(server_model_path)
            client_model.load_state_dict(checkpoint)
            
            train_loader = dataloader.DataLoader(train_dataset[i], batch_size, shuffle=False)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(client_model.parameters(), lr=lr)
            for _ in range(single_epoch_num):
                for j, (inputs, labels) in enumerate(train_loader):
                    optimizer.zero_grad()
                    outputs = client_model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
            client_model_path = f"stage2_model/client{i+1}_model.pth"
            torch.save(client_model.state_dict(), client_model_path)
        
        avg_model = LeNet()
        avg_model.load_state_dict(torch.load(server_model_path))    
        for i in idx:
            client_model_path = f"stage2_model/client{i+1}_model.pth"
            client_model = LeNet()
            client_model.load_state_dict(torch.load(client_model_path))
            for avg_param, client_param in zip(avg_model.parameters(), client_model.parameters()):
                if i == idx[0]:
                    avg_param.data = client_param.data
                else:
                    avg_param.data += client_param.data
                    
        for avg_param in avg_model.parameters():
            avg_param.data /= num_clients

        server_model.load_state_dict(avg_model.state_dict())
        server_model.eval()
        accuracy = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = server_model(inputs)
                accuracy += (outputs.argmax(1) == labels).float().mean().item()
        accuracy /= len(test_loader)
        print("Test accuracy:", accuracy)
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(server_model.state_dict(), server_model_path)
            print("Model saved")
    return best_acc, chosen_time

    

def main():
    torch.manual_seed(0)
    np.random.seed(99)
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clients", type=int, default=20)
    args = parser.parse_args()
    acc, chosen_time = train(num_clients=args.num_clients)
    print("Best accuracy:", acc)
    for i in range(len(chosen_time)):
        print(f"Client {i+1} was chosen {chosen_time[i]} times")
    
if __name__ == "__main__":
    main()