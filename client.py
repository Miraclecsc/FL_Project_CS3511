import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import dataloader
import os
import io
import dill
from tqdm import tqdm
import socket
import numpy as np
from model import LeNet
import argparse

class client:
    def __init__(self, client_id, num_epoch, lr):
        np.random.seed(99)
        torch.manual_seed(0)
        self.model = LeNet()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr)
        self.epoch = num_epoch
        self.server_address = ('localhost', 8080)
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        with open(os.path.join("DATA_CIFAR10", f"Client{client_id}.pkl"), 'rb') as f:
            self.train_dataset = dill.load(f)
        self.train_dataset = dataloader.DataLoader(self.train_dataset, batch_size=64, shuffle=False)
        self.model_path = f"stage3_model/client{client_id}_model.pth"
    
    def connect_to_server(self):
        print("Connecting to server at", self.server_address)
        self.client_socket.connect(self.server_address)
        print("Connected to server")
        
        try:
            data = b''
            while True:
                packet = self.client_socket.recv(4096)
                if not packet:
                    break
                data += packet
            buffer = io.BytesIO(data)
            buffer.seek(0)
            params = torch.load(buffer)
            self.model.load_state_dict(params)
            torch.save(self.model.state_dict(), self.model_path)
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")

        finally:
            print("Global model loaded directly from server")
    
    def train(self):
        for epoch in range(self.epoch):
            train_loss = 0
            print(f"Epoch {epoch}")
            for j, (inputs, labels) in tqdm(enumerate(self.train_dataset), desc="Training"):
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                train_loss += loss.item()
                loss.backward()
                self.optimizer.step()
            print(f"Train loss: {train_loss/len(self.train_dataset)}")
        torch.save(self.model.state_dict(), self.model_path)
    
    def send_model_to_server(self):
        print("Sending model to server")
        buffer = io.BytesIO()
        torch.save(self.model.state_dict(), buffer)
        buffer.seek(0)

        try:
            while True:
                data = buffer.read(4096)
                if not data:
                    break
                self.client_socket.send(data)
            print("Model sent to server")
            
        finally:
            self.client_socket.shutdown(socket.SHUT_WR)
            print("Finished sending model to server")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_id", type=int, required=True)
    parser.add_argument("--num_epoch", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()
    client_process = client(args.client_id, args.num_epoch, args.lr)
    try:
        client_process.connect_to_server()
        client_process.train()
        client_process.send_model_to_server()  
    finally:
        client_process.client_socket.close()
        print("Connection closed") 