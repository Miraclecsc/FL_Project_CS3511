import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import dataloader
import os
import io
import threading
import dill
import socket
import numpy as np
from model import LeNet

class Server:
    def __init__(self, address, port):
        np.random.seed(99)
        torch.manual_seed(0)
        self.model = LeNet()
        self.new_model = LeNet()
        self.new_model_list = []
        self.criterion = nn.CrossEntropyLoss()
        self.model_path = "stage3_model/server_model.pth"
        torch.save(self.model.state_dict(), self.model_path)
        self.model.load_state_dict(torch.load(self.model_path))
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((address, port))
        self.server_socket.listen()

        print(f"Server listening on {address}:{port}")
        with open(os.path.join("DATA_CIFAR10", "Test.pkl"), 'rb') as f:
            self.test_dataset = dill.load(f)
        self.test_dataset = dataloader.DataLoader(self.test_dataset, batch_size=64, shuffle=False)

    def send_model(self, client_socket):
        print("Sending global model to client...")
        buffer = io.BytesIO()
        torch.save(self.model.state_dict(), buffer)
        buffer.seek(0)
        data = buffer.read()
        client_socket.sendall(data)
        client_socket.shutdown(socket.SHUT_WR)
        print("Global model sent to client.")

    def receive_model(self, client_socket):
        print("Receiving model from client...")
        buffer = bytearray()
        try:
            while True:
                data = client_socket.recv(4096)
                if not data:
                    break
                buffer.extend(data)
            model_state_dict = torch.load(io.BytesIO(buffer))
            self.new_model.load_state_dict(model_state_dict)
            self.new_model_list.append(self.new_model)
            
        finally:
            print("Model received and loaded from client.")
    
    def handle_client(self, client_socket):
        try:
            self.send_model(client_socket)
            self.receive_model(client_socket)
        finally:
            x = input("press 'Y' or 'y' to merge and test the model, others to continue training...\n")
            if x == 'Y' or x == 'y':
                server_process.test()
            else:
                y = input("press 'C' or 'c' to clear the cached models list, others to remain status...\n")
                if y == 'C' or y == 'c':
                    server_process.new_model_list = []  
    
    def test(self):
        avg_model = LeNet()
        print("The number of models received from clients: ", len(self.new_model_list)) 
        for avg_param in avg_model.parameters():
            avg_param.data.zero_() 
        for model in self.new_model_list:
            for avg_param, param in zip(avg_model.parameters(), model.parameters()):
                avg_param.data += param.data
                    
        for avg_param in avg_model.parameters():
            avg_param.data /= len(self.new_model_list)
        
        self.model.eval()
        avg_model.eval()
        correct = 0
        new_correct = 0
        total = 0
        with torch.no_grad():
            for data in self.test_dataset:
                images, labels = data
                outputs = self.model(images)
                new_outputs = avg_model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                _, new_predicted = torch.max(new_outputs.data, 1)
                new_correct += (new_predicted == labels).sum().item()
        print(f"Accuracy of the original model on testing images: {100 * correct / total}%")
        print(f"Accuracy of the new model on testing images: {100 * new_correct / total}%")
        if new_correct > correct:
            torch.save(avg_model.state_dict(), self.model_path)
            self.model.load_state_dict(torch.load(self.model_path))
            self.new_model_list = []
            print("New model is better, saved.")  

def client_thread(connection, address):
    print(f"Connected by {address}")
    server_process.handle_client(connection)

if __name__ == "__main__":
    server_process = Server('localhost', 8080)
    while True:
        print("Waiting for client connection...")
        client_conn, client_addr = server_process.server_socket.accept()
        thread = threading.Thread(target=client_thread, args=(client_conn, client_addr))
        thread.start()
        
            

        