## 项目简介
本项目展示了一个模拟真实情况下的分布式联邦学习框架，实现了基于联邦学习框架的CIFAR10分类任务。

## 代码结构
```
.
├── stage1.py
├── stage2.py
├── stage3.py
├── client.py
├── stage1_model
    ├──server_model.pth
    ├──client1_model.pth
    ├──client2_model.pth
    ├──...
├── stage2_model
    ├──server_model.pth
    ├──client1_model.pth
    ├──client2_model.pth
    ├──...
├── stage3_model
    ├──server_model.pth
    ├──client1_model.pth
    ├──client2_model.pth
    ├──...
├── Data_CIFAR10
    ├──Client1.pkl
    ├──...
    ├──Client20.pkl
    ├──Test.pkl
├── model.py
└── readme.md
```
服务器模型server_model,以及客户端模型client{i}_model(i=1,2,...,20)会被初始化并保存在stage{n}_model(n=1,2,3)文件夹中。

## 代码运行
1. 运行第一阶段代码
```bash
python stage1.py
```

2. 运行第二阶段代码
```bash
python stage2.py --num_clients=10
```

3. 运行第三阶段代码

- 首先在终端中启动服务端
```bash
python stage3.py
```
- 然后在新建终端中启动客户端（支持多个新建终端，并行启动客户端）
```bash
python client.py --client_id=1 --num_epoch=50 --lr=0.01
```
```bash
python client.py --client_id=4 --num_epoch=80 --lr=0.01
```
```bash
python client.py --client_id=10 --num_epoch=10 --lr=0.1
```
- 其中client_id是客户端的id，epochs是训练的轮数，lr是学习率。新建的客户端首先与服务器完成连接，并立即接受当前服务器端的模型。随后客户端使用本地数据对接收到的模型进行训练。完成训练后，客户端会将更新后的模型发送给服务端，并断开连接。
- 服务端（支持同时与多个客户端并行交互）在与客户端连接后会打印消息提示，并将当前服务器模型传递给客户端。此后服务器等待客户端的提交，收到客户端提交后会在终端中打印消息，并等待接下来的指令键入（若未提示输入指令，尝试键入回车）。根据终端指令，服务器可以选择将当前已提交客户端更新的模型进行聚合，得到最终的模型；或者不更新模型参数，并选择是否清空已提交的模型列表。

## 代码细节
1. 第一阶段设定一个服务器与N个客户端共同训练一个分类模型。所有客户端在本地训练模型后，通过“.pth”文件格式交换模型参数，以实现模型的更新和同步。

2. 第二阶段引入部分参与机制，即从N个客户端中随机选取M个客户端参与每一轮的全局模型更新。这样可以在保证模型性能的同时，降低通信成本和提高系统效率。

3. 第三阶段在模型交互方式上进行改进，不再通过文件读写的方式进行模型参数的交换，而是通过套接字通信。服务器和客户端之间建立稳定的通信连接，通过Internet域套接字实现数据的即时传输和处理，提高了系统的响应速度和实时性。

