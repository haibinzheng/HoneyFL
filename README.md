# HoneyFL: Using Honeypots to Catch Backdoors in Federated Learning

In this paper, we propose a novel run-time defense named HoneyFL to secure FL against backdoor attacks. We intentionally construct Honey clients to insert HoneyDoor into global model, and observe the predicted classes for inputs with/without HoneyDoor. By observing the accordance with the the predicted classes and HoneyMap, we can distinguish trigger samples from benign samples. 

## Requirements
torch==0.4.1
torchvision==0.2.1
