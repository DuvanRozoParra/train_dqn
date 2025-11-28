import torch.nn as nn
import torch.optim as optim
from hidden.NoisyLinear import NoisyLinear

activaciones = {
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "softmax": nn.Softmax,
    "leaky_relu": nn.LeakyReLU,
    "identity": nn.Identity
}

errores = {
    "mse": nn.MSELoss,
    "cross_entropy": nn.CrossEntropyLoss,
    "bce": nn.BCELoss,
    "huber": nn.HuberLoss
}

optimizadores = {
    "sgd": optim.SGD,
    "adam": optim.Adam,
    "rmsprop": optim.RMSprop
}


class DQN(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_layers,
        output_size: int,
        hidden_activation: str ="relu",
        output_activation: str="identity",
        loss_function: str ="mse",
        optimizer: str ="adam",
        lr: int = 0.001,
        use_minibatch: bool = False,
        batch_size: int = 32,
        dropout: float = 0.0,

        use_noisy: bool = False   
    ):
        super(DQN, self).__init__()

        # Save Settings
        self.use_minibatch = use_minibatch
        self.batch_size = batch_size
        self.use_noisy = use_noisy

        # Neural Network Settings
        layers = []
        prev_size = input_size

        for hidden_size in hidden_layers:
            if use_noisy:
                layers.append(NoisyLinear(prev_size, hidden_size))
            else:
                layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(activaciones[hidden_activation]())
            prev_size = hidden_size
        
        if use_noisy:
            layers.append(NoisyLinear(prev_size, output_size))
        else:
            layers.append(nn.Linear(prev_size, output_size))
        
        if output_activation != "identity":
            if output_activation == "softmax":
                layers.append(activaciones[output_activation](dim=1))  # para softmax, se requiere especificar la dimensión
            else:
                layers.append(activaciones[output_activation]())

        self.net = nn.Sequential(*layers)

        self.criterion = errores[loss_function]()
        self.optimizer = optimizadores[optimizer](self.parameters(), lr=lr)

    def forward(self, x):
        return self.net(x)
    
    def reset_noise(self):
        if self.use_noisy:
            for module in self.modules():
                if isinstance(module, NoisyLinear):
                    module.reset_noise()
    
  