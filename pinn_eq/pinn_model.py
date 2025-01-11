import  torch
import  torch.nn            as      nn

# Fully-Connected Neural Network
class PINN(nn.Module):
    # Initialization function
    def __init__(self):
        # Initialize parent modules
        super().__init__()
        # Number of hidden layers
        self.hidden_dim = 8
        # Number of neurons per layer
        layer_dim = 256
        # Input Linear + ReLU block
        self.input = nn.Sequential(
            nn.LazyLinear(out_features=layer_dim),
            nn.ReLU()
        )
        # Linear + ReLU activation block
        self.linear = nn.Sequential(
            nn.LazyLinear(out_features=layer_dim),
            nn.ReLU()
        )
        # Output Linear layer
        self.output = nn.LazyLinear(out_features=2)

        # Parameters accompanying T differential equation
        self.alpha_T = nn.Parameter(torch.tensor([1.0], dtype=torch.float32), requires_grad=True)
        self.alpha_T_T = nn.Parameter(torch.tensor([1.0], dtype=torch.float32), requires_grad=True)
        self.alpha_T_H = nn.Parameter(torch.tensor([1.0], dtype=torch.float32), requires_grad=True)
        self.alpha_T_t = nn.Parameter(torch.tensor([1.0], dtype=torch.float32), requires_grad=True)
        self.alpha_T_dH = nn.Parameter(torch.tensor([1.0], dtype=torch.float32), requires_grad=True)
        # Parameters acoompanying H differential equation
        self.alpha_H = nn.Parameter(torch.tensor([1.0], dtype=torch.float32), requires_grad=True)
        self.alpha_H_T = nn.Parameter(torch.tensor([1.0], dtype=torch.float32), requires_grad=True)
        self.alpha_H_H = nn.Parameter(torch.tensor([1.0], dtype=torch.float32), requires_grad=True)
        self.alpha_H_t = nn.Parameter(torch.tensor([1.0], dtype=torch.float32), requires_grad=True)
        self.alpha_H_dT = nn.Parameter(torch.tensor([1.0], dtype=torch.float32), requires_grad=True)

    def forward(self, x):
        # Return model's response to tensor
        x = self.input(x)
        for _ in range(self.hidden_dim):
            x = self.linear(x)
        x = self.output(x)
        return x