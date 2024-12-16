import  torch
import  torch.nn            as      nn

# Fully-Connected Neural Network
class PINN(nn.Module):
    # Initialization function
    def __init__(self):
        # Initialize parent modules
        super().__init__()
        # Define the forward pass of the network sequentially
        self.network = nn.Sequential(
            nn.LazyLinear(out_features=64),
            nn.ReLU(),
            nn.LazyLinear(out_features=64),
            nn.ReLU(),
            nn.LazyLinear(out_features=64),
            nn.ReLU(),
            nn.LazyLinear(out_features=2)
        )

        # Trainable parameters used in equations (constants approx.)
        self.talpha = nn.Parameter(torch.tensor([1.0], dtype=torch.float32), requires_grad=True)
        self.tbeta = nn.Parameter(torch.tensor([1.0], dtype=torch.float32), requires_grad=True)
        self.tgamma = nn.Parameter(torch.tensor([1.0], dtype=torch.float32), requires_grad=True)
        self.tdelta = nn.Parameter(torch.tensor([1.0], dtype=torch.float32), requires_grad=True)
        self.halpha = nn.Parameter(torch.tensor([1.0], dtype=torch.float32), requires_grad=True)
        self.hbeta = nn.Parameter(torch.tensor([1.0], dtype=torch.float32), requires_grad=True)
        self.hgamma = nn.Parameter(torch.tensor([1.0], dtype=torch.float32), requires_grad=True)

    def forward(self, x):
        # Return model's response to tensor
        return self.network(x)