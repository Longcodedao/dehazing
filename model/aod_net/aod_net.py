import torch
import torch.nn as nn

class AODNet(nn.Module):
    def __init__(self):
        super(AODNet, self).__init__()

        # Encoder Layers
        self.conv1 = nn.Conv2d(3, 3, kernel_size = 1)
        self.conv2 = nn.Conv2d(3, 3, kernel_size = 3, padding = 1)
        self.conv3 = nn.Conv2d(6, 3, kernel_size = 5, padding = 2)
        self.conv4 = nn.Conv2d(6, 3, kernel_size = 7, padding = 3)
        self.conv5 = nn.Conv2d(12, 3, kernel_size = 3, padding = 1)

        # ReLU activation
        self.relu = nn.ReLU(inplace = True)

        # Parameter b (bias term)
        self.b = 1.0

    def forward(self, x):
        # x is the input hazy image 
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))

        # Concatenate features 
        cat1 = torch.cat((x1, x2), dim = 1)
        x3 = self.relu(self.conv3(cat1))

        cat2 = torch.cat((x2, x3), dim = 1)
        x4 = self.relu(self.conv4(cat2))

        cat3 = torch.cat((x1, x2, x3, x4), dim = 1)
        K = self.relu(self.conv5(cat3))

        # Apply the refrmulated atmospheric scattering model
        # J(x) = K(x) * I(x) - K(x) + b
        output = K * x - K + self.b

        return output