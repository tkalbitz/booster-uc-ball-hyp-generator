import torch
import torch.nn as nn
import torch.nn.functional as F


def _activation(act: str) -> nn.Module:
    if act == 'relu':
        return nn.ReLU()
    elif act == 'leakyrelu':
        return nn.LeakyReLU()
    elif act == 'relu6':
        return nn.ReLU6()
    elif act == 'elu':
        return nn.ELU()
    elif act == 'tanh':
        return nn.Tanh()
    elif act == 'linear':
        return nn.Identity()

    raise NotImplementedError()

class Conv2dBn(nn.Module):
    def __init__(self, in_channels: int, nb_filter: int, kernel_size: int, stride: int = 1, padding: int = 0, activation: str = 'relu') -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, nb_filter, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(nb_filter, affine=False)
        self.activation = _activation(activation)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class BigLittleReduction(nn.Module):
    def __init__(self, in_channels: int, filter_count: int, activation: str = 'relu') -> None:
        super().__init__()
        self.conv1_3x3 = Conv2dBn(in_channels, filter_count//2, kernel_size=3, stride=2, padding=1, activation=activation)
        self.conv2_3x3 = Conv2dBn(in_channels, filter_count, kernel_size=3, stride=1, padding=1, activation=activation)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv1_out = self.conv1_3x3(x)
        conv2_out = self.conv2_3x3(x)
        conv2_out = self.avg_pool(conv2_out)
        return torch.cat([conv1_out, conv2_out], dim=1)


class SEBlock(nn.Module):
    def __init__(self, input_channels: int, r: int = 4) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(input_channels, input_channels // r)
        self.fc2 = nn.Linear(input_channels // r, input_channels)
    
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        x = self.pool(input_tensor)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        x = x.view(x.size(0), x.size(1), 1, 1)
        return input_tensor * x


class NetworkV2(nn.Module):
    def __init__(self, input_height: int, input_width: int) -> None:
        super().__init__()
        self.input_bn = nn.BatchNorm2d(3)
        self.conv_bn1 = Conv2dBn(3, 8, kernel_size=3, stride=1, padding=1, activation='relu')
        
        self.big_little1 = BigLittleReduction(8, 10, 'relu')
        self.dropout1 = nn.Dropout(0.2)
        
        self.big_little2 = BigLittleReduction(15, 12, 'relu')
        self.se_block1 = SEBlock(18)
        self.dropout2 = nn.Dropout(0.1)
        
        self.big_little3 = BigLittleReduction(18, 12, 'relu')
        self.se_block2 = SEBlock(18)
        self.dropout3 = nn.Dropout(0.1)
        
        self.big_little4 = BigLittleReduction(18, 14, 'relu')
        self.se_block3 = SEBlock(21)
        self.dropout4 = nn.Dropout(0.1)
        
        # Calculate the size after all reductions
        # Each big_little_reduction halves the spatial dimensions
        final_h = input_height // (2 ** 4)  # 4 reductions
        final_w = input_width // (2 ** 4)
        final_channels = 21
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(final_channels * final_h * final_w, 32)
        self.fc1_bn = nn.BatchNorm1d(32)
        self.fc1_activation = _activation('relu')
        
        self.fc2 = nn.Linear(32, 2)
        self.output_activation = _activation('tanh')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_bn(x)
        x = self.conv_bn1(x)
        
        x = self.big_little1(x)
        x = self.dropout1(x)
        
        x = self.big_little2(x)
        x = self.se_block1(x)
        x = self.dropout2(x)
        
        x = self.big_little3(x)
        x = self.se_block2(x)
        x = self.dropout3(x)
        
        x = self.big_little4(x)
        x = self.se_block3(x)
        x = self.dropout4(x)
        
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc1_bn(x)
        x = self.fc1_activation(x)
        
        x = self.fc2(x)
        x = self.output_activation(x)
        
        return x


def create_network_v2(input_height: int, input_width: int) -> NetworkV2:
    return NetworkV2(input_height, input_width)
