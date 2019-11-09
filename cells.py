import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

# this is the class for depthwise separable convolution operation, which comprises of two convolutions depthwise and pointwise
# as mentioned in the DARTs paper separable convolution is applied twice
class Separable(nn.Module):
    
    def __init__(self, in_ch, out_ch, kernel = 3):
        super(Separable, self).__init__()
        layers = []
        
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(in_ch, in_ch, kernel_size = kernel, groups = in_ch, padding = 1))
        layers.append(nn.Conv2d(in_ch, out_ch, kernel_size = 1))
        layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(in_ch, in_ch, kernel_size = kernel, groups = in_ch, padding = 1))
        layers.append(nn.Conv2d(in_ch, out_ch, kernel_size = 1))
        layers.append(nn.BatchNorm2d(out_ch))
        
        self.layer = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.layer(x)
    

# this is the implementation of the dilation convolution but in depthwise separable way
class dilation_separable(nn.Module):
    
    def __init__(self, in_ch, out_ch, kernel = 3):
        super(dilation_separable,self).__init__()
        layers = []
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(in_ch, in_ch, kernel_size = kernel, dilation = 2, groups = in_ch, padding = 2))
        layers.append(nn.Conv2d(in_ch, out_ch, kernel_size = 1,padding = 0))
        layers.append(nn.BatchNorm2d(out_ch))
        
        self.layer = nn.Sequential(*layers)
    
    def forward(self,x):
        return self.layer(x)
    
    
# this class is implemented to make sure the size of feature maps in each cell is compatible with each other and can be processed further
class Compatible(nn.Module):
    
    def __init__(self, in_ch, out_ch, kernel, stride):
        
        super(Compatible, self).__init__()
        
        self.layer = nn.Sequential(nn.ReLU(),
                                  nn.Conv2d(in_ch, out_ch, kernel_size = kernel, stride = stride, padding = 0),
                                  nn.BatchNorm2d(out_ch))
        
    def forward(self, x):
        return self.layer(x)
    
    
# Normal cell implementation which, as per paper, does not change the feature maps size.
class Normal_cell(nn.Module):
    
    def __init__(self, kminus1_ch, kminus2_ch, prev_reduction = False):
        super(Normal_cell, self).__init__()
        
        self.prev_reduction = prev_reduction
        
        # if previous cell was reduction cell, then perform Comptaible operation on k-2 input
        if prev_reduction:
            #reduce spatial size of the k_minus_2 feature map and double its channels
            self.reduce_spatial = Compatible(kminus2_ch, kminus2_ch  * 2, 1, 2)
            kminus2_ch = kminus2_ch * 2
        
        
        total_ch = kminus1_ch
        self.maintain_ch = Compatible(total_ch * 4, total_ch, 1,1)
        
        # all Separable convolution 3x3
        self.kminus1to0 = Separable(kminus1_ch, kminus1_ch)
        self.kminus1to1 = Separable(kminus1_ch, kminus1_ch)
        self.kminus1to2 = Separable(kminus1_ch, kminus1_ch)
        
        # all Separable convolution 3x3
        self.kminus2to0 = Separable(kminus2_ch, kminus2_ch)
        self.kminus2to1 = Separable(kminus2_ch, kminus2_ch)
        
        # dilated convolution 3x3, this will have receptive field of 5x5
        self.zeroto3 = dilation_separable(kminus1_ch, kminus1_ch)
        
    def forward(self, k_minus_1, k_minus_2):
        #execute forward calls on input just as structure of normal cell
        
        if self.prev_reduction:
            k_minus_2 = self.reduce_spatial(k_minus_2)
        
        # make sure channels are same in both inputs
        assert k_minus_1.size(1) == k_minus_2.size(1)
        # perform operations for each node
        node0 = torch.add(self.kminus1to0(k_minus_1), self.kminus2to0(k_minus_2))
        node1 = torch.add(self.kminus1to1(k_minus_1), self.kminus2to1(k_minus_2))
        node2 = torch.add(k_minus_2, self.kminus1to2(k_minus_1))
        node3 = torch.add(k_minus_2, self.zeroto3(node0))
        
        # perform depthwise concatenation 
        out = torch.cat((node0, node1, node2, node3), dim = 1)
        # make sure number of channels are maintained by factor of 2 after concatenation operation
        out = self.maintain_ch(out)
        return out
    
    
class Reduction_cell(nn.Module):
    
    def __init__(self, in_ch, out_ch):
        super(Reduction_cell,self).__init__()
        
        total_ch = out_ch
        self.double_ch = Compatible(in_ch, total_ch, 1, 1)
        self.pool = nn.MaxPool2d(kernel_size= 3, stride = 2, padding= 1)
        self.maintain_ch = Compatible(total_ch * 4, total_ch, 1,1)
        
    def forward(self, k_minus_1, k_minus_2):
        
        # check if channels and spatial size of both inputs are same
        assert k_minus_1.size(1) == k_minus_2.size(1)
        assert k_minus_1.size(2) == k_minus_2.size(2)
        
        # double the number of channels
        k_minus_1 = self.double_ch(k_minus_1)
        k_minus_2 = self.double_ch(k_minus_2)
        
        # use max pooling operation to reduce the spatial size by half
        k1pool = self.pool(k_minus_1)
        k2pool = self.pool(k_minus_2)
        
        # addition operation to create nodes
        node0 = torch.add(k1pool, k2pool)
        node1 = torch.add(node0, k1pool)
        node2 = torch.add(node0, k2pool)
        node3 = torch.add(node0, k1pool)
        
        # perform depthwise concatenation 
        out = torch.cat((node0, node1, node2, node3),  dim = 1)
        # make sure number of channels are maintained by factor of 2 after concatenation operation
        out = self.maintain_ch(out)
        
        return out
    
    
    
    