import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

class PruneLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(PruneLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)
        self.mask = torch.ones(self.linear.weight.data.shape).to(device).float()
        m = self.in_features
        n = self.out_features
        self.sparsity = 1.0
        # Initailization
        self.linear.weight.data.normal_(0, math.sqrt(2. / (m+n)))

    def forward(self, x):
#         if self.codebook is not None:
#             old_values = self.linear.weight.data
#             new_values = self.codebook.data.cpu().numpy()[self.linear.weight.data.to('cpu').numpy().flatten().round().reshape(-1,1).astype('int')]
            
#             new_values = new_values.reshape(old_values.shape)
#             self.linear.weight.data = torch.from_numpy(new_values).to(device)*self.mask
#             out = self.linear(x)
#             self.linear.weight.data = old_values.to(device)
#         else:
        self.linear.weight.data = self.linear.weight.data*self.mask
        out = self.linear(x)

        #self.linear.weight.grad = self.linear.weight.grad*self.mask
        return out
    

        pass

    def prune_by_percentage(self, q=5.0):
        """
        Pruning the weight paramters by threshold.
        :param q: pruning percentile. 'q' percent of the least 
        significant weight parameters will be pruned.
        """
        """
        Prune the weight connections by percentage. Calculate the sparisty after 
        pruning and store it into 'self.sparsity'.
        Store the pruning pattern in 'self.mask' for further fine-tuning process 
        with pruned connections.
        --------------Your Code---------------------
        """
        q/=100
        weights = self.linear.weight.data
        count = weights.numel()
        indices = weights.flatten().abs().topk(count)[1].float()/count
        self.mask = (indices>q).reshape(weights.shape).float()

        mask_filter = self.mask
        
        self.sparsity = 1-mask_filter.sum()/mask_filter.numel()
        self.linear.weight.data = self.linear.weight.data*mask_filter


    def prune_by_std(self, s=0.25):
        """
        Pruning by a factor of the standard deviation value.
        :param std: (scalar) factor of the standard deviation value. 
        Weight magnitude below np.std(weight)*std
        will be pruned.
        """

        """
        Prune the weight connections by standarad deviation. 
        Calculate the sparisty after pruning and store it into 'self.sparsity'.
        Store the pruning pattern in 'self.mask' for further fine-tuning process 
        with pruned connections.
        --------------Your Code---------------------
        """
        
        weights = self.linear.weight
        
        self.mask = (torch.abs((weights-weights.mean())/weights.std())>s).float()
        mask_filter = self.mask
       
        self.sparsity = 1-mask_filter.sum()/mask_filter.numel()
        self.linear.weight.data = self.linear.weight.data*mask_filter

        
        
class PrunedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(PrunedConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        
        # Expand and Transpose to match the dimension
        # self.mask = np.ones_like([out_channels, in_channels, kernel_size, kernel_size])
        self.mask = torch.ones(self.conv.weight.data.shape).to(device)
        #self.mask = torch.ones_like(self.conv.weight.data).to(device)

        # Initialization
        n = self.kernel_size * self.kernel_size * self.out_channels
        m = self.kernel_size * self.kernel_size * self.in_channels
        self.conv.weight.data.normal_(0, math.sqrt(2. / (n+m) ))
        self.sparsity = 1.0
        
        self.codebook = None


    def forward(self, x):   
#         if self.codebook is not None:
#             old_values = self.conv.weight.data
#             new_values = self.codebook.data.cpu().numpy()[self.conv.weight.data.to('cpu').numpy().flatten().round().reshape(-1,1).astype('int')]
#             new_values = new_values.reshape(old_values.shape)
#             self.conv.weight.data = torch.from_numpy(new_values).to(device)*self.mask
#             out = self.conv(x)
#             self.conv.weight.data = old_values.to(device)
#         else:
        self.conv.weight.data = self.conv.weight.data*self.mask
        out = self.conv(x)
        return out

    def prune_by_percentage(self, q=5.0):
        """
        Pruning by a factor of the standard deviation value.
        :param s: (scalar) factor of the standard deviation value. 
        Weight magnitude below np.std(weight)*std
        will be pruned.
        """
        
        """
        Prune the weight connections by percentage. Calculate the sparisty after 
        pruning and store it into 'self.sparsity'.
        Store the pruning pattern in 'self.mask' for further fine-tuning process 
        with pruned connections.
        --------------Your Code---------------------
        """
        q/=100
        weights = self.conv.weight.data
        count = weights.numel()
        indices = weights.flatten().abs().topk(count)[1].float()/count
        #print(indices>q)
        self.mask = (indices>q).reshape(weights.shape).float()

        mask_filter = self.mask
        
        self.sparsity = 1-mask_filter.sum()/mask_filter.numel()
        self.conv.weight.data = self.conv.weight.data*mask_filter

    def prune_by_std(self, s=0.25):
        """
        Pruning by a factor of the standard deviation value.
        :param s: (scalar) factor of the standard deviation value. 
        Weight magnitude below np.std(weight)*std
        will be pruned.
        """
        
        """
        Prune the weight connections by standarad deviation. 
        Calculate the sparisty after pruning and store it into 'self.sparsity'.
        Store the pruning pattern in 'self.mask' for further fine-tuning process 
        with pruned connections.
        --------------Your Code---------------------
        """
     
        weights = self.conv.weight
        
        self.mask = (torch.abs((weights-weights.mean())/weights.std())>s).float()
        mask_filter = self.mask
        
        self.sparsity = 1-mask_filter.sum()/mask_filter.numel()
        self.conv.weight.data = self.conv.weight.data*mask_filter

#         self.mask = (torch.abs((weights-weights.mean())/weights.std())<s).float()
#         self.sparsity = self.mask.sum()/self.mask.numel()
#         self.linear.weight.data = self.linear.weight.data*self.mask

