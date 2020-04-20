import numpy as np
from sklearn.cluster import KMeans
from pruned_layers import *
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def quantize_whole_model(net, bits=8):
    """
    Quantize the whole model.
    :param net: (object) network model.
    :return: centroids of each weight layer, used in the quantization codebook.
    """
    
    num_clusters = 2**bits
    cluster_centers = []
    assert isinstance(net, nn.Module)
    layer_ind = 0
    for n, m in net.named_modules():  
        if isinstance(m, PrunedConv):       
            """
            Apply quantization for the PrunedConv layer.
            """
            
            weights = m.conv.weight.data.to('cpu').numpy().reshape(-1,1)
            init_weights = np.linspace(weights.min(),weights.max(),num_clusters).reshape(-1,1)
            model = KMeans(n_clusters=num_clusters, init=init_weights)
        
            with torch.no_grad():
                model.fit(weights[weights!=0].reshape(-1,1))
                indices = model.predict(weights).reshape(m.conv.weight.data.shape)
#                 if save_book:
#                     m.conv.weight.data = torch.tensor(indices).float()
#                     m.codebook = torch.nn.Parameter(torch.from_numpy(model.cluster_centers_))
                new_weights = model.cluster_centers_[indices].squeeze()
                new_data = torch.tensor(new_weights).float().to(device)*m.mask
                m.conv.weight.data = new_data
                    
            cluster_centers.append(model.cluster_centers_)
            layer_ind += 1
            print("Complete %d layers quantization..." %layer_ind)
       
        elif isinstance(m, PruneLinear):
            """
            Apply quantization for the PrunedLinear layer.
            """
            
            weights = m.linear.weight.data.to('cpu').numpy().reshape(-1,1)
            init_weights = np.linspace(weights.min(),weights.max(),num_clusters).reshape(-1,1)

            model = KMeans(n_clusters=num_clusters, init=init_weights)

            with torch.no_grad():
                model.fit(weights[weights!=0].reshape(-1,1))
                indices = model.predict(weights).reshape(m.linear.weight.data.shape)
#                 if save_book:
#                     m.linear.weight.data = torch.tensor(indices).float()
#                     m.codebook = torch.nn.Parameter(torch.from_numpy(model.cluster_centers_))
                new_weights = model.cluster_centers_[indices].squeeze()
                new_data = torch.tensor(new_weights).float().to(device)*m.mask
                m.linear.weight.data = new_data
            cluster_centers.append(model.cluster_centers_)
            layer_ind += 1
            print("Complete %d layers quantization..." %layer_ind)
    return np.array(cluster_centers)

