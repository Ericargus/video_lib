import torch
import torch.nn as nn
import torch.nn.functional as F

class NextVlad(nn.Module):
    def __init__(self, feature_size=1024, max_frames=128, nextvlad_cluster_size=32, expansion=2, groups=16):
        super().__init__()
        self.nextvlad_cluster_size=nextvlad_cluster_size
        self.groups=groups
        self.feature_size=feature_size
        self.max_frames=max_frames
        self.expansion=expansion
        self.fc1 = nn.Linear(self.feature_size, self.expansion*self.feature_size)
        self.fc2 = nn.Linear(self.expansion*self.feature_size, self.groups)
        #self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(self.nextvlad_cluster_size * self.expansion * self.feature_size // self.groups)
        self.cluster_weights1= nn.Parameter(torch.randn(self.expansion * self.feature_size, self.groups * self.nextvlad_cluster_size))
        self.cluster_weights2= nn.Parameter(torch.randn(self.expansion * self.feature_size // self.groups, self.nextvlad_cluster_size))
        
        
    def forward(self, x):
        x = self.fc1(x)                                   
        attention = torch.sigmoid(self.fc2(x))                    
        attention = torch.reshape(attention,(-1, self.max_frames*self.groups, 1)) 
        feature_size = self.expansion * self.feature_size // self.groups         
        activation = torch.matmul(x, self.cluster_weights1)                        
        activation = torch.reshape(activation, (-1, self.groups*self.max_frames, self.nextvlad_cluster_size)) 
        #activation = torch.reshape(activation, (-1, self.max_frames * self.groups, self.nextvlad_cluster_size)) 
        #print(activation.shape)
        activation = torch.softmax(activation, dim=-1)                            
        activation = torch.multiply(activation, attention)                      
        
        a_sum = torch.sum(activation, dim=-2, keepdim=True)                            
        a = torch.multiply(a_sum, self.cluster_weights2)             
        a = a.permute((0, 2, 1))                                     
        activation = activation.permute((0, 2, 1))                    
        
        reshaped_x = torch.reshape(x, (-1, self.max_frames * self.groups, feature_size)) 
        vlad = torch.matmul(activation, reshaped_x)                                                                  
        vlad = vlad-a
        vlad = F.normalize(vlad,p=2, dim=1)
        
        vlad = torch.reshape(vlad, (-1, self.nextvlad_cluster_size * feature_size))
        return vlad
