import torch
import torch.nn as nn
import torch.nn.functional as F
from memory_mlp import MemoryMLP
from utils import make_linear

# Neural Memory implementation
class NeuralMemory(nn.Module):
    def __init__(
        self,
        dim, 
        memory_hidden_dim, 
        memory_depth
    ):
        super().__init__()
        self.dim = dim

        self.memory = MemoryMLP(dim, memory_hidden_dim, memory_depth)
        
        self.key_proj = make_linear(dim, dim)
        self.value_proj = make_linear(dim, dim)
        self.query_proj = make_linear(dim, dim)
        self.surprise = {}
        
        self.alpha = 0.001
        self.eta = 0.60
        self.theta = 0.05

    def retrieve(self, x):
        # do normalize or not?
        query = self.query_proj(x)
        return self.memory(query)
    
    def update(self, x):
        # 3.1 Long-term Memory equation(11)
        key = F.normalize(self.key_proj(x), p=2, dim=-1)
        value = self.value_proj(x)
        
        # 3.1 Long-term Memory equation(12)
        residuals = self.memory(key)-value
        loss = residuals.pow(2).mean(dim=-1).sum()

        grads = torch.autograd.grad(loss, self.memory.parameters())
        updated_params = {}

        for (name, param), grad in zip(self.memory.named_parameters(), grads):
            if self.surprise.get(name, None) is None:
                self.surprise[name] = torch.zeros_like(grad)
               
            # 3.1 Long-term Memory equation(14)
            self.surprise[name] = self.surprise[name] * self.eta - self.theta * grad
            # 3.1 Long-term Memory equation(13)
            updated_params[name] = (1-self.alpha) * param.data + self.surprise[name]
            param.data = updated_params[name]

        return loss.item(), updated_params



        


