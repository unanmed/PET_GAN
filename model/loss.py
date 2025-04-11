import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim

class WGANLoss:
    def __init__(self, lambda_gp=10):
        self.lambda_gp = lambda_gp
        
    def compute_gradient_penalty(self, critic, origin, real, fake):
        epsilon = torch.rand(real.size(0), 1, 1, 1, device=real.device)
        interp = epsilon * real + (1 - epsilon) * fake
        
        interp.requires_grad_()
        
        interp_scores = critic(interp, origin)
        
        grad = torch.autograd.grad(
            outputs=interp_scores, inputs=interp,
            grad_outputs=torch.ones_like(interp_scores),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        
        grad_norm = grad.view(real.size(0), -1).norm(2, dim=1)
        gp_loss = ((grad_norm - 1.0) ** 2).mean()
        
        return gp_loss
        
    def loss_critic(self, critic, origin, real, fake):
        real_scores = critic(real, origin)
        fake_scores = critic(fake, origin)
        
        d_loss = torch.mean(fake_scores) - torch.mean(real_scores)
        gp_loss = self.compute_gradient_penalty(critic, origin, real, fake)
        
        return d_loss, d_loss + gp_loss * self.lambda_gp
        
    def loss_generator(self, critic, origin, fake):
        fake_scores = critic(fake, origin)
        return -torch.mean(fake_scores)
    
def validate_loss(fake, target):
    ssim_value, _ = ssim(fake, target, win_size=11, full=True, data_range=1)
    mse = np.mean((fake.astype(np.float64) - target.astype(np.float64)) ** 2)
    nmse = mse / np.mean(fake.astype(np.float64) ** 2) if np.mean(fake.astype(np.float64) ** 2) != 0 else float('inf')
    me = np.mean(fake.astype(np.float64) - target.astype(np.float64))
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(fake.astype(np.float64) - target.astype(np.float64)))
    
    return ssim_value, mse, nmse, me, rmse, mae
    