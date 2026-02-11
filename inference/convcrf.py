import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ConvCRF(nn.Module):
    """
    Minimal implementation of Convolutional CRF in PyTorch.
    Based on the idea of interacting with local neighbors defined by a kernel size.
    Features:
    - Gaussian (Spatial) smoothing
    - Bilateral (Appearance) smoothing using local neighborhood color difference
    """
    def __init__(self, num_classes, filter_size=7, n_iters=5, 
                 sxy_gaussian=3, sxy_bilateral=50, srgb_bilateral=10, compat_gaussian=3, compat_bilateral=10):
        super(ConvCRF, self).__init__()
        self.num_classes = num_classes
        self.filter_size = filter_size
        self.n_iters = n_iters
        self.sxy_g = sxy_gaussian
        self.sxy_b = sxy_bilateral
        self.srgb = srgb_bilateral
        self.compat_g = compat_gaussian
        self.compat_b = compat_bilateral
        
        # Gaussian Kernel (Spatial)
        self.gaussian_kernel = self._create_gaussian_kernel(filter_size, sxy_gaussian)
        
    def _create_gaussian_kernel(self, size, sigma):
        # Create 1D Gaussian
        coords = torch.arange(size).float() - (size - 1) / 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum() # Normalize
        # Create 2D Gaussian
        g2d = g.unsqueeze(0) * g.unsqueeze(1)
        return g2d

    def forward(self, unary, img):
        """
        unary: (B, C, H, W) - Negative Log Probabilities (or logits)
        img: (B, 3, H, W) - RGB Image [0, 255]
        """
        # Ensure img is float
        img = img.float()
        
        # Initialize Q (Probability) = Softmax(-Unary)
        # Note: If unary is already -log(prob), then prob = exp(-unary)
        # Usually unary inputs are logits or -log(prob).
        # Let's assume input is Logits or LogProbs.
        # If input is Probabilities, we convert to Unary = -log(P).
        
        # Here we assume input `unary` is NOT probabilities, but the Unary Energy.
        # Energy E = -log(P). P = exp(-E).
        # We start with Q = exp(-unary) / Z
        
        q = F.softmax(-unary, dim=1)
        
        # Define kernels for this batch
        # Spatial Kernel is constant (gaussian)
        # Bilateral Kernel depends on Image.
        
        # We use Unfold (im2col) to extract local neighborhoods for message passing
        # This allows determining color difference between center pixel and neighbors
        
        batch, c, h, w = unary.shape
        pad = self.filter_size // 2
        
        # Pre-compute Spatial Gaussian Weight for the kernel
        spatial_weight = self.gaussian_kernel.to(unary.device) # (K, K)
        spatial_weight = spatial_weight.view(1, 1, -1, 1) # (1, 1, K*K, 1)
        
        # Unfold Image to patches: (B, 3, H, W) -> (B, 3*K*K, H*W)
        img_unfold = F.unfold(img, kernel_size=self.filter_size, padding=pad, stride=1)
        img_unfold = img_unfold.view(batch, 3, self.filter_size**2, h, w)
        
        # Center pixel color: (B, 3, 1, H, W)
        img_center = img.unsqueeze(2) 
        
        # Compute Color Difference Squared: ||I_i - I_j||^2
        diff = img_unfold - img_center # (B, 3, K*K, H, W)
        diff_sq = (diff ** 2).sum(dim=1, keepdim=True) # (B, 1, K*K, H, W)
        
        # Bilateral Weight: exp( - ||I_i - I_j||^2 / 2*srgb^2 ) * SpatialGaussian
        # Spatial Gaussian term matches indices of K*K
        bilateral_weight = torch.exp(-diff_sq / (2 * self.srgb ** 2)) # Color part
        bilateral_weight = bilateral_weight * spatial_weight.unsqueeze(-1) # Multiply spatial
        
        for _ in range(self.n_iters):
            # Message Passing
            
            # 1. Expand Q to neighbors
            # (B, C, H, W) -> (B, C*K*K, H*W) via Unfold
            q_unfold = F.unfold(q, kernel_size=self.filter_size, padding=pad, stride=1)
            q_unfold = q_unfold.view(batch, c, self.filter_size**2, h, w)
            
            # 2. Appearance Kernel (Bilateral) Message
            # Message = sum_j ( Q_j * W_ij )
            # Q_j: q_unfold
            # W_ij: bilateral_weight
            msg_b = (q_unfold * bilateral_weight).sum(dim=2) # Sum over K*K neighbors -> (B, C, H, W)
            
            # 3. Spatial Kernel Message
            # Simple Gaussian Blur on Q
            # We can use q_unfold with just spatial_weight, or standard Conv2d
            # Let's use standard Conv2d with fixed Gaussian for speed/simplicity
            # (Assuming standard Gaussian pairwise)
            weight_g = spatial_weight.view(1, 1, self.filter_size, self.filter_size).repeat(c, 1, 1, 1)
            msg_g = F.conv2d(q, weight_g.to(q.device), padding=pad, groups=c)
            
            # 4. Compatiblity Transform (Weighting)
            # Usually strict Potts model: mu(L_i, L_j) = w if L_i != L_j else 0
            # For dense CRF, it's: E = Unary + w_g * Gaussian + w_b * Bilateral
            # The update is: Q_tilde = exp( -Unary - w_g*Msg_g - w_b*Msg_b )
             
            # In Mean Field:
            # Q = 1/Z * exp( -Unary - W*Q )
            # Implementation:
            # logits = -unary - (w_g * (msg_g - q) + w_b * (msg_b - q))
            # Note: The raw message includes self-contribution (center pixel).
            # Usually we subtract Q (center) if the kernel has 1 at center.
            # But let's follow standard update: energy -= pairwise
            
            pairwise = self.compat_g * msg_g + self.compat_b * msg_b
            
            # Invert compat for 'energy' subtraction? 
            # Mean field update: Q_i(l) = 1/Z * exp( -psi_u(l) - sum_k w_k * sum_j k(f_i, f_j) Q_j(l) )
            # So we SUBTRACT pairwise energy.
            
            energy = unary - pairwise # Assuming unary is NEGATIVE Log Prob ??
            # Wait, earlier I defined q = softmax(-unary). So unary is Energy (positive).
            # Total Energy E = Unary + Pairwise
            # Q = exp(-E)
            
            energy = unary + pairwise # Maybe?
            
            # Refinement - Potts Model usually penalizes DIFFERENT labels.
            # If Q is high for class C at neighbor, and weights are high (similar color),
            # then Message is high for class C.
            # This should ENCOURAGE class C at center.
            # So Energy should be LOWER.
            # So we want exp( something positive ).
            # If msg is positive "support" for class C, we add it to -E.
            # q_new = softmax( -unary + pairwise ) ?
            
            q = F.softmax(-unary + pairwise, dim=1)
            
        return q

