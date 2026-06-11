import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod


class Transform(ABC):
    @abstractmethod
    def forward(self, x_constrained):
        """Map from constrained → unconstrained space. Element-wise."""
        pass
    
    @abstractmethod
    def inverse(self, x_unconstrained):
        """Map from unconstrained → constrained space. Element-wise."""
        pass

    @abstractmethod
    def derivative(self, x_unconstrained):
        """
        Diagonal of the Jacobian dx_constrained/dx_unconstrained.
        Returns a tensor of the same shape as x_unconstrained.
        For element-wise transforms this is simply the element-wise derivative.
        """
        pass

    @abstractmethod
    def log_abs_det_jacobian(self, x_unconstrained):
        """
        Log absolute determinant of the Jacobian.
        Returns a scalar (summed over all elements).
        """
        pass

    @abstractmethod
    def grad_log_abs_det_jacobian(self, x_unconstrained):
        """
        Gradient of log_abs_det_jacobian w.r.t. x_unconstrained.
        Returns a tensor of the same shape as x_unconstrained.
        """
        pass


class LogTransform(Transform):
    def __init__(self, border=0, side="larger"):
        self.border = border
        self.side = side

    def forward(self, x_constrained):
        if self.side == "larger":
            x = (x_constrained - self.border).clamp(min=1e-8)
            return torch.log(x)
        else:
            x = (self.border - x_constrained).clamp(min=1e-8)
            return torch.log(x)

    def inverse(self, x_unconstrained):
        sign = 1 if self.side == "larger" else -1
        return self.border + sign * torch.exp(x_unconstrained)

    def derivative(self, x_unconstrained):
        sign = 1 if self.side == "larger" else -1
        return sign * torch.exp(x_unconstrained)

    def log_abs_det_jacobian(self, x_unconstrained):
        return x_unconstrained.sum()

    def grad_log_abs_det_jacobian(self, x_unconstrained):
        return torch.ones_like(x_unconstrained)


class InverseSoftPlusTransform(Transform):  
    def __init__(self, border=0, side="larger"):
        super().__init__()
        self.border = border
        self.side = side

    def forward(self, x_constrained):
        if self.side == "larger":
            x = (x_constrained - self.border).clamp(min=1e-8)
        else:
            x = (self.border - x_constrained).clamp(min=1e-8)
        return torch.where(x > 20, x, torch.log(torch.expm1(x)))

    def inverse(self, x_unconstrained):
        y = F.softplus(x_unconstrained) 
        if self.side == "larger":
            return self.border + y
        else:
            return self.border - y

    def derivative(self, x_unconstrained):
        sign = 1 if self.side == "larger" else -1
        return sign * torch.sigmoid(x_unconstrained)

    def log_abs_det_jacobian(self, x_unconstrained):
        return (-F.softplus(-x_unconstrained)).sum()

    def grad_log_abs_det_jacobian(self, x_unconstrained):
        return -F.softplus(-x_unconstrained)


class LogitTransform(Transform):
    def __init__(self, low=0, high=1):
        self.low = low
        self.high = high
        self.scale = high - low
    
    def forward(self, x_constrained):
        x_constrained = x_constrained.clamp(self.low + 1e-8, self.high - 1e-8)
        x_scaled = (x_constrained - self.low) / self.scale
        return torch.log(x_scaled) - torch.log(1 - x_scaled)
    
    def inverse(self, x_unconstrained):
        return 1 / (1 + torch.exp(-x_unconstrained)) * self.scale + self.low

    def derivative(self, x_unconstrained):
        x = self.inverse(x_unconstrained)
        x = x.clamp(self.low + 1e-8, self.high - 1e-8)
        x_scaled = (x - self.low) / self.scale
        return self.scale * x_scaled * (1 - x_scaled)

    def log_abs_det_jacobian(self, x_unconstrained):
        x = self.inverse(x_unconstrained)
        x = x.clamp(self.low + 1e-8, self.high - 1e-8)
        x_scaled = (x - self.low) / self.scale
        return torch.log(self.scale * x_scaled * (1 - x_scaled)).sum()

    def grad_log_abs_det_jacobian(self, x_unconstrained):
        x = self.inverse(x_unconstrained)
        x = x.clamp(self.low + 1e-8, self.high - 1e-8)
        x_scaled = (x - self.low) / self.scale
        return 1 - 2 * x_scaled


class IdentityTransform(Transform):    
    def forward(self, x_constrained):
        return x_constrained
    
    def inverse(self, x_unconstrained):
        return x_unconstrained
    
    def derivative(self, x_unconstrained):
        return torch.ones_like(x_unconstrained)

    def log_abs_det_jacobian(self, x_unconstrained):
        return torch.zeros((), dtype=x_unconstrained.dtype)

    def grad_log_abs_det_jacobian(self, x_unconstrained):
        return torch.zeros_like(x_unconstrained)


class SoftMaxTransform(Transform):
    def __init__(self, dim=-1):
        self.dim = dim

    def forward(self, x_constrained):
        # TODO: add this warning to docs once I make them
        # warn("The forward method of the SoftMaxTransform is not the true inverse of inverse because softmax maps to a lower-dimensional manifold (the simplex). Hence, inverse(forward(x)) == x, but forward(inverse(x)) != x in general.")
        x_constrained = x_constrained.clamp(min=1e-8)
        return torch.log(x_constrained)

    def inverse(self, x_unconstrained):
        e = torch.exp(x_unconstrained - torch.max(x_unconstrained, dim=self.dim, keepdim=True).values)
        return e / torch.sum(e, dim=self.dim, keepdim=True)

    def derivative(self, x_unconstrained):
        # For the softmax transform the Jacobian is not diagonal.
        # Return the full Jacobian matrix for the last dimension.
        sm = self.inverse(x_unconstrained)
        return torch.diag_embed(sm) - sm.unsqueeze(-1) @ sm.unsqueeze(-2)

    def log_abs_det_jacobian(self, x_unconstrained):
        return torch.zeros((), dtype=x_unconstrained.dtype)

    def grad_log_abs_det_jacobian(self, x_unconstrained):
        return torch.zeros_like(x_unconstrained)
