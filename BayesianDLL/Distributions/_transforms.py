import torch
from abc import ABC, abstractmethod


class Transform(ABC):
    @abstractmethod
    def forward(self, x_constrained):
        pass
    
    @abstractmethod
    def inverse(self, x_unconstrained):
        pass

    @abstractmethod
    def derivative(self, x_unconstrained):
        pass

    @abstractmethod
    def grad_log_abs_det_jacobian(x_unconstrained):
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

    def grad_log_abs_det_jacobian(self, x_unconstrained):
        return torch.ones_like(x_unconstrained)
    
# class InverseSoftPlusTransform(Transform):  # TODO: SAMPLING NOT WORKING YET WITH THIS TRANSFORM, BUT SHOULD BE NUMERICALLY MORE STABLE THAN LogTransform
#     def __init__(self, border=0, side="larger"):
#         self.border = border
#         self.side = side

#     def forward(self, x_constrained):
#         if self.side == "larger":
#             x = (x_constrained - self.border).clamp(min=1e-8)
#         else:
#             x = (self.border - x_constrained).clamp(min=1e-8)
#         return torch.log(torch.exp(x) - 1)

#     def inverse(self, x_unconstrained):
#         sign = 1 if self.side == "larger" else -1
#         if self.side == "larger":
#             return self.border + torch.log(1 + torch.exp(sign * x_unconstrained))
#         else:
#             return self.border - torch.log(1 + torch.exp(sign * x_unconstrained))

#     def derivative(self, x_unconstrained):
#         sign = 1 if self.side == "larger" else -1
#         return torch.sigmoid(sign * x_unconstrained)

#     def grad_log_abs_det_jacobian(self, x_unconstrained):
#         sign = 1 if self.side == "larger" else -1
#         sigmoid = torch.sigmoid(sign * x_unconstrained)
#         return -sigmoid.log() / sigmoid.log().abs() * sign * (1 - sigmoid)

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
    
    def grad_log_abs_det_jacobian(self, x_unconstrained):
        return torch.zeros_like(x_unconstrained)
