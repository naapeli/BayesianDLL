import torch

from .. import Model


def find_MAP(model: Model, lr=1e-2, epochs=100, betas=(0.9, 0.999), callback_frequency=1, verbose=True):
    history = []

    m = {}
    v = {}
    t = 0

    for name, param in model.params.items():
        m[name] = torch.zeros_like(param.unconstrained_value)
        v[name] = torch.zeros_like(param.unconstrained_value)

    for epoch in range(epochs):
        t += 1

        for name, param in model.params.items():
            grad = model.grad_log_prob(name, param.unconstrained_value)
            m[name] = betas[0] * m[name] + (1 - betas[0]) * grad
            v[name] = betas[1] * v[name] + (1 - betas[1]) * grad.pow(2)
            m_hat = m[name] / (1 - betas[0] ** t)
            v_hat = v[name] / (1 - betas[1] ** t)
            new_unconstrained_value = param.unconstrained_value + lr * m_hat / (torch.sqrt(v_hat) + 1e-8)
            param.set_unconstrained_value(new_unconstrained_value)

        if verbose and epoch % callback_frequency == 0:
            logp = model.model_log_prob()
            history.append(logp.item())
            print(f"Epoch: {epoch + 1} - Log_prob: {logp.item():.2f}")

    return history

