import torch
from . import elbo
from .. import Model, MeanFieldGuide


def BBVI(model: Model, guide: MeanFieldGuide, n_samples=100, epochs=100, lr=1e-2, betas=(0.9, 0.999), callback_frequency=1, verbose=True):
    """
    Black-box variational inference using Monte Carlo ELBO and score-function gradients.
    
    Args:
        model: Model instance (supports temporarily_set_many and model_log_prob)
        guide: MeanFieldGuide instance (supports params and log_pdf_param_grads)
        n_samples: number of Monte Carlo samples per ELBO estimate
        epochs: number of optimization steps
        lr: learning rate
        betas: Adam beta parameters (beta1, beta2)
        callback_frequency: print progress every this many epochs
        verbose: whether to print ELBO during training
    """
    m = {name + "_" + param_name: torch.zeros_like(param.value) for name in guide.params for param_name, param in guide.params[name].distribution.variational_parameters.items()}
    v = {name + "_" + param_name: torch.zeros_like(param.value) for name in guide.params for param_name, param in guide.params[name].distribution.variational_parameters.items()}
    t = 0

    history = []

    for epoch in range(1, epochs + 1):
        t += 1
        current_elbo, grads = elbo(model, guide, n_samples=n_samples)

        for key, grad in grads.items():
            param_tensor = None
            for random_variable_name, random_variable in guide.params.items():
                for variational_parameter_name, variational_parameter in random_variable.distribution.variational_parameters.items():
                    if random_variable_name + "_" + variational_parameter_name == key:
                        current_variational_parameter = variational_parameter
                        param_tensor = variational_parameter.value
                        break
                if param_tensor is not None:
                    break

            if param_tensor is None:
                raise ValueError(f"Parameter {key} not found in guide.")

            m[key] = betas[0] * m[key] + (1 - betas[0]) * grad
            v[key] = betas[1] * v[key] + (1 - betas[1]) * (grad ** 2)

            m_hat = m[key] / (1 - betas[0] ** t)
            v_hat = v[key] / (1 - betas[1] ** t)

            new_value = current_variational_parameter.value + lr * m_hat / (torch.sqrt(v_hat) + 1e-8)
            current_variational_parameter.set_new_value(new_value)

        if verbose and epoch % callback_frequency == 0:
            history.append(current_elbo.item())
            print(f"Epoch {epoch} | ELBO: {current_elbo.item():.4f}")

    return history
