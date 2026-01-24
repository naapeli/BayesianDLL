import torch

from .. import Model, MeanFieldGuide, sample


def elbo(model: Model, guide: MeanFieldGuide, n_samples=100):
    # uses the REINFORCE estimator or the reparametrization trick if available (https://pyro.ai/examples/svi_part_iii.html or https://mpatacchiola.github.io/blog/2021/02/08/intro-variational-inference-2.html)
    # is the gradient of the elbo (which we would like to maximize), so have to be sure of the signs when optimizing
    grads = {}
    total = 0
    for random_variable_name, param in guide.params.items():
        for variational_param_name, variational_parameter in param.distribution.variational_parameters.items():
            grads[f"{random_variable_name}_{variational_param_name}"] = torch.zeros_like(variational_parameter.value.reshape(1, -1))

    use_reparametrization_trick = {random_variable_name: hasattr(param.distribution, "sample") for random_variable_name, param in guide.params.items()}

    reparam_samples = {}
    reparam_grads = {}
    for random_variable_name, param in guide.params.items():
        if use_reparametrization_trick[random_variable_name]:
            samples, dz_dparams = param.distribution.sample(n_samples=n_samples, _reparametrization_trick_grad=True)
            reparam_samples[random_variable_name] = samples
            reparam_grads[random_variable_name] = dz_dparams
    if not all(use_reparametrization_trick.values()):
        params = guide.params
        guide.params = {name: param for name, param in params.items() if not use_reparametrization_trick[name]}
        reinforce_samples = sample(n_samples, 20, n_chains=1, model=guide, start_point_variance=1, progress_bar=False)
        guide.params = params

    for i in range(n_samples):
        z = {}
        for random_variable_name in guide.params:
            if use_reparametrization_trick[random_variable_name]:
                z[random_variable_name] = reparam_samples[random_variable_name][i:i+1, :]
            else:
                z[random_variable_name] = reinforce_samples[random_variable_name][0, i:i+1, :]

        with model.temporarily_set_many(z):
            log_p = model.model_log_prob()
        with guide.temporarily_set_many(z):
            log_q = guide.model_log_prob()
        elbo_sample = log_p - log_q
        total += elbo_sample

        for random_variable_name, param in guide.params.items():
            if use_reparametrization_trick[random_variable_name]:
                dz_dparams = reparam_grads[random_variable_name]

                grad_z_wrt_elbo = model.grad_log_prob(random_variable_name, z[random_variable_name]) - guide.grad_log_prob(random_variable_name, z[random_variable_name])

                for variational_param_name, dz in dz_dparams.items():
                    key = f"{random_variable_name}_{variational_param_name}"
                    grads[key] += (grad_z_wrt_elbo * dz[i:i+1, :]).sum(dim=1, keepdim=True)
            else:
                # REINFORCE gradient
                grad_dict = param.distribution.log_pdf_param_grads(z[random_variable_name])
                for variational_param_name, grad_val in grad_dict.items():
                    key = f"{random_variable_name}_{variational_param_name}"
                    grads[key] += elbo_sample * grad_val

    for key in grads:
        grads[key] /= n_samples

    return total / n_samples, grads

import torch
from .. import Model, MeanFieldGuide

def elbo(model: Model, guide: MeanFieldGuide, n_samples=100):
    """
    Compute the ELBO and its gradients w.r.t. variational parameters using torch.autograd.
    
    Returns:
        total_elbo: average ELBO over samples
        grads: dict mapping each variational parameter name to its gradient
    """
    # Ensure all variational parameters require gradients
    for rv_name, param in guide.params.items():
        for vp_name, vp in param.distribution.variational_parameters.items():
            vp.value = vp.value.detach().clone().requires_grad_(True)
            vp.value.retain_grad()

    total_elbo = 0.0
    grads = {}
    for random_variable_name, param in guide.params.items():
        for variational_param_name, variational_parameter in param.distribution.variational_parameters.items():
            grads[f"{random_variable_name}_{variational_param_name}"] = torch.zeros_like(variational_parameter.value.reshape(1, -1))

    use_reparametrization_trick = {random_variable_name: hasattr(param.distribution, "sample") for random_variable_name, param in guide.params.items()}

    reparam_samples = {}
    reparam_grads = {}
    for random_variable_name, param in guide.params.items():
        if use_reparametrization_trick[random_variable_name]:
            samples, dz_dparams = param.distribution.sample(n_samples=n_samples, _reparametrization_trick_grad=True)
            reparam_samples[random_variable_name] = samples
            reparam_grads[random_variable_name] = dz_dparams
    if not all(use_reparametrization_trick.values()):
        params = guide.params
        guide.params = {name: param for name, param in params.items() if not use_reparametrization_trick[name]}
        reinforce_samples = sample(n_samples, 20, n_chains=1, model=guide, start_point_variance=1, progress_bar=False)
        guide.params = params

    # Sample and accumulate ELBO
    for i in range(n_samples):
        z = {}
        for random_variable_name in guide.params:
            if use_reparametrization_trick[random_variable_name]:
                z[random_variable_name] = reparam_samples[random_variable_name][i:i+1, :]
            else:
                z[random_variable_name] = reinforce_samples[random_variable_name][0, i:i+1, :]

        with model.temporarily_set_many(z):
            log_p = model.model_log_prob()
        with guide.temporarily_set_many(z):
            log_q = guide.model_log_prob()

        elbo_sample = log_p - log_q
        total_elbo = total_elbo + elbo_sample

    # Average over samples
    total_elbo = total_elbo / n_samples

    # Compute gradients w.r.t. variational parameters
    grads = {}
    total_elbo.backward()
    for rv_name, param in guide.params.items():
        for vp_name, vp in param.distribution.variational_parameters.items():
            grads[f"{rv_name}_{vp_name}"] = vp.value.grad.clone().detach()

    return total_elbo.detach(), grads
