import torch

from .. import Model, MeanFieldGuide, sample


def elbo(model: Model, guide: MeanFieldGuide, n_samples=1):
    # uses the REINFORCE estimator or the reparametrization trick if available (https://pyro.ai/examples/svi_part_iii.html or https://mpatacchiola.github.io/blog/2021/02/08/intro-variational-inference-2.html)
    # is the gradient of the elbo (which we would like to maximize), so have to be sure of the signs when optimizing
    grads = {}
    total = 0
    for random_variable_name, param in guide.params.items():
        for variational_param_name, variational_parameter in param.distribution.variational_parameters.items():
            grads[f"{random_variable_name}_{variational_param_name}"] = torch.zeros_like(variational_parameter.value)

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
                z[random_variable_name] = reparam_samples[random_variable_name][i]
            else:
                z[random_variable_name] = reinforce_samples[random_variable_name][0, i]

        with model.temporarily_set_many(z):
            log_p = model.model_log_prob()
        with guide.temporarily_set_many(z):
            log_q = guide.model_log_prob()
        elbo_sample = log_p - log_q
        total += elbo_sample

        for random_variable_name, param in guide.params.items():
            if use_reparametrization_trick[random_variable_name]:
                # reparametrization trick (exactly the same as torch.autograd)
                dz_dparams = reparam_grads[random_variable_name]

                with model.temporarily_set_many(z):
                    grad_z_wrt_elbo = model.grad_log_prob(random_variable_name, z[random_variable_name]) - guide.grad_log_prob(random_variable_name, z[random_variable_name])
                grad_dict = param.distribution.log_pdf_param_grads(z[random_variable_name])
                for variational_param_name, dz in dz_dparams.items():
                    key = f"{random_variable_name}_{variational_param_name}"
                    grads[key] += (grad_z_wrt_elbo * dz[i]).sum() - grad_dict[variational_param_name].sum()
            else:
                # REINFORCE gradient (higher variance and not exact, but with n_samples high, close to the correct estimate)
                grad_dict = param.distribution.log_pdf_param_grads(z[random_variable_name])
                for variational_param_name, grad_val in grad_dict.items():
                    key = f"{random_variable_name}_{variational_param_name}"
                    grads[key] += elbo_sample * grad_val

    for key in grads:
        grads[key] /= n_samples

    return total / n_samples, grads
