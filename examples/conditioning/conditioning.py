import matplotlib.pyplot as plt
import torch

from BayesianDLL import Data, DeterministicParameter, Model, ObservedParameter, RandomParameter, condition, plate
from BayesianDLL.Distributions import HalfCauchy, Normal
from BayesianDLL.Evaluation import summary


torch.manual_seed(42)

N = 30
true_intercept = 1.5
true_slope = 2.0
true_sigma = 0.5

x_raw = torch.linspace(-2.0, 2.0, N, dtype=torch.float64)
noise = torch.randn(N, dtype=torch.float64) * true_sigma
y_raw = true_intercept + true_slope * x_raw + noise


with Model() as model:
    x_data = Data("x", x_raw)
    y_data = Data("y", y_raw)

    intercept = RandomParameter("intercept", Normal(0.0, 5.0))
    slope = RandomParameter("slope", Normal(0.0, 5.0))
    sigma = RandomParameter("sigma", HalfCauchy(1.0))

    mu = DeterministicParameter(
        "mu",
        lambda b, m, x_val: m * x_val + b,
        lambda b, m, x_val: {"slope": x_val, "intercept": torch.ones_like(x_val)},
        [intercept, slope, x_data],
    )

    with plate("data", x_data):
        ObservedParameter("obs", Normal(mu, sigma), y_data)


sampling_options = {
    "n_samples": 1000,
    "warmup_length": 500,
    "n_chains": 2,
    "progress_bar": True,
    "delta": 0.8,
}

print("\n--- Sampling Model with Original Dataset ---")
res_full = model.sample(**sampling_options)
print(summary(res_full, round_to=3))


N_alternative = 18
alternative_intercept = -0.75
alternative_slope = -1.25
x_alternative = torch.linspace(-3.0, 3.0, N_alternative, dtype=torch.float64)
y_alternative = (
    alternative_intercept
    + alternative_slope * x_alternative
    + torch.randn(N_alternative, dtype=torch.float64) * true_sigma
)

print("\n--- Conditioning on an Alternative Dataset ---")
model_alternative_data = model.condition(x=x_alternative, y=y_alternative)

res_alternative_data = model_alternative_data.sample(**sampling_options)
print(summary(res_alternative_data, round_to=3))


print("\n--- Conditioning on Intercept = 0.0 ---")
model_zero_intercept = model.condition(intercept=0.0)

res_zero_intercept = model_zero_intercept.sample(**sampling_options)
print(summary(res_zero_intercept, round_to=3))

print("\n--- Sensitivity Analysis on Observation Noise (sigma) ---")
model_low_noise = condition(model, {"sigma": 0.02})
model_high_noise = condition(model, {"sigma": 2})

res_low_noise = model_low_noise.sample(**sampling_options)
res_high_noise = model_high_noise.sample(**sampling_options)

print("\nPosterior summary with assumed low noise (sigma = 0.02):")
print(summary(res_low_noise, round_to=3))

print("\nPosterior summary with assumed high noise (sigma = 2):")
print(summary(res_high_noise, round_to=3))

_, axes = plt.subplots(1, 3, figsize=(18, 5), layout="constrained")

x_line = torch.linspace(
    min(x_raw.min(), x_alternative.min()),
    max(x_raw.max(), x_alternative.max()),
    200,
    dtype=torch.float64,
)

intercept_full_mean = res_full["intercept"].mean()
slope_full_mean = res_full["slope"].mean()
intercept_alternative_mean = res_alternative_data["intercept"].mean()
slope_alternative_mean = res_alternative_data["slope"].mean()

axes[0].scatter(x_raw, y_raw, color="C0", alpha=0.7, label=f"Original data (n={N})")
axes[0].scatter(
    x_alternative,
    y_alternative,
    color="C4",
    marker="D",
    alpha=0.7,
    label=f"Conditioned data (n={N_alternative})",
)
axes[0].plot(
    x_line,
    intercept_full_mean + slope_full_mean * x_line,
    color="C0",
    linewidth=2,
    label="Posterior mean: original",
)
axes[0].plot(
    x_line,
    intercept_alternative_mean + slope_alternative_mean * x_line,
    color="C4",
    linewidth=2,
    label="Posterior mean: conditioned copy",
)
axes[0].set_title("Conditioning the Same Model on New Data", fontsize=11)
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
axes[0].legend(frameon=False)
axes[0].grid(alpha=0.2)

slope_full = res_full["slope"].flatten().numpy()
slope_zero = res_zero_intercept["slope"].flatten().numpy()

axes[1].hist(slope_full, bins=25, density=True, alpha=0.55, color="C0", label="Unconditioned (latent intercept)")
axes[1].hist(slope_zero, bins=25, density=True, alpha=0.55, color="C3", label="Conditioned: intercept = 0.0")
axes[1].axvline(true_slope, color="black", linestyle="--", linewidth=1.5, label=f"True slope ({true_slope})")
axes[1].set_title("What-If Analysis: Effect on Slope with Intercept Fixed to 0", fontsize=11)
axes[1].set_xlabel("Slope")
axes[1].set_ylabel("Posterior Density")
axes[1].legend(frameon=False)
axes[1].grid(alpha=0.2)

slope_low = res_low_noise["slope"].flatten().numpy()
slope_high = res_high_noise["slope"].flatten().numpy()

axes[2].hist(slope_low, bins=25, density=True, alpha=0.55, color="C2", label="Low noise (sigma = 0.02)")
axes[2].hist(slope_full, bins=25, density=True, alpha=0.4, color="C0", label="Inferred noise (sigma ~ HalfCauchy)")
axes[2].hist(slope_high, bins=25, density=True, alpha=0.55, color="C1", label="High noise (sigma = 2)")
axes[2].axvline(true_slope, color="black", linestyle="--", linewidth=1.5, label=f"True slope ({true_slope})")
axes[2].set_title("Sensitivity Analysis: Slope Uncertainty vs Assumed Noise", fontsize=11)
axes[2].set_xlabel("Slope")
axes[2].set_ylabel("Posterior Density")
axes[2].legend(frameon=False)
axes[2].grid(alpha=0.2)

plt.show()
