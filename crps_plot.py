# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# Setup data
x = np.linspace(0, 12, 500)
y_obs = 6.0  # The observed actual value

# Generate a probabilistic forecast (e.g., a Normal distribution)
# Forecast slightly underpredicts the actual value
mu_forecast = 5.0
sigma_forecast = 1.5
cdf_forecast = norm.cdf(x, loc=mu_forecast, scale=sigma_forecast)

# Generate the observation step function (Heaviside step function)
# It is 0 before the observation, and 1 after.
cdf_obs = np.where(x >= y_obs, 1.0, 0.0)

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))

# 1. Plot the forecast CDF
ax.plot(x, cdf_forecast, "b-", linewidth=2, label="Forecast CDF $F(x)$")

# 2. Plot the observation step function
# Use drawstyle='steps-post' so the jump happens exactly at y_obs
ax.plot(
    x,
    cdf_obs,
    "r--",
    linewidth=2,
    drawstyle="steps-post",
    label=f"Observation $y={y_obs}$",
)

# 3. Color in the areas corresponding to the CRPS
# The CRPS is related to the integral of the squared difference between these curves.
# Visually, we shade the area between them.
ax.fill_between(
    x,
    cdf_forecast,
    cdf_obs,
    color="purple",
    alpha=0.4,
    label="Area representing CRPS",
)

# Customization and labels
ax.set_title("Continuous Ranked Probability Score (CRPS)", fontsize=14)
ax.set_xlabel("Value ($x$)", fontsize=12)
ax.set_ylabel("Cumulative Probability", fontsize=12)
ax.set_xlim(0, 12)
ax.set_ylim(-0.02, 1.02)  # Slight padding for visibility

ax.legend(loc="upper left", fontsize=11, frameon=True)
ax.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
plt.show()
