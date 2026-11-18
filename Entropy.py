import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

mu = 5
sigma = 5

guess_mu = 0
guess_sigma = 1

lr = 0.1
steps = 1000
samples = 1000


selection = np.random.normal(mu, sigma, samples)
mean = np.mean(selection)
var = np.var(selection)

mu_history = [guess_mu]
sigma_history = [guess_sigma]
cross_entropy_history = []

def gaussian_cross_entropy(mu_true, sigma_true, mu_guess, sigma_guess):
    return 0.5 * np.log(2 * np.pi * sigma_guess**2) + \
           (sigma_true**2 + (mu_true - mu_guess)**2) / (2 * sigma_guess**2)

for i in range(steps):
    gradient_delta_mu = (guess_mu - mean) / guess_sigma**2
    gradient_delta_sigma = (guess_sigma**2 - var) / guess_sigma**3

    guess_mu += -lr*gradient_delta_mu
    guess_sigma += -lr*gradient_delta_sigma
    mu_history.append(guess_mu)
    sigma_history.append(guess_sigma)

    cross_entropy_history.append(gaussian_cross_entropy(mu, sigma, guess_mu, guess_sigma))

x = np.linspace(-15 + mu, 15 + mu, 1000)
true_pdf = 1/(np.sqrt(2 * np.pi) * sigma) * np.exp(-(x - mu)**2 / (2 * sigma**2))

fig = make_subplots(rows=2, cols=1, shared_xaxes=False,
                    vertical_spacing=0.15,
                    subplot_titles=("PDFs", "Cross-Entropy (last 500 steps)"))

fig.add_trace(go.Scatter(x=x, y=true_pdf, mode='lines', name='True Dist', line=dict(color="red", dash="dash")), row=1, col=1)
fig.add_trace(
    go.Histogram(
        x=selection,
        nbinsx=50,
        histnorm='probability',
        name='True Dist',
        opacity=0.6,
        marker_color='blue'
    ),
    row=1, col=1
)

guess_pdf = 1/(np.sqrt(2 * np.pi) * sigma_history[0]) * np.exp(-(x - mu_history[0])**2 / (2 * sigma_history[0]**2))

fig.add_trace(go.Scatter(x=x, y=guess_pdf, mode='lines', name='Guess Dist'), row=1, col=1)
fig.add_trace(go.Scatter(x=[0], y=[cross_entropy_history[0]], mode='lines', name='Cross-Entropy'), row=2, col=1)

slider_steps = []
window = 500

for i in range(len(mu_history)):
    guess_pdf = 1/(np.sqrt(2 * np.pi) * sigma_history[i]) * np.exp(-(x - mu_history[i])**2 / (2 * sigma_history[i]**2))
    
    # start_idx = max(0, i - window)
    # ce_slice = cross_entropy_history[start_idx:i+1]
    # ce_x = list(range(start_idx, i+1))
    ce_slice = cross_entropy_history[0:i+1]
    ce_x = list(range(0, i+1))
    
    step = dict(
        method="update",
        args=[
            {"y": [true_pdf, None, guess_pdf, ce_slice],
             "x": [x, selection, x, ce_x]}
        ],
        label=str(i)
    )
    slider_steps.append(step)

fig.update_layout(
    height=700,
    sliders=[dict(active=0, steps=slider_steps)],
    showlegend=True
)

fig.show()
