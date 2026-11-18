import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

min_sigma = 0.1
K = 3
N = 3000
steps = 100

mus = np.array([0, -10, 10])
sigmas = np.array([1, 5, 5])

x_1 = np.random.normal(mus[0], sigmas[0], N//K)
x_2 = np.random.normal(mus[1], sigmas[1], N//K)
x_3 = np.random.normal(mus[2], sigmas[2], N//K)

x = np.append(np.append(x_1, x_2), x_3)

guess_mus = np.random.choice(x, 3)
print(guess_mus)
guess_sigmas = np.array([1, 1, 1])
guess_coefs = np.array([1/K, 1/K, 1/K])

guess_gammas = np.zeros((N, K))

e_K = np.ones(K)
e_N = np.ones(N)

def pdf(x, mu, sigma):
    coefficient = 1/(np.sqrt(2*np.pi) * sigma)
    exponent = -((x-mu) ** 2)/(2 * (sigma**2))
    return coefficient * np.exp(exponent)

mu_history = []
sigma_history = []
coef_history = []
cross_entropy_history = []

for step in range(steps):
    for k in range(K):
        guess_gammas[:, k] = guess_coefs[k] * pdf(x, guess_mus[k], guess_sigmas[k])

    guess_sum = (guess_gammas @ e_K).reshape(N, 1) + 1e-12
    guess_gammas = guess_gammas / guess_sum

    gamma_sum = e_N @ guess_gammas
    guess_coefs = gamma_sum / N
    guess_sigmas = np.sqrt(e_N @ (guess_gammas * (np.column_stack([x, x, x]) - guess_mus)**2) / gamma_sum)
    guess_mus = e_N @ (guess_gammas * x.reshape(N, 1)) / gamma_sum

    mu_history.append(guess_mus.copy())
    sigma_history.append(guess_sigmas.copy())
    coef_history.append(guess_coefs.copy())
    
    mixture_pdf = np.zeros(N)
    for k in range(K):
        mixture_pdf += guess_coefs[k] * pdf(x, guess_mus[k], guess_sigmas[k])
    ce = -np.mean(np.log(mixture_pdf + 1e-12))
    cross_entropy_history.append(ce)

print(mus, guess_mus)
print(sigmas, guess_sigmas)

linspace = 1000
x_plot = np.linspace(-25, 25, linspace)

true_pdf_plot = np.zeros((linspace, K))
for k in range(K):
    true_pdf_plot[:,k] = 1/K * pdf(x_plot, mus[k], sigmas[k])

guess_pdf_plot = np.zeros((linspace, K))
for k in range(K):
    guess_pdf_plot[:,k] = coef_history[0][k] * pdf(x_plot, mu_history[0][k], sigma_history[0][k])

fig = make_subplots(rows=2, cols=1)
fig.add_trace(go.Histogram(x=x, nbinsx=100, histnorm='probability density', opacity=0.6, name='Data'), row=1, col=1)
for k in range(K):
    fig.add_trace(go.Scatter(x=x_plot, y=true_pdf_plot[:,k], mode='lines', name=f'True PDF {k+1}', line=dict(color='red', dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_plot, y=guess_pdf_plot[:,k], mode='lines', name=f'Guess PDF {k+1}'), row=1, col=1)
fig.add_trace(go.Scatter(x=[0], y=[cross_entropy_history[0]], mode='lines', name='Cross-Entropy'), row=2, col=1)

slider_steps = []
for i in range(0, steps):
    guess_pdf_plot = np.zeros((linspace, K))
    for k in range(K):
        guess_pdf_plot[:,k] = coef_history[i][k] * pdf(x_plot, mu_history[i][k], sigma_history[i][k])
    step = dict(
    method="update",
    args=[
        {"y": [
            None,
            true_pdf_plot[:,0],
            guess_pdf_plot[:,0],
            true_pdf_plot[:,1],
            guess_pdf_plot[:,1],
            true_pdf_plot[:,2],
            guess_pdf_plot[:,2],
            cross_entropy_history[:i+1]
        ],
         "x": [
             x,
             x_plot,
             x_plot,
             x_plot,
             x_plot,
             x_plot,
             x_plot,
             list(range(i+1))
         ]}
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
