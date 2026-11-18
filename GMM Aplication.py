import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import aiohttp
import asyncio
import pandas as pd
import matplotlib.pyplot as plt

async def live_price():
    url = "https://api.blockchain.info/charts/market-price?timespan=all&format=json"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            data = await resp.json()
            dataframe = pd.DataFrame(data["values"]).rename(columns={"x": "date", "y": "market_price"})
            dataframe["date"] = (pd.to_datetime(dataframe['date'], unit="s") - pd.Timedelta(days=1))
            dataframe = dataframe.astype({"date": str}).set_index("date")
            full_idx = pd.date_range(start=dataframe.index[0], end=dataframe.index[-1], freq="D").astype(str)
            dataframe = dataframe.reindex(full_idx)
            return dataframe.interpolate(method="linear")
        

if __name__ == "__main__":
    dataframe = asyncio.run(live_price()).pct_change().dropna().rename(columns={"market_price": "price_change"})

    # dates = ["2021-11-01", "2022-11-25"]
    dates = [ "2022-11-25",  "2026-02-02"]

    min_sigma = 0.1
    K = 3
    steps = 300

    x = dataframe[dates[0]:dates[1]]["price_change"].to_numpy() + 1
    N=len(x)

    guess_mus = np.random.choice(x, K)
    guess_sigmas = np.array([1, 1, 1]) / 100
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

    for step in range(steps):
        for k in range(K):
            guess_gammas[:, k] = guess_coefs[k] * pdf(x, guess_mus[k], guess_sigmas[k])

        guess_sum = (guess_gammas @ e_K).reshape(N, 1) + 1e-12
        guess_gammas = guess_gammas / guess_sum

        gamma_sum = e_N @ guess_gammas
        guess_coefs = gamma_sum / N
        guess_sigmas = np.sqrt(e_N @ (guess_gammas * (np.column_stack([x, x, x]) - guess_mus)**2) / gamma_sum)
        guess_mus = e_N @ (guess_gammas * x.reshape(N, 1)) / gamma_sum

        # overfitting
        # guess_sigmas[guess_sigmas <= 0.01] = 0.01
        # guess_coefs[guess_coefs <= 0.15] = 0.15
        # guess_coefs = guess_coefs / guess_coefs.sum()

        mu_history.append(guess_mus.copy())
        sigma_history.append(guess_sigmas.copy())
        coef_history.append(guess_coefs.copy())

    print("guess_coefs : ", guess_coefs)
    print("guess_mus : ", guess_mus)
    print("guess_sigmas : ", guess_sigmas)

    regimes = guess_gammas.argmax(axis=1)

    print(regimes)
    linspace = 1000
    x_plot = np.linspace(0.9, 1.1, linspace)


    guess_pdf_plot = np.zeros((linspace, K))
    true_pdf = np.zeros(linspace)
    for k in range(K):
        guess_pdf_plot[:,k] = coef_history[0][k] * pdf(x_plot, mu_history[0][k], sigma_history[0][k])
        true_pdf += guess_pdf_plot[:, k]


    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Histogram(x=x, nbinsx=100, histnorm='probability density', opacity=0.6, name='Data'), row=1, col=1)
    for k in range(K):
        fig.add_trace(go.Scatter(x=x_plot, y=guess_pdf_plot[:,k], mode='lines', name=f'Guess PDF {k+1}'), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_plot, y=true_pdf, mode='lines', name=f'True PDF'), row=1, col=1)

    slider_steps = []
    for i in range(0, steps):
        guess_pdf_plot = np.zeros((linspace, K))
        true_pdf = np.zeros(linspace)
        for k in range(K):
            guess_pdf_plot[:,k] = coef_history[i][k] * pdf(x_plot, mu_history[i][k], sigma_history[i][k])
            true_pdf += guess_pdf_plot[:,k]
        step = dict(
        method="update",
        args=[
            {"y": [
                None,
                guess_pdf_plot[:,0],
                guess_pdf_plot[:,1],
                guess_pdf_plot[:,2],
                true_pdf,
            ],
            "x": [
                x,
                x_plot,
                x_plot,
                x_plot,
                x_plot,
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
