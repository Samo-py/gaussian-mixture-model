import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import aiohttp
import asyncio
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

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

    today = datetime.now().date()
    previous_time = today - timedelta(365 * 2)
    dates = [previous_time.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")]
    min_sigma = 0.1
    K = 2
    steps = 300

    x = dataframe[dates[0]:dates[1]]["price_change"].to_numpy() + 1
    N=len(x)
    guess_mus = np.random.choice(x, K)
    guess_sigmas = np.array([1, 1]) / 100
    guess_coefs = np.array([1/K, 1/K])

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
        guess_sigmas = np.sqrt(e_N @ (guess_gammas * (np.column_stack([x, x]) - guess_mus)**2) / gamma_sum)
        guess_mus = e_N @ (guess_gammas * x.reshape(N, 1)) / gamma_sum


        mu_history.append(guess_mus.copy())
        sigma_history.append(guess_sigmas.copy())
        coef_history.append(guess_coefs.copy())

    print("guess_coefs : ", guess_coefs)
    print("guess_mus : ", guess_mus)
    print("guess_sigmas : ", guess_sigmas)

    regimes = guess_gammas.argmax(axis=1)





    time = np.arange(len(regimes))





    colors = ["#FFDDDD", "#DDFFDD"]
    regime_labels = ["Regime 0", "Regime 1"]

    fig, ax = plt.subplots(figsize=(12, 4))

    start = 0
    for t in range(1, N):
        if regimes[t] != regimes[t-1] or t == N-1:
            ax.axvspan(start, t, color=colors[regimes[t-1]], alpha=0.3)
            start = t

    ax.plot(time, x.cumprod(), color='tab:blue', linewidth=2)

    for t in range(1, N):
        if regimes[t] != regimes[t-1]:
            ax.axvline(t, color='gray', linestyle='--', alpha=0.5)

    ax.set_xlabel("Time step")
    ax.set_ylabel("Price")
    ax.set_title("Price with Regime Background (2 regimes)")
    ax.grid(True, alpha=0.3)

    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor=colors[i], alpha=0.3, label=regime_labels[i]) for i in range(2)]
    ax.legend(handles=legend_patches, loc='upper left')

    plt.tight_layout()
    plt.show()