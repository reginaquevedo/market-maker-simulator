"""Run the full experiment: compare strategies, sweep risk aversion, save figures."""
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from marketmaker import MarketParams, AvellanedaStoikov, SymmetricStrategy, simulate

FIG = Path(__file__).parent / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3, "font.size": 10})
C_AS, C_SYM = "#1f5fbf", "#d9822b"


def results_table(summaries):
    lines = ["| Strategy | Mean PnL | Std PnL | Sharpe | Mean abs. final inventory |",
             "|---|---|---|---|---|"]
    for s in summaries:
        lines.append(f"| {s['strategy']} | {s['mean_pnl']:.2f} | {s['std_pnl']:.2f} | "
                     f"{s['sharpe']:.2f} | {s['mean_abs_final_inventory']:.2f} |")
    return "\n".join(lines)


def plot_sample_path(res, p):
    t = np.arange(p.n_steps + 1) * p.dt
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    j = 0
    ax1.plot(t, res.mid[:, j], color="black", lw=1.2, label="Mid-price")
    ax1.plot(t[:-1], res.bid[:, j], color="#2e9e5b", lw=0.8, label="Bid quote")
    ax1.plot(t[:-1], res.ask[:, j], color="#c23b3b", lw=0.8, label="Ask quote")
    ax1.set_ylabel("Price")
    ax1.set_title("Avellaneda-Stoikov market maker: one simulated trading session")
    ax1.legend(loc="upper left")
    ax2.step(t, res.inventory[:, j], where="post", color=C_AS)
    ax2.axhline(0, color="gray", lw=0.8)
    ax2.set_ylabel("Inventory")
    ax2.set_xlabel("Time")
    fig.tight_layout()
    fig.savefig(FIG / "sample_path.png")
    plt.close(fig)


def plot_distributions(res_as, res_sym):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    bins = np.linspace(min(res_as.final_pnl.min(), res_sym.final_pnl.min()),
                       max(res_as.final_pnl.max(), res_sym.final_pnl.max()), 50)
    ax1.hist(res_sym.final_pnl, bins=bins, alpha=0.6, color=C_SYM, label="Symmetric")
    ax1.hist(res_as.final_pnl, bins=bins, alpha=0.6, color=C_AS, label="Avellaneda-Stoikov")
    ax1.set_title("Final PnL distribution")
    ax1.set_xlabel("PnL")
    ax1.legend()
    lim = int(max(np.abs(res_as.final_inventory).max(), np.abs(res_sym.final_inventory).max())) + 1
    ibins = np.arange(-lim, lim + 2) - 0.5
    ax2.hist(res_sym.final_inventory, bins=ibins, alpha=0.6, color=C_SYM, label="Symmetric")
    ax2.hist(res_as.final_inventory, bins=ibins, alpha=0.6, color=C_AS, label="Avellaneda-Stoikov")
    ax2.set_title("Final inventory distribution")
    ax2.set_xlabel("Inventory at close")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(FIG / "distributions.png")
    plt.close(fig)


def plot_frontier(p, gammas, n_paths):
    rows = []
    for g in gammas:
        s = simulate(AvellanedaStoikov(), replace(p, gamma=g), n_paths=n_paths, seed=1).summary()
        rows.append((g, s["mean_pnl"], s["std_pnl"]))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    g, m, sd = map(np.array, zip(*rows))
    ax.plot(sd, m, "-o", color=C_AS)
    for gi, x, y in rows:
        ax.annotate(f"γ={gi:g}", (y, x), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("Std of final PnL (risk)")
    ax.set_ylabel("Mean final PnL (return)")
    ax.set_title("Risk/return trade-off as risk aversion γ varies")
    fig.tight_layout()
    fig.savefig(FIG / "risk_frontier.png")
    plt.close(fig)
    return rows


def main():
    p = MarketParams()
    n_paths = 2000
    res_as = simulate(AvellanedaStoikov(), p, n_paths=n_paths, seed=42)
    res_sym = simulate(SymmetricStrategy(), p, n_paths=n_paths, seed=42)

    table = results_table([res_as.summary(), res_sym.summary()])
    print(table)

    plot_sample_path(res_as, p)
    plot_distributions(res_as, res_sym)
    rows = plot_frontier(p, [0.01, 0.05, 0.1, 0.5, 1.0], n_paths=n_paths)
    print("\ngamma sweep (gamma, mean, std):")
    for r in rows:
        print(f"  {r[0]:>5}: {r[1]:7.2f} {r[2]:7.2f}")
    (Path(__file__).parent / "results.md").write_text(table + "\n")


if __name__ == "__main__":
    main()
