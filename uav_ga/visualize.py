"""All plots (matplotlib, headless)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

COLORS = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#ff7f0e", "#17becf"]


def _base_map(ax, env, show_priority=True):
    N = env.N
    ext = (-0.5, N - 0.5, -0.5, N - 0.5)
    if show_priority:
        ax.imshow(env.priority, origin="lower", extent=ext, cmap="YlOrBr", alpha=0.75, vmin=0, vmax=1)
    ax.contour(env.hazard, levels=[0.3, 0.6, 0.85], colors="purple", linewidths=[0.8, 1.0, 1.3],
               linestyles="--", origin="lower")
    ov = np.ma.masked_where(~env.nfz, np.ones_like(env.priority))
    ax.imshow(ov, origin="lower", extent=ext, cmap="Reds", alpha=0.55, vmin=0, vmax=1.2)
    ax.set_xlim(-0.5, N - 0.5); ax.set_ylim(-0.5, N - 0.5); ax.set_aspect("equal")
    ax.set_xlabel("x (cells)"); ax.set_ylabel("y (cells)")


def _draw_paths(ax, genes):
    for d, route in enumerate(genes):
        col = COLORS[d % len(COLORS)]
        for i in range(len(route) - 1):
            on = route[i, 4] > 0.5
            ax.plot(route[i:i + 2, 0], route[i:i + 2, 1], color=col,
                    lw=(1.2 + 2.2 * route[i, 3]) if on else 1.0,
                    ls="-" if on else ":", alpha=0.95, zorder=3)
        ax.scatter(route[1:, 0], route[1:, 1], s=14, color=col, zorder=4)
        ax.scatter(*route[0, :2], marker="s", s=70, color=col, edgecolor="k", zorder=5)
        ax.annotate(f"D{d + 1}", route[0, :2], xytext=(4, 4), textcoords="offset points",
                    fontsize=8, fontweight="bold", zorder=6)


def _legend(ax, n):
    h = [Line2D([], [], color=COLORS[i], lw=2, label=f"Drone {i + 1}") for i in range(n)]
    h += [Line2D([], [], marker="s", color="w", markerfacecolor="grey", markeredgecolor="k", ms=8, label="Deployment"),
          Line2D([], [], color="grey", lw=3, label="Sensor ON (width = power)"),
          Line2D([], [], color="grey", lw=1, ls=":", label="Sensor OFF"),
          Line2D([], [], marker="o", color="w", markerfacecolor="lime", markeredgecolor="k", ms=7, label="Target detected"),
          Line2D([], [], marker="x", color="red", ls="", ms=7, label="Target missed"),
          Patch(color="red", alpha=0.5, label="No-fly zone"),
          Line2D([], [], color="purple", ls="--", label="Hazard contours")]
    ax.legend(handles=h, fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)


def draw_solution(ax, env, genes, info, title=""):
    _base_map(ax, env)
    _draw_paths(ax, genes)
    det = info["det"] >= 0.5
    ax.scatter(*env.target_xy[det].T, s=26, c="lime", edgecolors="k", linewidths=0.6, zorder=7)
    ax.scatter(*env.target_xy[~det].T, s=30, c="red", marker="x", zorder=7)
    ax.set_title(title, fontsize=10)


def plot_environment(env, path):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.8))
    N = env.N; ext = (-0.5, N - 0.5, -0.5, N - 0.5)
    for ax, (arr, cm, t) in zip(axs[:2], [(env.priority, "YlOrBr", "Mission priority (disaster zones)"),
                                          (env.hazard, "Purples", "Hazard intensity (wind / radiation)")]):
        im = ax.imshow(arr, origin="lower", extent=ext, cmap=cm, vmin=0, vmax=1)
        ax.set_title(t); plt.colorbar(im, ax=ax, fraction=0.046)
    _base_map(axs[2], env)
    axs[2].scatter(*env.target_xy.T, s=30, c="blue", edgecolors="k", zorder=5)
    axs[2].set_title("Combined map: NFZ (red), hazards (dashed), targets (blue)")
    for ax in axs[:2]:
        ax.imshow(np.ma.masked_where(~env.nfz, np.ones_like(env.priority)), origin="lower", extent=ext,
                  cmap="Reds", alpha=0.6, vmin=0, vmax=1.2)
        ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_convergence(h, path, title="GA convergence"):
    g = np.arange(len(h["best"]))
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(g, h["best"], lw=2.2, color="tab:green", label="Best fitness")
    ax.plot(g, h["avg"], lw=1.8, color="tab:blue", label="Average fitness")
    ax.plot(g, h["median"], lw=1.4, color="tab:orange", ls="--", label="Median fitness")
    ax.fill_between(g, h["worst"], h["best"], color="tab:blue", alpha=0.10, label="Worst-to-best range")
    ax.set_ylim(max(min(h["avg"]) * 1.1, -1.0), max(h["best"]) + 0.08)
    ax.set_xlabel("Generation"); ax.set_ylabel("Fitness"); ax.set_title(title)
    ax.grid(alpha=0.3); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_diversity(h, cfg, path):
    g = np.arange(len(h["diversity"]))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(g, h["diversity"], color="tab:purple", lw=2, label="Population diversity")
    ax.axhline(cfg.diversity_threshold, color="red", ls="--", label="Hyper-mutation threshold")
    ax.set_xlabel("Generation"); ax.set_ylabel("Mean gene std (normalised)")
    ax.set_title("Population diversity (premature-convergence monitor)")
    ax.grid(alpha=0.3); ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_paths(env, genes, info, n_drones, path, title="Best evolved UAV flight paths"):
    fig, ax = plt.subplots(figsize=(10, 7))
    draw_solution(ax, env, genes, info, f"{title}\nfitness = {info['fitness']:.3f}")
    _legend(ax, n_drones)
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def plot_coverage(env, info, path, title="Coverage heatmap of the best solution"):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    N = env.N; ext = (-0.5, N - 0.5, -0.5, N - 0.5)
    cov = info["cov_map"]
    for ax, arr, cm, t in [(axs[0], env.priority, "YlOrBr", "Required: mission priority"),
                           (axs[1], cov, "viridis", f"Achieved coverage (priority-weighted {info['coverage']:.1%})"),
                           (axs[2], env.priority * (1 - cov), "Reds", "Remaining uncovered priority")]:
        im = ax.imshow(arr, origin="lower", extent=ext, cmap=cm, vmin=0, vmax=1)
        ax.contour(env.nfz.astype(float), levels=[0.5], colors="red", linewidths=1.2, origin="lower")
        ax.set_title(t, fontsize=10); plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_snapshots(env, ev, history, gens, path):
    gens = sorted(set(g for g in gens if g < len(history["best_genes"])))
    fig, axs = plt.subplots(2, len(gens), figsize=(3.9 * len(gens), 8))
    for j, g in enumerate(gens):
        info = ev.analyze(history["best_genes"][g], g / max(len(history["best"]) - 1, 1))
        draw_solution(axs[0, j], env, history["best_genes"][g], info,
                      f"Generation {g}\nfitness {info['fitness']:+.3f}, coverage {info['coverage']:.0%}")
        im = axs[1, j].imshow(info["cov_map"], origin="lower", cmap="viridis", vmin=0, vmax=1)
        axs[1, j].set_title(f"Coverage heatmap, gen {g}", fontsize=10)
        axs[1, j].contour(env.nfz.astype(float), levels=[0.5], colors="red", linewidths=1)
    fig.suptitle("Evolution of the best UAV plan across generations", fontsize=13)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_ga_vs_random(env, n_drones, ga_g, ga_i, rs_g, rs_i, h, rs_curve, path):
    fig = plt.figure(figsize=(17, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.35, 1])
    for k, (g, i, t) in enumerate([(ga_g, ga_i, "Genetic Algorithm (best)"), (rs_g, rs_i, "Random search (best)")]):
        ax = fig.add_subplot(gs[0, k]); draw_solution(ax, env, g, i, f"{t}\nfitness = {i['fitness']:+.3f}")
    ax = fig.add_subplot(gs[0, 2]); ax.axis("off"); _legend(ax, n_drones)
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(h["evals"], h["best"], color="tab:green", lw=2, label="GA best-so-far")
    ax.plot(h["evals"][:len(rs_curve)], rs_curve, color="tab:red", lw=2, label="Random search best-so-far")
    ax.set_xlabel("Fitness evaluations"); ax.set_ylabel("Fitness"); ax.grid(alpha=0.3); ax.legend()
    ax.set_title("Same evaluation budget")
    ax = fig.add_subplot(gs[1, 1:])
    names = ["coverage", "detection", "energy", "time", "hazard"]
    xs = np.arange(len(names)); w = 0.38
    ax.bar(xs - w / 2, [ga_i[n] for n in names], w, label="GA", color="tab:green")
    ax.bar(xs + w / 2, [rs_i[n] for n in names], w, label="Random", color="tab:red")
    ax.set_xticks(xs); ax.set_xticklabels(["Coverage ↑", "Detection ↑", "Energy ↓", "Time ↓", "Hazard ↓"])
    ax.set_title("Normalised objective values (↑ higher is better, ↓ lower is better)"); ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)
