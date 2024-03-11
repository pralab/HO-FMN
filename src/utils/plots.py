import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from ax.plot.contour import plot_contour_plotly
from ax.plot.slice import plot_slice_plotly

import torch
import numpy as np

mpl.rc('font', size=40)
mpl.rcParams['figure.dpi'] = 100


def plot_contour(model=None, param_x='lr', param_y='momentum', metric_name='distance'):
    if model is None:
        raise ValueError('Please provide a model!')

    # plotly returns a Figure object, you can access plots' data through data.data (the list of plots' data)
    data = plot_contour_plotly(
        model=model,
        param_x=param_x,
        param_y=param_y,
        metric_name=metric_name,
        lower_is_better=True)

    plots_data = data.data
    mean_contour = plots_data[0]
    std_contour = plots_data[1]
    scatter_plot = data.data[-1]

    fig, ax = plt.subplots(1,2, figsize=(44,15))
    cs1 = ax[0].contourf(mean_contour.x, mean_contour.y, mean_contour.z, levels=15, cmap='Greens', alpha=0.8)
    ax[0].contour(mean_contour.x, mean_contour.y, mean_contour.z, levels=15, colors='black', linewidths=2)
    ax[0].scatter(scatter_plot.x, scatter_plot.y, color='#73130d', marker='s',  s=110, zorder=10)
    cbar = fig.colorbar(cs1, ax=ax[0], shrink=0.7, location='right', format='{x:.4f}')
    ax[0].set_xlim(8/255, 10)
    ax[0].set_ylim(0.0, 0.9)
    ax[0].set_xscale('log')
    ax[0].set_xlabel(r'$\gamma$')
    ax[0].set_ylabel(r'$\mu$')
    ax[0].set_title(r'Mean($\hat D(C,(h))$)')

    cs2 = ax[1].contourf(std_contour.x, std_contour.y, std_contour.z, levels=15, cmap='Blues', alpha=0.78)
    ax[1].contour(std_contour.x, std_contour.y, std_contour.z, levels=15, colors='black', linewidths=2)
    ax[1].scatter(scatter_plot.x, scatter_plot.y, color='#73130d', marker='s', s=110, zorder=10)
    cbar = fig.colorbar(cs2, ax=ax[1], shrink=0.7, location='right', format='{x:.4f}')
    ax[1].set_xlim(8/255, 10)
    ax[1].set_ylim(0.0, 0.9)
    ax[1].set_xscale('log')
    ax[1].set_xlabel(param_x)
    ax[1].set_ylabel(param_y)
    ax[1].set_title(r'Std($\hat D(C,(h))$)')

    plt.show()


def plot_slice(model=None, param_name='lr', metric_name='distance'):
    if model is None:
        raise ValueError('Please provide a model!')

    data = plot_slice_plotly(
        model=model,
        param_name=param_name,
        metric_name=metric_name)
    plots_data = data.data

    shadow = plots_data[0]
    line = plots_data[1]
    shadow_up_x = shadow.x[0:50]
    shadow_down_x = np.array(shadow.x[50:])
    shadow_up_y = shadow.y[0:50]
    shadow_down_y = shadow.y[50:]

    plt.figure(figsize=(12,8))
    ax = plt.axes([0, 1, 2.5, 1])
    ax.plot(line.x, line.y, color='#80b1d3', linewidth=5)
    ax.plot(shadow_up_x, shadow_up_y, color='#80b1d380')
    ax.plot(shadow_down_x, shadow_down_y, color='#80b1d380')
    ax.fill_between(line.x,  line.y, shadow_up_y, color='#80b1d380')
    ax.fill_between(shadow_down_x[::-1],line.y, shadow_down_y[::-1],  color='#80b1d380')
    ax.set_facecolor("#80b1d333")
    ax.set_xscale('log')
    ax.set_xlabel(param_name)
    ax.set_ylabel(r"$\hat D(C,(h))$")
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.set_xlim(8/255, 10)
    ax.grid(color='white')
    plt.gca().xaxis.set_major_formatter(ScalarFormatter())


def compare_adv_robustness(baseline_best_adv, tuned_best_adv, images):
    figure, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=120)

    pert_sizes = torch.linspace(0, 0.15, 1000).unsqueeze(1)
    norms = (tuned_best_adv.cpu() - images.cpu()).flatten(1).norm(torch.inf, dim=1)
    base_norms = (baseline_best_adv.cpu() - images.cpu()).flatten(1).norm(torch.inf, dim=1)

    rob_acc = (norms > 8 / 255).float().mean().item()
    base_rob_acc = (base_norms > 8 / 255).float().mean().item()

    acc = (norms > 0).float().mean().item()

    print(f"Clean accuracy:\n\t{acc}")
    print(f"Baseline Robust accuracy (at 8/255):\n\t{base_rob_acc}")
    print(f"Robust accuracy (at 8/255):\n\t{rob_acc}")

    base_norms = (base_norms > pert_sizes).float().mean(dim=1)
    norms = (norms > pert_sizes).float().mean(dim=1)

    ax.plot(pert_sizes, norms, color='blue', label='HO-FMN', lw=0.8)
    ax.plot(pert_sizes, base_norms, color='grey', label='FMN', lw=0.8, linestyle='--')

    ax.axvline(x=8 / 255, color='#EE004D', linewidth=1, linestyle='--')
    ax.scatter(8 / 255, rob_acc, marker='s', color='red')

    ax.grid(True)
    ax.legend()

    ax.set_xlim(0.0, 0.15)
    ax.set_ylim(0.0, 1.0)

    plt.show()