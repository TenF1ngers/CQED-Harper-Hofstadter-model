import numpy as np
import qutip as qt
from numpy import pi
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from datetime import datetime
from scipy.optimize import curve_fit
from tools import FitLin

class Interpreter:

  def __init__(self):
    pass
  
  def PlotInterference(self, calculations, experiment_config, num_atoms):
    data_full = []

    site_monitor_id_list = experiment_config['site_monitor_id_list'].copy()

    for i in range(len(site_monitor_id_list)):
      data_full.append([calculations[phi_id][i] for phi_id in range(len(experiment_config['phi_list'][0]) - 1, -1, -1)])

    fig, axes = plt.subplots(1, len(site_monitor_id_list), figsize=(2 * len(site_monitor_id_list), 2), sharey=True, dpi=200)
    axes = np.array(axes).reshape(len(site_monitor_id_list))

    axes[0].set_ylabel(r"Flux [$\Phi_0$]")

    for i in range(axes.shape[0]):
      image = axes[i].pcolormesh(experiment_config['operating_time_points'], experiment_config['phi_list'][0], data_full[i], cmap='inferno')
      axes[i].set_xlabel("t [ns]", fontsize=10)
      axes[i].set_yticks([-pi, 0, pi])
      axes[i].set_yticklabels([r"-$\pi$", r"$0$", r"$\pi$"])
      axes[i].text(1, 1.15 * experiment_config['phi_list'][0][-1], f"({site_monitor_id_list[i]})")

    cax = fig.add_axes([1.01, 0.38, 0.02, 0.4])
    cbar = fig.colorbar(image, cax=cax)
    cbar.set_ticks([0, np.max(data_full[axes.shape[0] - 1]) / 2, np.max(data_full[axes.shape[0] - 1])])
    cbar.set_ticklabels(["0", "", f"{np.max(data_full[axes.shape[0] - 1]):.1f}"])
    cbar.ax.tick_params(labelsize=8)
    cax.text(0, np.max(data_full[axes.shape[0] - 1]) + 0.05, r"$\langle \hat{n} \rangle$")

    timestamp = datetime.now().strftime("%d.%m.%Y_%H.%M.%S")

    plt.tight_layout()
    plt.savefig(f"Results/Interference-{num_atoms}q-{len([site for t in experiment_config['chosen_sites'] for site in t])}-[{experiment_config['site_exc_id'][0]}-{experiment_config['site_monitor_id_list'][-1]}]-{timestamp}.png", bbox_inches='tight')


  def PlotDoublonDynamics(self, calculations, experiment_config, num_atoms):
    # site_monitor_id_list = experiment_config['site_monitor_id_list'].copy()
    calculations = calculations[0]

    fig, ax = plt.subplots(1, 1, figsize=(16, 9), dpi=200)

    ax.set_ylabel("Population", fontsize=15)
    ax.set_xlabel("t [ns]", fontsize=15)

    for population in calculations:
      ax.plot(experiment_config['operating_time_points'], population)

    # cax = fig.add_axes([1.01, 0.38, 0.02, 0.4])
    # cbar = fig.colorbar(image, cax=cax)
    # cbar.set_ticks([0, np.max(data_full[axes.shape[0] - 1]) / 2, np.max(data_full[axes.shape[0] - 1])])
    # cbar.set_ticklabels(["0", "", f"{np.max(data_full[axes.shape[0] - 1]):.1f}"])
    # cbar.ax.tick_params(labelsize=8)
    # cax.text(0, np.max(data_full[axes.shape[0] - 1]) + 0.05, r"$\langle \hat{n} \rangle$")

    # timestamp = datetime.now().strftime("%d.%m.%Y_%H.%M.%S")

    plt.tight_layout()
    plt.show()


  def PlotLocalization(self, calculations, experiment_config, num_atoms):
    fig, axes = plt.subplots(2, len(calculations), figsize=(len(calculations) * 2, 4), sharey=True, sharex=True, dpi=150)
    axes = np.array(axes).reshape((2, len(calculations)))
    axes[0, 0].set_ylabel("t [ns]")
    axes[1, 0].set_ylabel("t [ns]")
    chain_length = len(experiment_config['site_monitor_id_list'])

    for idx in range(len(calculations)):
      data = np.array(calculations[idx]).T # pcolormesh draws up-down order from bottom to top

      ax = axes[0, idx]
      # image = ax.pcolormesh(data, cmap='bwr', norm=LogNorm(vmin=1e-6, vmax=1)) # PuRd plt.cm.RdBu_r seismic
      image = ax.pcolormesh(data, cmap='bwr')
      if (idx == (len(calculations) - 1)):
        image_last_top = image
      ax.set_xlabel("Site [№]", fontsize=10)
      ax.set_xticklabels([])
      ax.set_yticks([0, experiment_config['operating_time_points'][-1] // 2, experiment_config['operating_time_points'][-1]])
      ax.text(0, experiment_config['operating_time_points'][-1] * 1.05, f"F = {experiment_config['ef_strength_list'][idx]:.2f}J")

      ax = axes[1, idx]
      image = ax.pcolormesh(data, cmap='bwr', norm=LogNorm(vmin=1e-6, vmax=1)) # PuRd plt.cm.RdBu_r seismic
      if (idx == (len(calculations) - 1)):
        image_last_bottom = image
      ax.set_xlabel("Site [№]", fontsize=10)
      ax.set_xticklabels([])
      ax.set_yticks([0, experiment_config['operating_time_points'][-1] // 2, experiment_config['operating_time_points'][-1]])
      ax.text(0, experiment_config['operating_time_points'][-1] * 1.05, f"F = {experiment_config['ef_strength_list'][idx]:.2f}J")

    cax_bottom = fig.add_axes([1.01, 0.11, 0.02, 0.2])
    cbar_bottom = fig.colorbar(image_last_bottom, cax=cax_bottom)
    cax_bottom.text(0, 5, r"$\langle \hat{n} \rangle$")

    cax_top = fig.add_axes([1.01, 0.57, 0.02, 0.2])
    cbar_top = fig.colorbar(image_last_top, cax=cax_top)
    cax_top.text(0, 1.1, r"$\langle \hat{n} \rangle$")

    timestamp = datetime.now().strftime("%Y.%m.%d_%H.%M.%S")

    plt.tight_layout()
    plt.savefig(f"Results/Localization-{num_atoms}q-{len(experiment_config['chosen_sites'])}-{experiment_config['site_exc_id']}_{timestamp}.png", bbox_inches='tight')


  def PlotDeflection(self, calculations, experiment_config):
    if (experiment_config['launch_code'] == 4):
      height = 4
      width = 4
      calculations = np.array(calculations).reshape(len(experiment_config['ef_strength_list']), height, width, -1)

      x_avgs_ef_depend_rotated = []
      y_avgs_ef_depend_rotated = []
      x_time_avgs = np.zeros((len(experiment_config['ef_strength_list']), len(experiment_config['operating_time_points'])))
      y_time_avgs = np.zeros((len(experiment_config['ef_strength_list']), len(experiment_config['operating_time_points'])))

      for ef_id, result in enumerate(calculations):        
        y_avgs_t = []
        x_avgs_t = []

        for t in range(len(experiment_config['operating_time_points'])):
          site_probs_t = result[:, :, t]

          x_probs = site_probs_t.sum(axis=1)
          x_avg = np.sum(x_probs * np.arange(1, 5, 1)) - 1

          y_probs = site_probs_t.sum(axis=0)
          y_avg = np.sum(y_probs * np.arange(1, 5, 1)) - 1

          x_avgs_t.append(x_avg / np.sqrt(2) + y_avg / np.sqrt(2)) # in rotated axes
          y_avgs_t.append(-x_avg / np.sqrt(2) + y_avg / np.sqrt(2)) # in rotated axes
        
        for t in range(len(experiment_config['operating_time_points'])):
          x_time_avgs[ef_id, t] = 1 / (t + 1) * \
                                      np.trapz(x_avgs_t[:(t + 1)], experiment_config['operating_time_points'][:(t + 1)])
          y_time_avgs[ef_id, t] = 1 / (t + 1) * \
                                      np.trapz(y_avgs_t[:(t + 1)], experiment_config['operating_time_points'][:(t + 1)])
        
        x_avgs_ef_depend_rotated.append(x_avgs_t)
        y_avgs_ef_depend_rotated.append(y_avgs_t)

      fig, axes = plt.subplots(2, 3, figsize=(16, 9), dpi=200)

      cmap = plt.get_cmap("coolwarm")
      colors = [cmap((ef_strength - experiment_config['ef_strength_list'].min()) / (experiment_config['ef_strength_list'].max() - experiment_config['ef_strength_list'].min())) for ef_strength in experiment_config['ef_strength_list']]

      ax = axes[0][0]
      for i, color in enumerate(colors):
        ax.plot(x_time_avgs[i], y_time_avgs[i], color=color, label=f"{experiment_config['ef_strength_list'][i]:.2f}")
      ax.grid(ls=':')
      ax.set_xlabel(r"$\langle\overline{x}\rangle$", fontsize=15)
      ax.set_ylabel(r"$\langle \overline{y} \rangle$", fontsize=15)

      ax = axes[0][1]
      for i, color in enumerate(colors):
        ax.plot(experiment_config['operating_time_points'], x_time_avgs[i], color=color, label=f"{experiment_config['ef_strength_list'][i]:.2f}")
      ax.grid(ls=':')
      ax.set_xlabel("t [ns]", fontsize=15)
      ax.set_ylabel(r"$\langle \overline{x} \rangle$", fontsize=15)

      ax = axes[0][2]
      for i, color in enumerate(colors):
        ax.plot(experiment_config['operating_time_points'], y_time_avgs[i], color=color, label=f"{experiment_config['ef_strength_list'][i]:.2f}")
      ax.grid(ls=':')
      ax.set_xlabel("t [ns]", fontsize=15)
      ax.set_ylabel(r"$\langle \overline{y} \rangle$", fontsize=15)

      ax = axes[1][0]
      for i, color in enumerate(colors):
        ax.plot(x_avgs_ef_depend_rotated[i], y_avgs_ef_depend_rotated[i], color=color, label=f"{experiment_config['ef_strength_list'][i]:.2f}")
      ax.grid(ls=':')
      ax.set_xlabel(r"$\langle x \rangle$", fontsize=15)
      ax.set_ylabel(r"$\langle y \rangle$", fontsize=15)

      ax = axes[1][1]
      for i, color in enumerate(colors):
        ax.plot(experiment_config['operating_time_points'], x_avgs_ef_depend_rotated[i], color=color, label=f"{experiment_config['ef_strength_list'][i]:.2f}")
      ax.grid(ls=':')
      ax.set_xlabel("t [ns]", fontsize=15)
      ax.set_ylabel(r"$\langle x \rangle$", fontsize=15)


      ax = axes[1][2]
      for i, color in enumerate(colors):
        ax.plot(experiment_config['operating_time_points'], y_avgs_ef_depend_rotated[i], color=color, label=f"{experiment_config['ef_strength_list'][i]:.2f}")
      ax.grid(ls=':')
      ax.set_xlabel("t [ns]", fontsize=15)
      ax.set_ylabel(r"$\langle y \rangle$", fontsize=15)

      ax.legend(title="ef_strength", fontsize=15, bbox_to_anchor=(1.05, 0.5), loc="center left", borderaxespad=0)

      plt.tight_layout()
      plt.show()
    
    elif (experiment_config['launch_code'] == 5):
      y_time_avgs = np.zeros((len(experiment_config['ef_strength_list']), len(experiment_config['phi_list'])))

      for ef_id, result_ef_fixed in enumerate(calculations):
        for phi_id, result_ef_phi_fixed in enumerate(result_ef_fixed):
          y_avgs_t = []

          for t in range(len(experiment_config['operating_time_points'])):
            site_probs_t = result_ef_phi_fixed[:, :, t]

            x_probs = site_probs_t.sum(axis=1)
            x_avg = np.sum(x_probs * np.arange(1, 5, 1)) - 1

            y_probs = site_probs_t.sum(axis=0)
            y_avg = np.sum(y_probs * np.arange(1, 5, 1)) - 1

            y_avgs_t.append(-x_avg / np.sqrt(2) + y_avg / np.sqrt(2)) # in rotated axes

          y_time_avgs[ef_id, phi_id] = 1 / len(experiment_config['operating_time_points']) * \
                                       np.trapz(y_avgs_t, experiment_config['operating_time_points'])
          
      hall_resistance = []
      popts = []
      for phi_id in range(len(experiment_config['phi_list'])):
        data = y_time_avgs[:, phi_id]
        p0 = [
          (np.max(data) - np.min(data)) / (experiment_config['ef_strength_list'][-1] - experiment_config['ef_strength_list'][0]),
          0
        ]
        popt, _ = curve_fit(FitLin, experiment_config['ef_strength_list'], data, p0=p0, maxfev=10000)
        hall_resistance.append(popt[0])
        popts.append(popt)

      fig, axes = plt.subplots(1, 3, figsize=(16, 9), dpi=200)

      cmap = plt.get_cmap("coolwarm")
      colors_ef = [cmap((ef_strength - experiment_config['ef_strength_list'].min()) / (experiment_config['ef_strength_list'].max() - experiment_config['ef_strength_list'].min())) for ef_strength in experiment_config['ef_strength_list']]

      ax = axes[0]
      for i, color in enumerate(colors_ef):
        ax.plot(experiment_config['phi_list'], y_time_avgs[i, :], color=color, label=f"{experiment_config['ef_strength_list'][i]:.2f}")
      ax.grid(ls=':')
      ax.set_xlabel("Flux", fontsize=15)
      ax.set_ylabel(r"$\langle \overline{y} \rangle$", fontsize=15)
      ax.legend(title="ef_strength", fontsize=15, bbox_to_anchor=(1.05, 0.5), loc="center left", borderaxespad=0)

      colors_mf = [cmap((phi - experiment_config['phi_list'].min()) / (experiment_config['phi_list'].max() - experiment_config['phi_list'].min())) for phi in experiment_config['phi_list']]
      ax = axes[1]
      for i, color in enumerate(colors_mf):
        ax.scatter(experiment_config['ef_strength_list'], y_time_avgs[:, i], color=color, label=f"{experiment_config['phi_list'][i]:.2f}")
        ax.plot(experiment_config['ef_strength_list'], FitLin(experiment_config['ef_strength_list'], *popts[i]), color=color)
      ax.grid(ls=':')
      ax.set_xlabel("Electric field", fontsize=15)
      ax.set_ylabel(r"$\langle \overline{y} \rangle$", fontsize=15)
      ax.legend(title="Flux", fontsize=15, bbox_to_anchor=(1.05, 0.5), loc="center left", borderaxespad=0)

      colors_mf = [cmap((phi - experiment_config['phi_list'].min()) / (experiment_config['phi_list'].max() - experiment_config['phi_list'].min())) for phi in experiment_config['phi_list']]
      ax = axes[2]
      ax.plot(experiment_config['phi_list'], hall_resistance, color='k')
      ax.grid(ls=':')
      ax.set_xlabel("Flux", fontsize=15)
      ax.set_ylabel(r"$\frac{\Delta\langle \overline{y} \rangle}{\Delta F}$", fontsize=15)
      # ax.legend(title="Flux", fontsize=15, bbox_to_anchor=(1.05, 0.5), loc="center left", borderaxespad=0)

      plt.tight_layout()
      plt.show()
