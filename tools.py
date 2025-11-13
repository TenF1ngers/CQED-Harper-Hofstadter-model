import numpy as np
from numpy import pi
import qutip as qt
import copy as cp

def FitCos(x, a, b, c, d):
  return a * np.cos(b * x + c) + d

def FitLin(x, a, b):
  return a * x + b

# Josephson energy depending on flux through SQUID
def Ej_phi(phi, Ej1, Ej2, d):
  return (Ej1 + Ej2) * np.sqrt(np.cos(pi * phi)**2 + d**2 * np.sin(pi * phi)**2)

# Omega_01 of the transmon depenging on flux through SQUID
def Omega(phi, Ej1, Ej2, d, Ec):
  return np.sqrt(8 * Ec * Ej_phi(phi, Ej1, Ej2, d)) - Ec

def SolveOmegaForPhi(phi, omega_01, Ec, Ej1, Ej2, d):
      return (8 * Ec * (Ej1 + Ej2)) * np.sqrt(np.cos(pi * phi)**2 + d**2 * np.sin(pi * phi)**2) - (omega_01 + Ec)**2

# Recurrent function for building le states directly for time efficiency
def CreateCompositions(length, site_capacity, system_population,
                       max_simultaneously_above_first_excited):
  if length == 1:
    if 0 <= system_population <= site_capacity:
      yield tuple(map(tuple, qt.basis(site_capacity + 1, system_population).full().T))
    return
  
  # Choose available occupancy at each entry into the function
  for chosen_value in np.arange(0, min(system_population, site_capacity) + 1):
    if ((chosen_value > 1) and (max_simultaneously_above_first_excited != 0)):
      for tail in CreateCompositions(length - 1, site_capacity, system_population - chosen_value,
                                     max_simultaneously_above_first_excited - 1):
        yield tuple(map(tuple, qt.basis(site_capacity + 1, chosen_value).full().T)) + tail
    
    elif (chosen_value <= 1):
      for tail in CreateCompositions(length - 1, site_capacity, system_population - chosen_value,
                                     max_simultaneously_above_first_excited):
        yield tuple(map(tuple, qt.basis(site_capacity + 1, chosen_value).full().T)) + tail
    
    else:
      return


# coupling matrix for num_sites = M x N
def CreateCouplingMatrix(M, N, J0_nearest, J0_next_nearest, next_nearest=False):
  num_sites = M * N
  Js = np.zeros((num_sites, num_sites))

  for i in range(0, num_sites - N):
    Js[i, i + N] = J0_nearest
  for i in range(0, N - 1):
    for j in range(i, i + num_sites - N + 1, N):
      Js[j, j + 1] = J0_nearest

  if next_nearest:
    for idx in range(0, num_sites - 3 * N + 1, N):
      Js[idx, idx + 2] = Js[idx, idx + 5] = Js[idx, idx + 8] = J0_next_nearest
      Js[idx + N - 1, (idx + N - 1) + 8] = Js[idx + N - 1, (idx + N - 1) + 3] = J0_next_nearest

      for j in range(1, N - 1):
        Js[idx + j, (idx + j) + 2] = Js[idx + j, (idx + j) + 5] = \
        Js[idx + j, (idx + j) + 8] = Js[idx + j, (idx + j) + 3] = J0_next_nearest

    idx = num_sites - N * 2
    Js[idx, idx + 2] = Js[idx, idx + 5] = J0_next_nearest
    Js[idx + 1, (idx + 1) + 2] = Js[idx + 1, (idx + 1) + 5] = \
                                 Js[idx + 1, (idx + 1) + 3] = J0_next_nearest
    Js[idx + 2, (idx + 2) + 5] = Js[idx + 2, (idx + 2) + 3] = J0_next_nearest
    Js[idx + 3, (idx + 3) + 3] = J0_next_nearest

    idx = num_sites - N * 1
    Js[idx, idx + 2] = J0_next_nearest
    Js[idx + 1, (idx + 1) + 2] = J0_next_nearest

  return Js


def CreateConfigList(processor, experiment_config_base):
  chosen_sites = experiment_config_base['chosen_sites']
  other_sites = [idx for idx in range(processor._num_sites) if idx not in \
                                                       [site for t in chosen_sites for site in t]]

  # Standard parametric modulation configuration
  # -------------------------------------------------------------------------------
  for site in [site for t in chosen_sites for site in t]:
    experiment_config_base['dc_freqs']["%d" % site] = processor._dc_freqs["%d" % site]
    experiment_config_base['modul_freqs']["%d" % site] = processor._modul_freqs["%d" % site]
    experiment_config_base['modul_phases']["%d" % site] = 0

  for site in other_sites:
    experiment_config_base['dc_freqs']["%d" % site] = 4.1 * 2 * pi
    experiment_config_base['modul_ampls']["%d" % site] = 0 * 2 * pi
    experiment_config_base['modul_freqs']["%d" % site] = 0 * 2 * pi
    experiment_config_base['modul_phases']["%d" % site] = 0

    experiment_config_base['drive_ampls']["%d" % site] = 0
    experiment_config_base['drive_freqs']["%d" % site] = 0
    experiment_config_base['drive_phases']["%d" % site] = 0
  # -------------------------------------------------------------------------------

  # Aharonov-Bohm interference, Interaction-induced delocalization
  # -------------------------------------------------------------------------------
  if (experiment_config_base['launch_code'] in (1, 3)):
    experiment_config_list = []

    for phi_idx in range(len(experiment_config_base['phi_list'][0])):
      tmp_config = cp.deepcopy(experiment_config_base)
      
      for site_idx_modul in range(len(tmp_config['site_modul_id'])):
        tmp_config['modul_phases']["%d" % tmp_config['site_modul_id'][site_idx_modul]] = \
                                       tmp_config['phi_list'][site_idx_modul][phi_idx]

      experiment_config_list.append(tmp_config)
    
    return experiment_config_list
  # -------------------------------------------------------------------------------

  # Inducing linear potential like behaviour for Wannier-Stark localization
  # -------------------------------------------------------------------------------
  elif (experiment_config_base['launch_code'] == 2):
    grad = processor._Js[0, 1] * (1 + 1 / 10) / 2
    experiment_config_list = []

    for ef_strength in experiment_config_base['ef_strength_list']:
      tmp_config = cp.deepcopy(experiment_config_base)

      for site in np.array([site for t in chosen_sites for site in t]):
        tmp_config['modul_freqs']["%d" % site] += ef_strength * grad
      experiment_config_list.append(tmp_config)
    
    return experiment_config_list
  # -------------------------------------------------------------------------------

  # Particle dynamics in em-field
  # -------------------------------------------------------------------------------
  elif (experiment_config_base['launch_code'] == 4):
    grad = processor._Js[0, 1] * (1 + 1 / 10) / 2
    phi = experiment_config_base['phi_list'][0]
    phi_dict = {'3': 0., '2': phi, '7': -phi,
                '1': 2 * phi, '6': 0., '11': -2 * phi,
                '0': 3 * phi, '5': phi, '10': -phi,
                '15': -3 * phi, '4': 2 * phi, '9': 0,
                '14': -2 * phi, '8': phi, '13': -phi,
                '12': 0.}

    experiment_config_list = []

    for site in [site for t in chosen_sites for site in t]:
        experiment_config_base['modul_phases']["%d" % site] = phi_dict["%d" % site]

    for ef_strength in experiment_config_base['ef_strength_list']:
      tmp_config = cp.deepcopy(experiment_config_base)

      for site in [site for t in chosen_sites for site in t]:
        # just plus cuz direction of e-field due to right-to-left parametric coupling
        tmp_config['modul_freqs']["%d" % site] += ef_strength * grad
      
      experiment_config_list.append(tmp_config)

    return experiment_config_list
  # -------------------------------------------------------------------------------

  # Hall coefficients
  # -------------------------------------------------------------------------------
  elif (experiment_config_base['launch_code'] == 5):
    grad = processor._Js[0, 1] * (1 + 1 / 10) / 2
    experiment_config_list = []

    for ef_strength in experiment_config_base['ef_strength_list']:
      for phi in experiment_config_base['phi_list']:
        tmp_config = cp.deepcopy(experiment_config_base)
        phi_dict = {'3': 0., '2': phi, '7': -phi,
                '1': 2 * phi, '6': 0., '11': -2 * phi,
                '0': 3 * phi, '5': phi, '10': -phi,
                '15': -3 * phi, '4': 2 * phi, '9': 0,
                '14': -2 * phi, '8': phi, '13': -phi,
                '12': 0.}

        for site in [site for t in chosen_sites for site in t]:
          tmp_config['modul_phases']["%d" % site] = phi_dict["%d" % site]
          # just plus cuz direction of e-field due to right-to-left parametric coupling
          # & sign of Peierls phases does not depend on sign of the modul_freq
          tmp_config['modul_freqs']["%d" % site] += ef_strength * grad
        experiment_config_list.append(tmp_config)

    return experiment_config_list
  # -------------------------------------------------------------------------------
  
  return [experiment_config_base] # launch_code == 6


def SetUpExperiment(processor, experiment_config_base):
  for site in [site for t in experiment_config_base['chosen_sites'] for site in t]:
    experiment_config_base['modul_ampls']["%d" % site] = processor._modul_ampls["%d" % site]

  return CreateConfigList(processor, experiment_config_base)
