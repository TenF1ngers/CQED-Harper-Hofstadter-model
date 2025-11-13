import numpy as np
from numpy import pi
from scipy.optimize import curve_fit, root_scalar
import qutip as qt
from scipy import sparse as sp
from joblib import Parallel, delayed
from tools import CreateCompositions, CreateConfigList, FitCos, SolveOmegaForPhi, Omega
from Interpreter import Interpreter

from matplotlib import pyplot as plt
from IPython.display import clear_output

class Processor:

  def __init__(self, config: dict):
    self._M = config['M'] # width of rectangular lattice
    self._N = config['N'] # height of rectangular lattice
    self._num_sites = self._M * self._N # number of atoms in superconducting lattice
    self._sites_columns = config["sites_columns"] # columns of certain base frequency
    self._space_trunc = config['space_trunc'] # atom Hilbert space truncation
    self._total_population = config['total_population'] # total excitations number
    # number of atoms in >= 2nd state
    self._max_simultaneously_above_first_excited = config['max_simultaneously_above_first_excited']

    self._dc_freqs = {} # base frequencies of atoms
    self._Ec = config["Ec"]
    self._Ej1 = config["Ej1"]
    self._Ej2 = config["Ej2"]
    self._d = (self._Ej2 / self._Ej1 - 1) / (self._Ej2 / self._Ej1 + 1)
    for column in self._sites_columns:
      for site in column:
        self._dc_freqs["%d" % site] = config['dc_freqs_base']["%d" % column[0]]
        # self._dc_freqs["%d" % site] = config['dc_freqs_base']["%d" % site]
    self._dc_phis = {} # base flux for each transmon
    self.CalculateFlux()

    self._Js = config['Js'] # upper triangular coupling matrix
    self._J_eff = config['J_eff'] # target hopping rate between nearest neighbors
    
    self._anharmonicity = -self._Ec # on-site interaction
    if (self._anharmonicity != 0):
      print("On-site interaction is on")
    else:
      print("On-site interaction is off")

    self._identity_ops_array = np.array([qt.qeye(self._space_trunc).full()] * self._num_sites)
    self._low_energy_states = {} # {le (low energy) state: idx}
    self.BuildLowEnergyStates()
    self._site_destroy_ops = []
    self.BuildDestroyOps(config['site_destroy_ops_init'])
    self._e_ops = [] # expectation operators (observables)
    self.BuildExpectOperators()

    self._modul_freqs = {}
    for order, column in enumerate(self._sites_columns):
        if (order == 0):
          self._modul_freqs["%d" % column[0]] = 0 * 2 * pi
          continue

        for site in column:
          self._modul_freqs["%d" % site] = self._dc_freqs["%d" % site] - self._dc_freqs["%d" % \
                                           self._sites_columns[order - 1][0]]

    self._modul_ampls = {}
    self.CalibrateModulationAmplitudes(config['modul_ampls_init'])


  def CalculateFlux(self):
    for column in self._sites_columns:
      for site in column:
        omega_01 = self._dc_freqs["%d" % site]

        fun = lambda phi: SolveOmegaForPhi(phi, omega_01, self._Ec, self._Ej1, self._Ej2, self._d)
        solution = root_scalar(fun, bracket=[-0.5, 0])
        self._dc_phis["%d" % site] = solution.root


  # Working in the le subspace for computational efficiency
  def BuildLowEnergyStates(self):
    states = []

    for system_population in range(self._total_population + 1):
      # Generating fock sequences directly
      for le_state in CreateCompositions(self._num_sites, self._space_trunc - 1, system_population,
                                         self._max_simultaneously_above_first_excited):
        states.append(le_state)

    for idx, state in enumerate(states):
      self._low_energy_states[state] = idx
    
    print("Low energy subspace of %d kets has been created" % len(self._low_energy_states))


  # Low energy basis truncation
  def TruncateToLowEnergySubspace(self, operator: np.array, obj_type='matrix'):
    if obj_type == 'matrix':
      coef_matrix = sp.lil_matrix((len(self._low_energy_states), len(self._low_energy_states)),
                                   dtype=np.complex64) # for subspaces of > 1e4 states
      
      for j, basis_jth in enumerate(np.array(tuple(self._low_energy_states.keys()))):
        # tmp_state = operator|basis_jth>
        tmp_state = np.squeeze(np.matmul(operator, basis_jth[..., np.newaxis]),
                               axis=-1) # temporary/intermediate state

        # <basis_ith|tmp_state>
        if (self._space_trunc == 2):
          try:
            i = self._low_energy_states[tuple(map(tuple, tmp_state))]
            coef_matrix[i, j] = 1
          except KeyError:
            pass
        else:
          coef_column = sp.lil_matrix((len(self._low_energy_states), 1),
                                       dtype=np.complex64) # for subspaces of > 1e4 states

          for i, basis_ith in enumerate(self._low_energy_states):
            basis_ith = np.array(basis_ith)

            coef_column[i] = np.prod(np.squeeze(
                              np.matmul(basis_ith[:, np.newaxis, :], tmp_state[..., np.newaxis]),
                              axis=(-1, -2)))
          coef_matrix[:, j] = coef_column

      return qt.Qobj(sp.csr_matrix(coef_matrix))

    elif obj_type == 'vector':
      operator = np.squeeze(operator, axis=-1)
      coef_column = sp.lil_matrix((len(self._low_energy_states), 1),
                                   dtype=np.complex64) # for subspaces of > 1e4 states

      # <basis_ith|operator>
      for i, basis_ith in enumerate(self._low_energy_states):
        basis_ith = np.array(basis_ith)

        coef_column[i] = np.prod(np.squeeze(
                           np.matmul(basis_ith[:, np.newaxis, :], operator[..., np.newaxis]),
                           axis=(-1, -2)))
      
      state = qt.Qobj(sp.csr_matrix(coef_column))
      if (len(np.where(state.full() != 0)[0]) != 0):
        state = state.unit()

      return state


  def BuildDestroyOps(self, site_destroy_ops_init):
    if ((site_destroy_ops_init != None) and (len(self._low_energy_states) == \
                                             site_destroy_ops_init[0].full().shape[0])):
      self._site_destroy_ops = site_destroy_ops_init
    else:
      print("Creating destroy operators...")
      for idx in range(self._num_sites):
        processor_ops_array = self._identity_ops_array.copy()
        processor_ops_array[idx] = qt.destroy(self._space_trunc).full()

        self._site_destroy_ops.append(self.TruncateToLowEnergySubspace(processor_ops_array))

    print("Destroy operators have been created")


  def BuildExpectOperators(self):
    for i in range(self._num_sites):
      self._e_ops.append(self._site_destroy_ops[i].dag() * self._site_destroy_ops[i])


  def CalibrateModulationAmplitudes(self, modul_ampls_init):
    if (modul_ampls_init != None):
      self._modul_ampls = modul_ampls_init
    else:
      print("Calibrating modulation amplitudes...")
      target_ratio = self._J_eff / np.max(self._Js)

      for order, column in enumerate(self._sites_columns):
        if (order == 0):
          self._modul_ampls["%d" % column[0]] = 0 * 2 * pi
          continue

        for site in column:

          idx_neighbors = []
          if ((site // self._N == (site + 1) // self._N) and ((site + 1) in 
                                                              self._sites_columns[order - 1])):
            idx_neighbors.append(site + 1)
          if ((site - self._N >= 0) and ((site - self._N) in self._sites_columns[order - 1])):
            idx_neighbors.append(site - self._N)

          # let's find _modul_ampl[site] using binary search
          Omegas_middle = []

          if len(idx_neighbors) == 0:
            Omegas_middle.append(0)

          for neighbor in idx_neighbors:
            Omega_left = 1e-1
            Omega_right = 2 * np.abs(self._modul_freqs["%d" % site])
            Omega_middle = (Omega_left + Omega_right) / 2

            time_limit = int(2 * pi / np.max(self._Js))

            while True:
              experiment_config_base = {
              'launch_code': 6,
              
              'operating_time_points': np.linspace(0, time_limit, time_limit + 1),
              'phi_list': None, 'ef_strength_list': None,

              'chosen_sites': [(neighbor,), (site,)],
              'site_exc_id': [site], 'site_fock_number': [1], 'site_modul_phi_id': None, 
              'site_monitor_id_list': [[neighbor]], 'site_monitor_fock_numbers': [[1]],

              'dc_freqs': {},
              'modul_ampls': {("%d" % neighbor): self._modul_ampls["%d" % neighbor],
                              ("%d" % site): Omega_middle},
              'modul_freqs': {},
              'modul_phases': {},

              'drive_ampl_factor': None,
              'drive_ampls': {},
              'drive_freqs': {},
              'drive_phases': {}}
              experiment_config_list = CreateConfigList(self, experiment_config_base)

              result = self.LaunchExperiment(experiment_config_list)

              p0 = [-np.max(result) / 2,
                    2 * pi / (np.where(result > (np.max(result[:time_limit]) - 0.005))[0][0] * 2),
                    0,
                    np.max(result) / 2]
              popt, _ = curve_fit(FitCos, experiment_config_base['operating_time_points'], result,
                                  p0=p0, maxfev=10000)

              J_eff_ratio = popt[1] / np.max(self._Js) / 2

              if (np.abs(J_eff_ratio - target_ratio) < 1e-5):
                Omegas_middle.append(Omega_middle)
                print(f"Effective coupling of site №{site} with neighbor №{neighbor} is \
                       {J_eff_ratio * np.max(self._Js) / 2 / pi * 1e3:.3f} MHz")
                break

              if (J_eff_ratio > target_ratio):
                Omega_right = Omega_middle
              else:
                Omega_left = Omega_middle
              Omega_middle = (Omega_left + Omega_right) / 2
          
          self._modul_ampls["%d" % site] = np.mean(np.array(Omegas_middle))

    print("Calibration has been done")


  def MakeEnergyProfiles(self, experiment_config: dict):
    energy_profiles = []

    dc_freqs = experiment_config['dc_freqs'].copy()
    modul_ampls = experiment_config['modul_ampls'].copy()
    modul_freqs = experiment_config['modul_freqs'].copy()
    modul_phases = experiment_config['modul_phases'].copy()

    for i in range(self._num_sites):
      energy_profiles.append("%f - %f * np.cos(%f * t + %f)" % \
                              (dc_freqs["%d" % i], modul_ampls["%d" % i],
                              modul_freqs["%d" % i], modul_phases["%d" % i]))

    return energy_profiles


  def BuildHamiltonianBoseHubbard(self, phi_list):
    H_device = []

    for i in range(self._num_sites):
      H_device += \
      [self._site_destroy_ops[i].dag() * self._site_destroy_ops[i] * \
       Omega(phi_list["%d" % i], self._Ej1, self._Ej2, self._d, self._Ec),
      self._anharmonicity / 2 * self._site_destroy_ops[i].dag() * self._site_destroy_ops[i].dag() * \
      self._site_destroy_ops[i] * self._site_destroy_ops[i]] # anharmonicity is either zero or not

      for j in range(i + 1, self._num_sites):
        H_device += \
        [self._Js[i, j] * (self._site_destroy_ops[i].dag() * self._site_destroy_ops[j] + \
        self._site_destroy_ops[j].dag() * self._site_destroy_ops[i])]

    return sum(H_device)


  def BuildHamiltonianEffectiveHH(self, experiment_config: dict):
    H_device = []
    energy_profiles = self.MakeEnergyProfiles(experiment_config)
    
    for i in range(self._num_sites):
      H_device += \
      [[self._site_destroy_ops[i].dag() * self._site_destroy_ops[i], energy_profiles[i]],
      self._anharmonicity / 2 * self._site_destroy_ops[i].dag() * self._site_destroy_ops[i].dag() * \
      self._site_destroy_ops[i] * self._site_destroy_ops[i]] # anharmonicity is either zero or not

      for j in range(i + 1, self._num_sites):
        H_device += \
        [self._Js[i, j] * (self._site_destroy_ops[i].dag() * self._site_destroy_ops[j] + \
        self._site_destroy_ops[j].dag() * self._site_destroy_ops[i])]

    return H_device


  # Task for each core working in parallel with others
  def RunForConfig(self, experiment_config):

    # Preparation: building initial state & required e_ops
    # -------------------------------------------------------------------------------
    states_array = np.array([qt.basis(self._space_trunc, 0).full()] * self._num_sites)
    for site_idx, site in enumerate(experiment_config['site_exc_id']):
      states_array[site] = qt.basis(self._space_trunc,
                                    experiment_config['site_fock_number'][site_idx]).full()
    psi0 = self.TruncateToLowEnergySubspace(states_array, obj_type='vector')

    if (experiment_config['launch_code'] in (1, 3, 6, 7)):
      e_ops = []

      for array_idx, site_array in enumerate(experiment_config['site_monitor_id_list']):
        states_array = np.array([qt.basis(self._space_trunc, 0).full()] * self._num_sites)

        for site_idx, site in enumerate(site_array):
          states_array[site] = \
          qt.basis(self._space_trunc,
                   experiment_config['site_monitor_fock_numbers'][array_idx][site_idx]).full()

        psi_target = self.TruncateToLowEnergySubspace(states_array, obj_type='vector')

        e_ops.append(psi_target * psi_target.dag())
    else:
      e_ops = [self._e_ops[idx] for idx in experiment_config['site_monitor_id_list']]
    # -------------------------------------------------------------------------------

    H_device = self.BuildHamiltonianEffectiveHH(experiment_config)

    result = qt.mesolve(H_device, psi0, experiment_config['operating_time_points'],
                        c_ops=[], e_ops=e_ops)
    return result.expect


  def LaunchExperiment(self, experiment_config_list):
    interpreter = Interpreter()

    if (experiment_config_list[0]['launch_code'] == 1): # Aharonov-Bohm interference
      calculations = Parallel(n_jobs=7, verbose=10)(
          delayed(self.RunForConfig)(experiment_config) for experiment_config in \
                                                            experiment_config_list
      )
      interpreter.PlotInterference(calculations, experiment_config_list[0], self._num_sites)

    elif (experiment_config_list[0]['launch_code'] == 2): # Wannier-Stark localization
      calculations = Parallel(n_jobs=7, verbose=0)(
          delayed(self.RunForConfig)(experiment_config) for experiment_config in \
                                                            experiment_config_list
      )
      interpreter.PlotLocalization(calculations, experiment_config_list[0], self._num_sites)

    elif (experiment_config_list[0]['launch_code'] == 4): # Particle dynamics in em-field

      calculations = Parallel(n_jobs=7, verbose=5)(
        delayed(self.RunForConfig)(experiment_config) for experiment_config in \
                                                          experiment_config_list
      )
      interpreter.PlotDeflection(calculations, experiment_config_list[0])

    elif (experiment_config_list[0]['launch_code'] == 5): # Hall coefficients
      experiment_config_list = np.array(experiment_config_list).reshape(
            (len(experiment_config_list[0]['ef_strength_list']),
             len(experiment_config_list[0]['phi_list'])
          )
        )
      calculations = []

      for experiment_config_list_ef_fixed in experiment_config_list:
        calculations.append(Parallel(n_jobs=7, verbose=1)(
          delayed(self.RunForConfig)(experiment_config) for experiment_config in \
                                                            experiment_config_list_ef_fixed
      ))
      calculations = np.array(calculations).reshape(
          (len(experiment_config_list[0][0]['ef_strength_list']),
           len(experiment_config_list[0][0]['phi_list']), 4, 4, -1
         )
        )
      interpreter.PlotDeflection(calculations, experiment_config_list[0][0])

    elif (experiment_config_list[0]['launch_code'] == 6): # Calibrating modulation amplitudes
      calculations = Parallel(n_jobs=7, verbose=0)(
        delayed(self.RunForConfig)(experiment_config) for experiment_config in \
                                                          experiment_config_list
      )
      return np.array(calculations).ravel()
