# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 21:42:23 2024

@author: cca79
"""

import sys
from os import environ, makedirs
environ["QT_API"] = "pyside6"

from os.path import splitext, basename
import logging
from _utils import round_sf
from solvers import eulermaruyama
import numpy as np
from importlib.util import spec_from_file_location, module_from_spec

from qtpy.QtWidgets import QApplication, QMainWindow, QLabel, QLineEdit, QGridLayout, QWidget, QFileDialog, QGroupBox, QErrorMessage
from qtpy.QtGui import QAction, QActionGroup
from qtpy import QtCore
from gui.resources.qtdesigner.QT_simGUI import Ui_MainWindow


class ODE_Explorer_Controller(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(ODE_Explorer_Controller, self).__init__()
        self.init_logging()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.precision = np.float64
        self.init_precision_action_group()
        self.diffeq_system_file_path = None

        logging.info("ODE_Explorer_Controller initialized")

        # self.plot_state = {
        #     'param1': None,
        #     'param2': None,
        #     'current_fft_freq': None,
        #     'single_plot_style': None,
        #     'x_var': None,
        #     'y_var': None,
        #     'z_var': None,
        #     'x_slice': None,
        #     'y_slice': None,
        #     'z_slice': None,
        #     'x_scale': None,
        #     'y_scale': None,
        #     'z_scale': None,
        # }

        self.sim_state = {
            'dt': None,
            'duration': None,
            'fs': None,
            'param1_sweep_bounds': None,
            'param1_sweep_scale': None,
            'param1_num_values': 100,
            'param1_var': None,
            'param1_values': [],
            'param2_sweep_bounds': None,
            'param2_sweep_scale': None,
            'param2_num_values': 100,
            'param2_var': None,
            'param2_values': [],
            'warmup': None,

            'y0': None
        }
        self.param_index_map = {}
        self.display_sig_figs = 4
    def init_logging(self):
        # Create logs directory if it doesn't exist
        makedirs("logs", exist_ok=True)

        # Configure the logger
        logging.basicConfig(
            filename='logs/GUIlog.log',
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(funcName)s - %(message)s',
            datefmt='%m/%d/%Y %I:%M:%S %p'
        )

        logging.debug('Logfile initialised')

    def init_precision_action_group(self):
        """Group 32-64 bit precision buttons as QTDesigner has dropped support for
        doing this inside the GUI"""

        self.action_64bit = self.findChild(QAction, "action64_bit_2")
        self.action_32bit = self.findChild(QAction, "action32_bit_2")

        self.precision_group = QActionGroup(self)
        self.precision_group.addAction(self.action_64bit)
        self.precision_group.addAction(self.action_32bit)
        self.precision_group.setExclusive(True)

        self.action_64bit.triggered.connect(lambda: self.set_precision(self.action_64bit))
        self.action_32bit.triggered.connect(lambda: self.set_precision(self.action_32bit))

        def set_precision(self, action):
            if action.text() == '64-bit':
                self.precision = np.float64
            elif action.text() == '32-bit':
                self.precision = np.float32
            print(f"Precision set to {self.precision}")

    def load_system_from_filedialog(self):
        """Open file dialog to select diffeq_system file, instantiate and start solver"""

        self.diffeq_system_file_path, _ = QFileDialog.getOpenFileName(self, "Select Python File", "", "Python Files (*.py);;All Files (*)")
        if self.diffeq_system_file_path:
            system = self.get_system_from_file(self.diffeq_system_file_path)
        else:
            logging.warning("diffeq system file path not valid")

        self.load_system(system)


    def get_system_from_file(self, file_path):
        module_name = splitext(basename(file_path))[0]
        spec = spec_from_file_location(module_name, file_path)
        module = module_from_spec(spec)
        spec.loader.exec_module(module)

        if hasattr(module, 'diffeq_system'):
            system = getattr(module, 'diffeq_system')(precision = self.precision)
            return system
        else:
            logging.warning("The module you selected has no diffeq_system class")
            raise AttributeError("The module you selected has no diffeq_system class")


    def load_system(self, system):
        """ Start a new solver instance, and load a provided ODE system into it.
        Populate the system paramaters box, the inits and noise sigmas fields,
        fill the param select lists, load default sweep if present. """

        self.solver = eulermaruyama.Solver(self.precision)
        self.solver.load_system(system)
        self.ui.simController.load_system(self.solver.system.constants_dict, self.solver.system.noise_sigmas)
        self.ui.plotController.load_state_labels(self.solver.system.state_labels)
        # self.fill_paramSelect_lists()


    def reload_system(self):
        """ Assumes we have already located the ODE_system file, and are just
        reloading to either change solvers of modify precision"""
        system = self.get_system_from_file(self.diffeq_file_path)
        self.load_system(system)



    def save_results(self):
        pass



    # def update_selected_params(self):
    #     p1 = self.plot_state['param1_val']
    #     p2 = self.plot_state['param2_val']
    #     self.solver.param_index = self.solver.get_param_index((p1, p2))
    #     self.single_state = self.solver.solved_ODE.output_array[:, self.solver.param_index, :]


    def set_duration(self):
        duration  = self.ui.duration_e.value()
        duration = self.check_type(duration, self.precision)
        if duration:
            self.sim_state['duration'] = duration

    def set_fs(self):
        fs = self.ui.fs_e.value()
        fs = self.check_type(fs, self.precision)
        if fs:
            self.sim_state['fs'] = self.ui.fs_e.value()

    def set_dt(self):
        dt = self.ui.stepsize_e.value()
        dt = self.check_type(dt, self.precision)
        if dt:
            self.sim_state['fs'] = dt

    def set_warmup(self):
        warmup = self.ui.warmup_e.value()
        warmup = self.check_type(warmup, self.precision)
        if warmup:
            self.sim_state['fs'] = warmup

    def set_param1Sweep_bounds(self):
        lower_bound = self.ui.p1SweepLower_entry.value()
        upper_bound = self.ui.p1SweepUpper_entry.value()
        self.sim_state['param1_sweep_bounds'] = (lower_bound, upper_bound)

    def set_param1Sweep_scale(self, index):
        if index == 1:
            scale = 'log'
        else:
            scale = 'lin'
        self.sim_state['param1_sweep_scale'] = scale

    def set_param1_var(self):
        p1var = self.ui.p1Select_dd.value()
        p1var = self.check_type(p1var, str)
        if p1var:
            self.sim_state['param1_var'] = p1var

    def set_param2Sweep_bounds(self):
        lower_bound = self.ui.p2SweepLower_entry.value()
        upper_bound = self.ui.p2SweepUpper_entry.value()
        self.sim_state['param2_sweep_bounds'] = (lower_bound, upper_bound)

    def set_param2Sweep_scale(self, index):
        if index == 1:
            scale = 'log'
        else:
            scale = 'lin'
        self.sim_state['param1_sweep_scale'] = scale

    def set_param2_var(self):
        p2var = self.ui.p2Select_dd.value()
        p2var = self.check_type(p2var, str)
        if p2var:
            self.sim_state['param2_var'] = p2var

    def build_sweeps(self):
        param1_bounds = self.sim_state['param1_sweep_bounds']
        param1_scale = self.sim_state['param1_sweep_scale']
        param1_num_values = self.sim_state['param1_num_values']

        param2_bounds = self.sim_state['param2_sweep_bounds']
        param2_scale = self.sim_state['param2_sweep_scale']
        param2_num_values = self.sim_state['param2_num_values']

        if param1_bounds and param1_scale and param1_num_values:
            if param1_scale == 'lin':
                self.sim_state['param1_values'] = np.linspace(param1_bounds[0],
                                                              param1_bounds[1],
                                                              param1_num_values,
                                                              dtype=self.precision)
            elif param1_scale == 'log':
                try:
                    self.sim_state['param1_values'] = np.logspace(np.log10(param1_bounds[0]),
                                                                  np.log10(param1_bounds[1]),
                                                                  param1_num_values,
                                                                  dtype=self.precision)
                except:
                    logging.warning(f"P1 bounds: {param1_bounds} unable to create a log space")
                    self.displayerror("Param 1 values are unloggable, try a linear sweep or change values")

        if param2_bounds and param2_scale and param2_num_values:
            if param2_scale == 'lin':
                self.sim_state['param2_values'] = np.linspace(param2_bounds[0],
                                                              param2_bounds[1],
                                                              param2_num_values,
                                                              dtype=self.precision)
            elif param2_scale == 'log':
                try:
                    self.sim_state['param2_values'] = np.logspace(np.log10(param2_bounds[0]),
                                                                  np.log10(param2_bounds[1]),
                                                                  param2_num_values,
                                                                  dtype=self.precision)
                except:
                    logging.warning(f"P1 bounds: {param1_bounds} unable to create a log space")
                    self.displayerror("Param 1 values are unloggable, try a linear sweep or change values")

    def generate_index_map(self):
        """Set up a dict mapping parameter sets to their index in the output
        array, cutting some compute time when populating the z mesh for plotting.

        Saves results to self.param_index_map.
        """
        self.param_index_map = {}
        grid_values = [(p1, p2) for p1 in self.sim_state['param1_values'] for p2 in self.sim_state['param2_values']]
        for idx, (p1, p2) in enumerate(grid_values):
            p1_round = self.precision(round_sf(p1, self.display_sig_figs))
            p2_round = self.precision(round_sf(p2, self.display_sig_figs))
            self.param_index_map[(p1_round, p2_round)] = idx



    def set_y0(self):
        inits = []
        for i in range(self.y0_box.count()):
            sigma = self.y0_box.itemAt(i).widget().value()
            sigma = self.check_type(sigma, self.precision)
            if sigma is not None: #Nonecheck as val could be 0
                inits.append(sigma)
        self.sim_state['inits'] = inits

    def set_noise_sigmas(self):
        noise_sigmas = []
        for i in range(self.noiseSigmas_box.count()):
            sigma = self.noiseSigmas_box.itemAt(i).widget().value()
            sigma = self.check_type(sigma, self.precision)
            if sigma is not None:
                noise_sigmas.append(sigma)
        self.sim_state['noise_sigmas'] = noise_sigmas

    def solve_ODE(self):
        self.build_sweeps()

        # Update solver with simulation parameters
        self.solver.duration = self.sim_state['duration']
        self.solver.fs = self.sim_state['fs']
        self.solver.step_size = self.sim_state['dt']
        self.solver.warmup_time = self.sim_state['warmup']
        self.t = np.linspace(0,  self.duration -  1/self.fs, int( self.duration * self.fs))


    def update_plot(self):
        pass


    def closeEvent(self, event):
        self
        print("Goodbye.")


    def displayerror(self, message):
        error_dialog = QErrorMessage()
        logging.warning(message)
        error_dialog.showMessage(message)

    def check_type(self, argument, dtype):
        try:
            # If the dtype is a numpy array type, attempt to cast the entire array
            if isinstance(argument, (list, np.ndarray)):
                argument = np.array(argument, dtype=dtype)
            else:
                argument = dtype(argument)
        except (ValueError, TypeError):
            self.display_error(f"The argument {argument} cannot be interpreted as {dtype}")
            return None
        return argument



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ODE_Explorer_Controller()
    window.show()
    sys.exit(app.exec())
