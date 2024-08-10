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
from solvers.eulermaruyama import Solver
from gui.modelController import modelController
import numpy as np
from importlib.util import spec_from_file_location, module_from_spec

from datetime import datetime
from qtpy.QtWidgets import QApplication, QMainWindow, QLabel, QLineEdit, QGridLayout, QWidget, QFileDialog, QGroupBox, QErrorMessage
from qtpy.QtGui import QAction, QActionGroup
from qtpy import QtCore
from gui.resources.qtdesigner.QT_simGUI import Ui_MainWindow


class ODE_Explorer_Controller(QMainWindow, Ui_MainWindow):
    variable_label_keys = {'PSD at selected frequency': 'psd',
                           'PSD magnitude': 'psd',
                           'FFT phase at selected frequency': 'phase',
                           'FFT phase': 'phase',
                           'RMS amplitude': 'rms',
                           'Signal amplitude': 'ampl'
                           }

    def __init__(self):
        super(ODE_Explorer_Controller, self).__init__()
        self.init_logging()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.set_starting_state()
        self.precision = np.float64
        self.init_precision_action_group()
        self.diffeq_system_file_path = None

        logging.info("ODE_Explorer_Controller initialized")

        self.param_index_map = {}
        self.grid_list = np.zeros(1, dtype=self.precision)
        self.display_sig_figs = 4

        self.model = modelController()
        self.model.load_solver(Solver)


    def set_starting_state(self):
        self.ui.controlToolBox.setCurrentIndex(1)
        self.ui.plotController.plotSettingsTabs.setCurrentIndex(0)
        self.ui.simController.simSettingsTabs.setCurrentIndex(0)

    def init_logging(self):
        # Create logs directory if it doesn't exist
        makedirs("logs", exist_ok=True)

        # Configure the logger
        logging.basicConfig(
            filename = f"logs/log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
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
        self.model.load_system(system)
        self.ui.simController.load_system(self.model.system.constants_dict, self.model.system.noise_sigmas)
        self.ui.plotController.load_state_labels(self.model.system.state_labels)

    def load_solver(self, solver):
        self.model.load_solver(solver, precision=self.precision)

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
    #     self.model.param_index = self.model.get_param_index((p1, p2))
    #     self.single_state = self.model.solved_ODE.output_array[:, self.model.param_index, :]


    def fetch_independent_variables(self):
        p1view, p2view = self.ui.simController.get_parameter_sweeps()

        independent_variables = {"t": self.ui.simController.get_time_vector(),
                                "psd_freq": self.ui.simController.get_psd_freq_vector(),
                                "fft_freq": self.ui.simController.get_fft_freq_vector(),
                                "param1": p1view,
                                "param2": p2view,
                                "sweep_labels": self.ui.simController.get_sweep_labels()}

        return independent_variables

    def pass_independent_variables_to_plotter(self):
        variables = self.fetch_independent_variables()
        self.ui.plotController.set_independent_variables(variables)

    def generate_grid_list_and_map(self, param1_sweep, param2_sweep):
        """Generate 1D list of all requested parameter combinations.
        Save a rounded version of each index into a dict for easy lookup
        to match with a dataset request from the plotter.

        """
        self.param_index_map = {}
        grid_list = [(p1, p2) for p1 in param1_sweep for p2 in param2_sweep]
        for idx, (p1, p2) in enumerate(grid_list):
            p1_round = self.precision(round_sf(p1, self.display_sig_figs))
            p2_round = self.precision(round_sf(p2, self.display_sig_figs))
            self.param_index_map[(p1_round, p2_round)] = idx
        self.grid_list = grid_list

    def update_plotController_sweeps(self):
        self.ui.plotController.populate_swept_parameter_values(self.param1_values, self.param2_values)
        self.ui.plotController.update_fixed_sliders('param2')

    def on_solve_complete(self):
        self.update_plotController_sweeps()
        self.pass_independent_variables_to_plotter()

    def on_solve_request(self):
        p1view, p2view = self.ui.simController.get_parameter_sweeps()
        self.generate_grid_list_and_map(p1view, p2view)

        params = self.ui.simController.get_sysparams()
        self.model.update_system_parameters(params)

        self.solve_ODE()

    def solve_ODE(self):

        sweeps = {'values': self.grid_list,
                  'vars': self.sweep_labels}
        sim_state = self.ui.simController.get_sim_state()
        self.model.run_solver(self.model.system,
                              sim_state, sweeps)
        self.on_solve_complete()

    def fetch_data(self, variable, state, bounds):
        """where you're up to:
            model.fetch_data takes a data type (psd, phase, amplitude, rms), a
            state index, and either None, int, or a list of int indices for
            extracting desired freqs, times, grid values. This should work for
            time-domain and state-space variables (if the data type variable is
           in the state labels dict, it reutrns that "state" for the grid indices chosen.
           As all the grid-level mapping is 1d in the solver implementation, all outputs are 1d).

            Next step is to filter by variable name in the drop-downs. These are named shittily.
            This should probably be done inside the plot controller, and then it can send a datatype/indices combo.
            Note: the plot controller is naiive to grid indices, these will need to be passed as param values then
            mapped to list indieces in GUI."""

        pass

    def update_plot(self):
        pass

    def closeEvent(self, event):
        self.ui.Plotwidget.closeEvent(event)
        self.close()
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