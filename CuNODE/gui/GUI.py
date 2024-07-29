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
from gui.modelController import modelController
import numpy as np
from importlib.util import spec_from_file_location, module_from_spec

from datetime import datetime
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
        self.set_starting_state()
        self.precision = np.float64
        self.init_precision_action_group()
        self.diffeq_system_file_path = None

        logging.info("ODE_Explorer_Controller initialized")

        self.param_index_map = {}
        self.grid_list = np.zeros(1, dtype=self.precision)
        self.display_sig_figs = 4

        self.model = modelController()
        self.model.loadController(eulermaruyama)

    def set_starting_state(self):
        self.ui.controlToolBox.setCurrentIndex(1)
        self.ui.plotController.plotSettingsTabs.setCurrentIndex(0)
        self.ui.simController.simSettingsTabs.setCurrentIndex(0)

    def init_logging(self):
        # Create logs directory if it doesn't exist
        makedirs("logs", exist_ok=True)

        # Configure the logger
        logging.basicConfig(
            filename = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
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
        #TODO: Load model controller instead.
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
    def generate_sweep(self, bounds, scale, n):
        if scale == 'Linear':
            sweep = np.linspace(bounds[0], bounds[1], n, dtype=self.precision)
        elif scale == 'Logarithmic':
            try:
                sweep = np.logspace(np.log10(bounds[0], bounds[1], n, dtype=self.precision))
            except:
                logging.warning(f"Bounds: {bounds} unable to create a log space")
                self.displayerror("The requested logarithmic sweep contains un-loggable values, change bounds or try a linear sweep.")

    def prepare_parameter_sweeps(self):
        p1_bounds, p1_n, p1_scale = self.ui.simController.get_swept_parameters('param1')
        p2_bounds, p2_n, p2_scale = self.ui.simController.get_swept_parameters('param2')
        param1_values = self.generate_sweep(p1_bounds, p1_n, p1_scale)
        param2_values = self.generate_sweep(p1_bounds, p1_n, p1_scale)

        self.param1_values = param1_values
        self.param2_values = param2_values
        self.generate_grid_list_and_map(param1_values, param2_values)


    def generate_grid_list_and_map(self, param1_sweep, param2_sweep):
        """Generate 1D list of all requested parameter combinations.
        Save a rounded version of each index into a dict for easy lookup
        to match with a dataset request from the plotter.

        """
        self.param_index_map = {}
        grid_list = [(p1, p2) for p1 in param1_sweep for p2 in param2_sweep]
        for idx, (p1, p2) in enumerate(self.grid_list):
            p1_round = self.precision(round_sf(p1, self.display_sig_figs))
            p2_round = self.precision(round_sf(p2, self.display_sig_figs))
            self.param_index_map[(p1_round, p2_round)] = idx
        self.grid_list = grid_list

    def update_plotController_sweeps(self):
        self.ui.plotController.populate_swept_parameter_values(self.param1_values, self.param2_values)
        self.ui.plotController.update_fixed_sliders('param2')

    def on_solve_complete(self):
        self.update_plotController_sweeps()

    def solve_ODE(self):
        self.prepare_parameter_sweeps()
        # Update solver with simulation parameters
        self.solver.duration = self.sim_state['duration']
        self.solver.fs = self.sim_state['fs']
        self.solver.step_size = self.sim_state['dt']
        self.solver.warmup_time = self.sim_state['warmup']
        self.t = np.linspace(0,  self.duration -  1/self.fs, int( self.duration * self.fs))


    def update_plot(self):
        pass

    def closeEvent(self, event):
        self.Plotwidget.closeEvent(event)
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
