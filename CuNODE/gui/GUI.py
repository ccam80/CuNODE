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
from _utils import round_sf, round_list_sf
from solvers.eulermaruyama import Solver
from gui.modelController import modelController
import numpy as np
from importlib.util import spec_from_file_location, module_from_spec
from pubsub import PubSub
from datetime import datetime
from qtpy.QtWidgets import QApplication, QMainWindow, QLabel, QLineEdit, QGridLayout, QWidget, QFileDialog, QGroupBox, QErrorMessage
from qtpy.QtGui import QAction, QActionGroup
from qtpy import QtCore
from gui.resources.qtdesigner.QT_simGUI import Ui_MainWindow


class ODE_GUI(QMainWindow, Ui_MainWindow):
    variable_label_keys = {'PSD at selected frequency': 'psd',
                           'PSD magnitude': 'psd',
                           'FFT phase at selected frequency': 'phase',
                           'FFT phase': 'phase',
                           'RMS amplitude': 'rms',
                           'Signal amplitude': 'ampl'
                           }

    def __init__(self):
        super().__init__()
        self.init_logging()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        #Set up message-passing between widgets
        self.messenger = PubSub()
        self.subscribe_to_messages()
        self.ui.simController.register_messaging_service(self.messenger)
        self.ui.plotController.register_messaging_service(self.messenger)
        self.ui.Plotwidget.register_messaging_service(self.messenger)

        self.set_starting_state()
        self.precision = np.float64
        self.init_precision_action_group()
        self.diffeq_system_file_path = None

        logging.info("ODE_Explorer_Controller initialized")

        self.param_index_map = {}
        self.grid_list = np.zeros(1, dtype=self.precision)
        self.display_sig_figs = 4 # Add to settings pane somewhere
        self.ui.plotController.update_display_sigfigs(self.display_sig_figs)
        self.external_variables = {'param1_values': np.zeros(1, dtype=self.precision),
                                   'param2_values': np.zeros(1, dtype=self.precision),
                                   'param_labels': ["",""]}
        self.param_index_map = {}
        self.model = modelController()
        self.model.register_messaging_service(self.messenger)
        self.model.load_solver(Solver)

        #AUTOLOAD SYSTEM FOR DEBUG
        system = self.get_system_from_file(r"\\file\Usersc$\cca79\Home\My Documents\Work\MEMS\sims\CUDA_system\CuNODE\systems\thermal_cantilever_ax_b.py")
        self.load_system(system)

    def set_starting_state(self):
        self.ui.controlToolBox.setCurrentIndex(1)
        self.ui.plotController.plotSettingsTabs.setCurrentIndex(1)
        self.ui.plotController.plotSettingsTabs.setCurrentIndex(0)
        self.ui.simController.simSettingsTabs.setCurrentIndex(0)
        self.ui.plotController.update_animation_step(1)

    def subscribe_to_messages(self):
        self.messenger.subscribe("param1_values", self.update_param1_view)
        self.messenger.subscribe("param2_values", self.update_param2_view)
        self.messenger.subscribe("param_labels", self.update_param_labels)

    def publish(self, topic, data):
        self.messenger.publish(topic, data)

    def update_param1_view(self, p1):
        self.external_variables["param1_values"] = p1

    def update_param2_view(self, p2):
        self.external_variables["param2_values"] = p2

    def update_param_labels(self, labels):
        self.external_variables["param_labels"] = labels

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
            self.publish('precision', self.precision)
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


    def generate_grid_list_and_map(self):
        """Generate 1D list of all requested parameter combinations.
        Save a rounded version of each index into a dict for easy lookup
        to match with a dataset request from the plotter.

        """
        p1view = self.external_variables["param1_values"]
        p2view = self.external_variables["param2_values"]
        #Round them to avoid any gremlins caused by mismatched precision/rounding when checking for equality
        p1_rounded = self.precision(round_list_sf(p1view, self.display_sig_figs))
        p2_rounded = self.precision(round_list_sf(p2view, self.display_sig_figs))
        self.param_index_map = {}
        self.grid_list = [(p1, p2) for p1 in p1_rounded for p2 in p2_rounded]
        self.param_index_map = {tup: idx for idx, tup in enumerate(self.grid_list)}
        self.publish("param_index_map", self.param_index_map)


    def on_solve_request(self):
        self.generate_grid_list_and_map()

        #replace with pubsub
        params = self.ui.simController.get_sysparams()

        self.model.update_system_parameters(params)

        self.solve_ODE()

    def solve_ODE(self):

        #pubsubise this part
        sim_state = self.ui.simController.get_sim_state()

        sweeps = {'values': self.grid_list,
                  'vars': self.external_variables['param_labels']}
        self.model.run_solver(self.model.system,
                              sim_state, sweeps)
        self.on_solve_complete()

    def on_solve_complete(self):
        # self.update_plotController_sweeps()
        # self.pass_independent_variables_to_plotter()
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
    window = ODE_GUI()
    window.show()
    sys.exit(app.exec())