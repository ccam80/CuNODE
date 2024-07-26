# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 20:02:51 2024

@author: cca79
"""

from qtpy.QtWidgets import QFrame, QGridLayout, QLabel, QLineEdit, QGroupBox, QWidget
from qtpy.QtCore import Slot, Signal
from numpy import zeros
from gui.resources.widgets.qtdesigner.sim_controller import Ui_simController
import logging

class sim_controller_widget(QFrame, Ui_simController):

    update_system = Signal()

    def __init__(self, parent=None):
        super(sim_controller_widget, self).__init__(parent)
        self.setupUi(self)

        self.sim_state = {
            'dt': 0.0,
            'duration': 0.0,
            'fs': 0.0,
            'param1_sweep_bounds': [0.0,1.00],
            'param1_sweep_scale': 'Linear',
            'param1_num_values': 100,
            'param1_var': '',
            'param1_values': zeros(1),
            'param2_sweep_bounds': [0.0,1.0],
            'param2_sweep_scale': 'Linear',
            'param2_num_values': 100,
            'param2_var': '',
            'param2_values': zeros(1),
            'warmup': 0.0,
            'y0': zeros(1)
        }

        self.system_params_local = {}
        self.param_data = {'param1_var': '',
                           'param2_var': '',
                           'param1_vals': zeros(1),
                           'param2_vals': zeros(1)}

    @Slot(str)
    def update_p1_from(self, _from):
        self.sim_state['param1_sweep_bounds'][0] = float(_from)
        logging.debug(f"P1 from updated: {_from}")

    @Slot(str)
    def update_p1_to(self, to):
        self.sim_state['param1_sweep_bounds'][1] = float(to)
        logging.debug(f"P1 to updated: {to}")

    @Slot(str)
    def update_p1_n(self, n):
        self.sim_state['param1_num_values'] = float(n)
        logging.debug(f"P1 n updated: {n}")

    @Slot(str)
    def update_p1_scale(self, scale):
        self.sim_state["param1_sweep_scale"] = scale
        logging.debug(f"P1 scale updated: {scale}")

    @Slot(str)
    def update_p1_var(self, var):
        self.sim_state["param1_var"] = var
        logging.debug(f"P1 var updated: {var}")

    @Slot(str)
    def update_p2_from(self, _from):
        self.sim_state['param2_sweep_bounds'][0] = float(_from)
        logging.debug(f"P2 from updated: {_from}")

    @Slot(str)
    def update_p2_to(self, to):
        self.sim_state['param2_sweep_bounds'][1] = float(to)
        logging.debug(f"P2 to updated: {to}")

    @Slot(str)
    def update_p2_n(self, n):
        self.sim_state['param2_num_values'] = float(n)
        logging.debug(f"P2 n updated: {n}")

    @Slot(str)
    def update_p2_scale(self, scale):
        self.sim_state["param2_sweep_scale"] = scale
        logging.debug(f"P2 scale updated: {scale}")

    @Slot(str)
    def update_p2_var(self, var):
        self.sim_state["param2_var"] = var
        logging.debug(f"P2 var updated: {var}")

    @Slot(str)
    def update_duration(self, duration):
        self.sim_state['duration'] = float(duration)
        logging.debug("Duration updated: {duration}")

    @Slot(str)
    def update_warmup(self, warmup):
        self.sim_state['warmup'] = float(warmup)
        logging.debug("Warmup updated: {warmup}")

    @Slot(str)
    def update_fs(self, fs):
        self.sim_state['fs'] = float(fs)
        logging.debug("Fs updated: {fs}")

    @Slot(str)
    def update_dt(self, dt):
        self.sim_state['dt'] = float(dt)
        logging.debug("Dt updated: {dt}")

    @Slot(str, str)
    def update_param(self, param, value):
        self.system_params_local[param] = float(value)
        logging.debug(f"Param updated: {param}, {value}")

    @Slot(str, str)
    def update_init(self, index, value):
        self.inits[index] =  float(value)
        logging.debug(f"Init updated: {index}, {value}")

    @Slot(str, str)
    def update_noise(self, index, value):
        self.noise[index] =  float(value)

        logging.debug(f"Noise updated: {index}, {value}")


    def load_system(self, sysparams, noise_sigmas):
        self.local_sysparams_dict = sysparams
        self.local_noise_array = noise_sigmas
        self.populate_sysParams_tab(sysparams, noise_sigmas)
        self.inits = zeros(len(noise_sigmas))
        self.param1Sweep_options.varDdItems = sysparams.keys()
        self.param2Sweep_options.varDdItems = sysparams.keys()
        self.populate_groupbox_with_array(self.inits_box, self.inits)

    def populate_sysParams_tab(self, sysparams_dict, noise_array):
        if self.sysParams_tab.layout():
            for i in reversed(range(self.sysParams_tab.layout().count())):
                widget = self.sysParams_tab.layout().itemAt(i).widget()
                if widget:
                    widget.deleteLater()

        layout = QGridLayout()
        keys = list(sysparams_dict.keys())

        numcols = 2
        for row in range((len(keys) + 1) // numcols):
            col = 0
            for i in range(numcols):
                idx = row * numcols + i
                if idx >= len(keys):
                    break
                key = keys[idx]
                value = sysparams_dict[key]

                label = QLabel(key)
                entry = QLineEdit()
                entry.setText(f"{value:.2f}")
                entry.textChanged.connect(lambda text, k=key: self.update_sysparam(k, text))
                label.setStyleSheet("font-size: 12pt;")
                entry.setStyleSheet("font-size: 12pt;")

                layout.addWidget(label, row, col)
                layout.addWidget(entry, row, col + 1)
                col += numcols

        noise_groupbox = QGroupBox("Gaussian noise (std dev)")
        self.populate_groupbox_with_array(noise_groupbox, noise_array)
        layout.addWidget(noise_groupbox, (len(keys) + 1) // numcols, 0, 1, numcols * 2)

        container = QWidget()
        container.setLayout(layout)
        self.sysParams_tab.layout().addWidget(container)

    def populate_groupbox_with_array(self, groupbox, array):
        layout = groupbox.layout()
        if layout is not None:
            for i in reversed(range(layout.count())):
                widget = layout.itemAt(i).widget()
                if widget:
                    widget.deleteLater()
        else:
            layout = QGridLayout()
            groupbox.setLayout(layout)

        for i, value in enumerate(array):
            label = QLabel(str(i))
            entry = QLineEdit()
            entry.setText(f"{value:.2f}")
            entry.textChanged.connect(lambda text, idx=i: self.update_array(idx, text, array))
            label.setStyleSheet("font-size: 12pt;")
            entry.setStyleSheet("font-size: 12pt;")

            layout.addWidget(label, i, 0)
            layout.addWidget(entry, i, 1)

    def update_sysparam(self, key, text):
        try:
            self.local_sysparams_dict[key] = float(text)
        except ValueError:
            pass

    def update_array(self, index, text, array):
        try:
            array[index] = float(text)
        except ValueError:
            pass
