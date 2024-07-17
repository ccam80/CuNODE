# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 21:14:22 2024

@author: cca79
"""

import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure


def update_combobox_values(combobox, new_values):
    """
    Update the values of the given combobox.

    :param combobox: The tk.Combobox widget to update.
    :param new_values: A list of new values to set in the combobox.
    """
    combobox['values'] = new_values

def update_label_text(label, new_text):
    """
    Update the text of the given label.

    :param label: The tk.Label widget to update.
    :param new_text: The new text to set in the label.
    """
    label.config(text=new_text)

def fill_plotsettings_frame(self):

    self.grid_or_set = tk.StringVar(value='grid')
    self.x_axis_var = tk.StringVar(value='param_1')
    self.y_axis_var = tk.StringVar(value='param_2')
    self.z_axis_var = tk.StringVar(value='fft_mag')
    self.singleplot_style_var = tk.StringVar(value='time')
    self.param1_var = tk.DoubleVar()
    self.param2_var = tk.DoubleVar()
    self.chosen_frequency_var = tk.DoubleVar()
    self.xscale = tk.StringVar(value='lin')
    self.yscale = tk.StringVar(value='lin')
    self.zscale = tk.StringVar(value='lin')

    self.xslice_min_var = tk.DoubleVar(value=0)
    self.xslice_max_var = tk.DoubleVar(value=1e6)
    self.yslice_min_var = tk.DoubleVar(value=0)
    self.yslice_max_var = tk.DoubleVar(value=1e6)
    self.zslice_min_var = tk.DoubleVar(value=0)
    self.zslice_max_var = tk.DoubleVar(value=1e6)

    #sensible inits
    self.param1_var.set(self.param1_values[0])
    self.param2_var.set(self.param2_values[1])

    self.grid_or_set_l = ttk.Label(self.plotsettings_frame, text="Plot whole grid or single set")
    self.gridplot_b = ttk.Radiobutton(self.plotsettings_frame, variable=self.grid_or_set, value='grid', text="Grid", command=self.update_axes)
    self.singleplot_b = ttk.Radiobutton(self.plotsettings_frame, variable=self.grid_or_set, value='single', text="Single", command=self.update_axes)


    self.gridplot_opts_f = ttk.LabelFrame(self.plotsettings_frame, text="3d plot options")

    self.gridx_dd = ttk.Combobox(self.gridplot_opts_f,
                                   textvariable=self.x_axis_var,
                                   values=['param_1', 'param_2', 'time', 'freq'])
    self.gridy_dd = ttk.Combobox(self.gridplot_opts_f,
                                   textvariable=self.y_axis_var,
                                   values=['param_1', 'param_2'])
    self.gridz_dd = ttk.Combobox(self.gridplot_opts_f,
                                   textvariable=self.z_axis_var,
                                   values=['fft_mag', 'fft_phase', 'time-domain'])

    self.axes_slice_frame = tk.Frame(self.gridplot_opts_f)
    self.xslice_min_e = tk.Entry(self.axes_slice_frame, textvariable=self.xslice_min_var, width=7).grid(row=0,column=1, sticky='nsew')
    self.xslice_max_e = tk.Entry(self.axes_slice_frame, textvariable=self.xslice_max_var, width=7).grid(row=0,column=3, sticky='nsew')
    self.yslice_min_e = tk.Entry(self.axes_slice_frame, textvariable=self.yslice_min_var, width=7).grid(row=0,column=6, sticky='nsew')
    self.yslice_max_e = tk.Entry(self.axes_slice_frame, textvariable=self.yslice_max_var, width=7).grid(row=0,column=8, sticky='nsew')
    self.zslice_min_e = tk.Entry(self.axes_slice_frame, textvariable=self.zslice_min_var, width=7).grid(row=0,column=11, sticky='nsew')
    self.zslice_max_e = tk.Entry(self.axes_slice_frame, textvariable=self.zslice_max_var, width=7).grid(row=0,column=13, sticky='nsew')

    openbracket = tk.Label(self.axes_slice_frame, text="[").grid(row=0,column=0, sticky='nsew', padx=5)
    colonlabel1 = tk.Label(self.axes_slice_frame, text=":").grid(row=0,column=2, sticky='nsew')
    closebracket = tk.Label(self.axes_slice_frame, text="]").grid(row=0,column=4, sticky='nsew', padx=5)
    openbracket = tk.Label(self.axes_slice_frame, text="[").grid(row=0,column=5, sticky='nsew', padx=5)
    colonlabel2 = tk.Label(self.axes_slice_frame, text=":").grid(row=0,column=7, sticky='nsew')
    closebracket = tk.Label(self.axes_slice_frame, text="]").grid(row=0,column=9, sticky='nsew', padx=5)
    openbracket = tk.Label(self.axes_slice_frame, text="[").grid(row=0,column=10, sticky='nsew', padx=5)
    colonlabel3 = tk.Label(self.axes_slice_frame, text=":").grid(row=0,column=12, sticky='nsew')
    closebracket = tk.Label(self.axes_slice_frame, text="]").grid(row=0,column=14, sticky='nsew', padx=5)

    self.xscale_l = tk.Label(self.axes_slice_frame, text="X Scale:")
    self.xscale_lin_b = ttk.Radiobutton(self.axes_slice_frame, variable=self.xscale, value='lin', text="Linear")
    self.xscale_log_b = ttk.Radiobutton(self.axes_slice_frame, variable=self.xscale, value='log', text="Logarithmic")
    self.yscale_l = tk.Label(self.axes_slice_frame, text="Y Scale:")
    self.yscale_lin_b = ttk.Radiobutton(self.axes_slice_frame, variable=self.yscale, value='lin', text="Linear")
    self.yscale_log_b = ttk.Radiobutton(self.axes_slice_frame, variable=self.yscale, value='log', text="Logarithmic")
    self.zscale_l = tk.Label(self.axes_slice_frame, text="Z Scale:")
    self.zscale_lin_b = ttk.Radiobutton(self.axes_slice_frame, variable=self.zscale, value='lin', text="Linear")
    self.zscale_log_b = ttk.Radiobutton(self.axes_slice_frame, variable=self.zscale, value='log', text="Logarithmic")

    self.xscale_l.grid(row=1, column=0, columnspan=2, sticky='nsew')
    self.xscale_lin_b.grid(row=1, column=2, sticky='nsew')
    self.xscale_log_b.grid(row=1, column=3, sticky='nsew')
    self.yscale_l.grid(row=1, column=4, columnspan=2, sticky='nsew')
    self.yscale_lin_b.grid(row=1, column=6, sticky='nsew')
    self.yscale_log_b.grid(row=1, column=7, sticky='nsew')
    self.zscale_l.grid(row=1, column=8, columnspan=2, sticky='nsew')
    self.zscale_lin_b.grid(row=1, column=10, sticky='nsew')
    self.zscale_log_b.grid(row=1, column=11, sticky='nsew')

    self.updateplot_button = tk.Button(self.gridplot_opts_f, text="Update", command=self.update_3d)
    self.axes_slice_frame.bind("<Return>", self.update_plot_slice)
    self.set_cell_weights(self.axes_slice_frame)

    self.singleplot_l = ttk.Label(self.plotsettings_frame,text="Single plot settings")
    self.singletime_b = ttk.Radiobutton(self.plotsettings_frame, variable=self.singleplot_style_var, value='time', text="time", command=self.update_axes)
    self.singlespec_b = ttk.Radiobutton(self.plotsettings_frame, variable=self.singleplot_style_var, value='spec', text="spectrogram", command=self.update_axes)

    self.grid_or_set_l.grid(row=0, column=0, columnspan=1, sticky='nsew')
    self.gridplot_b.grid(row=0, column=1, columnspan=1, sticky='nsew')
    self.singleplot_b.grid(row=0, column=2, columnspan=1, sticky='nsew')
    self.gridplot_opts_f.grid(row=1, column=0, columnspan=3, rowspan=3, sticky='nsew')
    self.gridx_dd.grid(row=0, column=0, sticky='nsew')
    self.gridy_dd.grid(row=0, column=1, sticky='nsew')
    self.gridz_dd.grid(row=0, column=2, sticky='nsew')
    self.axes_slice_frame.grid(row=1, column=0, columnspan=3, rowspan=2, sticky='nsew')
    self.updateplot_button.grid(row=0, rowspan=5, column=15, sticky='nsew')
    self.singleplot_l.grid(row=6, column=0, sticky='nsew')
    self.singletime_b.grid(row=6, column=1, sticky='nsew')
    self.singlespec_b.grid(row=6, column=2, sticky='nsew')

    # Create dropdowns for parameter selection
    self.p1select_label = ttk.Label(self.plotsettings_frame, text="Select parameter 1:")
    self.p1select_dd = ttk.Combobox(self.plotsettings_frame,
                                   textvariable=self.param1_var,
                                   values=sorted(list(set([param[0] for param in self.grid_values]))))

    self.p2select_label = ttk.Label(self.plotsettings_frame, text="Select parameter 2:")
    self.p2select_dd = ttk.Combobox(self.plotsettings_frame
                                   , textvariable=self.param2_var,
                                   values=sorted(list(set([param[1] for param in self.grid_values]))))

    self.p1select_dd.bind("<<ComboboxSelected>>", self.update_selected_params)
    self.p2select_dd.bind("<<ComboboxSelected>>", self.update_selected_params)

    self.p1select_label.grid(row=7, column=0)
    self.p1select_dd.grid(row=7, column=1)
    self.p2select_label.grid(row=8, column=0)
    self.p2select_dd.grid(row=8, column=1)


    self.freqselect_label = ttk.Label(self.plotsettings_frame, text="Frequency to slice at:")
    self.freqselect_button = ttk.Button(self.plotsettings_frame, text="Open Selection window", command=self.freqselect_menu)

    self.freqselect_label.grid(row=9, column=0)
    self.freqselect_button.grid(row=9, column=1)


    self.set_cell_weights(self.plotsettings_frame)


def fill_simsettings_frame(self):
    self.current_param1_label = tk.StringVar()
    self.current_param2_label = tk.StringVar()
    self.param1_start_var = tk.DoubleVar()
    self.param1_end_var = tk.DoubleVar()
    self.param1_n_var = tk.IntVar()
    self.param2_start_var = tk.DoubleVar()
    self.param2_end_var = tk.DoubleVar()
    self.param2_n_var = tk.IntVar()
    self.constant_vars = {}
    self.sigma_vars = {}
    self.init_vars = {}
    self.fs_var = tk.DoubleVar(value=1.0)
    self.duration_var = tk.DoubleVar(value=100)
    self.step_size_var = tk.DoubleVar(value=0.001)
    self.warmup_var = tk.DoubleVar(value=100)
    self.param1_mode = tk.StringVar(value='lin')
    self.param2_mode = tk.StringVar(value='lin')


    self.param1_l = tk.Label(self.simsettings_frame, text='Parameter 1')

    constants = list(self.solved_ODE.system.constants_dict.keys())
    self.param1_select_dd = ttk.Combobox(self.simsettings_frame, textvariable=self.current_param1_label, values=constants)
    self.param1_select_dd.bind("<<ComboboxSelected>>", self.update_param_label)
    self.param1_select_dd.current(0)


    self.p1start_l = tk.Label(self.simsettings_frame, text="Start")
    self.p1start_e = tk.Entry(self.simsettings_frame, textvariable=self.param1_start_var)

    self.p1end_l = tk.Label(self.simsettings_frame, text="End")
    self.p1end_e = tk.Entry(self.simsettings_frame, textvariable=self.param1_end_var)

    self.p1_n_l = tk.Label(self.simsettings_frame, text="Points")
    self.p1_n_e = tk.Entry(self.simsettings_frame, textvariable=self.param1_n_var)

    self.param1_mode_l = tk.Label(self.simsettings_frame, text="Param 1 Sweep mode:")
    self.param1_lin_b = ttk.Radiobutton(self.simsettings_frame, variable=self.param1_mode, value='lin', text="Linear")
    self.param1_log_b = ttk.Radiobutton(self.simsettings_frame, variable=self.param1_mode, value='log', text="Logarithmic")

    self.param1_l.grid(row=0, column=0, columnspan=4, sticky='nsew')
    self.param1_select_dd.grid(row=2, column=0, sticky='nsew')
    self.param1_mode_l.grid(row=3, column=0, columnspan=2, sticky='nsew')
    self.param1_lin_b.grid(row=3, column=2, sticky='nsew')
    self.param1_log_b.grid(row=3, column=3, sticky='nsew')
    self.p1start_l.grid(row=1, column=1, sticky='nsew')
    self.p1start_e.grid(row=2, column=1, sticky='nsew')
    self.p1end_l.grid(row=1, column=2, sticky='nsew')
    self.p1end_e.grid(row=2, column=2, sticky='nsew')
    self.p1_n_l.grid(row=1, column=3, sticky='nsew')
    self.p1_n_e.grid(row=2, column=3, sticky='nsew')


    self.param2_l = tk.Label(self.simsettings_frame, text='Parameter 2')

    self.param2_select_dd = ttk.Combobox(self.simsettings_frame, textvariable=self.current_param2_label, values=constants)
    self.param2_select_dd.bind("<<ComboboxSelected>>", self.update_param_label)
    self.param2_select_dd.current(1)

    self.p2start_l = tk.Label(self.simsettings_frame, text="Start")
    self.p2start_e = tk.Entry(self.simsettings_frame, textvariable=self.param2_start_var)
    self.p2end_l = tk.Label(self.simsettings_frame, text="End")
    self.p2end_e = tk.Entry(self.simsettings_frame, textvariable=self.param2_end_var)
    self.p2_n_l = tk.Label(self.simsettings_frame, text="Points")
    self.p2_n_e = tk.Entry(self.simsettings_frame, textvariable=self.param2_n_var)
    self.param2_mode_l = tk.Label(self.simsettings_frame, text="Param 2 Sweep mode:")
    self.param2_lin_b = ttk.Radiobutton(self.simsettings_frame, variable=self.param2_mode, value='lin', text="Linear")
    self.param2_log_b = ttk.Radiobutton(self.simsettings_frame, variable=self.param2_mode, value='log', text="Logarithmic")

    self.param2_l.grid(row=4, column=0, columnspan=4, sticky='nsew')
    self.param2_select_dd.grid(row=6, column=0, sticky='nsew')
    self.param2_mode_l.grid(row=7, column=0, columnspan=2, sticky='nsew')
    self.param2_lin_b.grid(row=7, column=2, sticky='nsew')
    self.param2_log_b.grid(row=7, column=3, sticky='nsew')
    self.p2start_l.grid(row=5, column=1, sticky='nsew')
    self.p2start_e.grid(row=6, column=1, sticky='nsew')
    self.p2end_l.grid(row=5, column=2, sticky='nsew')
    self.p2end_e.grid(row=6, column=2, sticky='nsew')
    self.p2_n_l.grid(row=5, column=3, sticky='nsew')
    self.p2_n_e.grid(row=6, column=3, sticky='nsew')

    self.constants_frame = ttk.Labelframe(self.simsettings_frame, text="System constants")
    self.noise_sigmas_frame = ttk.LabelFrame(self.constants_frame, text='Noise')
    self.init_frame = ttk.LabelFrame(self.constants_frame, text = "y0")

    lastrow = 7
    row=0
    col=0
    for key, value in self.solved_ODE.system.constants_dict.items():
        label = tk.Label(self.constants_frame, text=key)
        var = tk.DoubleVar(value=value)
        entry = tk.Entry(self.constants_frame, textvariable=var)

        label.grid(row=row, column=col, sticky='nsew')
        entry.grid(row=row, column=col + 1, sticky='nsew')

        self.constant_vars[key] = var

        col += 2
        if col >= 4:
            col = 0
            row += 1

    sigmarow = 0

    for i, value in enumerate(self.solved_ODE.noise_sigmas):
        var = tk.DoubleVar(value=value)
        label = tk.Label(self.noise_sigmas_frame, text=f'Noise Sigma {i}')
        entry = tk.Entry(self.noise_sigmas_frame, textvariable=var)
        self.sigma_vars[i] = var

        label.grid(row=i, column=0, sticky='nsew')
        entry.grid(row=i, column=1, sticky='nsew')

        label = tk.Label(self.init_frame, text=f'y_{i}[0]')
        var = tk.DoubleVar(value=value)
        entry = tk.Entry(self.init_frame, textvariable=var)
        self.init_vars[i] = var

        label.grid(row=i, column=0, sticky='nsew')
        entry.grid(row=i, column=1, sticky='nsew')

        sigmarow += 1

        sigmarows = i

    self.constants_frame.grid(row=lastrow + 1,column=0, rowspan=row, columnspan=7, sticky='nsew')
    self.noise_sigmas_frame.grid(row=0,column=5, rowspan=sigmarows, columnspan=2, sticky='nsew')
    self.init_frame.grid(row=sigmarows,column=5, rowspan=sigmarows, columnspan=2, sticky='nsew')


    row = lastrow + max(row, sigmarows)+1

    self.fs_label = tk.Label(self.simsettings_frame, text="fs")
    self.fs_entry = tk.Entry(self.simsettings_frame, textvariable=self.fs_var)

    self.duration_label = tk.Label(self.simsettings_frame, text="Duration")
    self.duration_entry = tk.Entry(self.simsettings_frame, textvariable=self.duration_var)
    self.step_size_label = tk.Label(self.simsettings_frame, text="Step Size")
    self.step_size_entry = tk.Entry(self.simsettings_frame, textvariable=self.step_size_var)
    self.warmup_label = tk.Label(self.simsettings_frame, text="Warmup Time")
    self.warmup_entry = tk.Entry(self.simsettings_frame, textvariable=self.warmup_var)
    self.solve_button = tk.Button(self.simsettings_frame, text="Solve", command=self.solve_ode)

    self.fs_label.grid(row=row, column=0, sticky='nsew')
    self.fs_entry.grid(row=row, column=1, sticky='nsew')
    self.duration_label.grid(row=row+1, column=0, sticky='nsew')
    self.duration_entry.grid(row=row+1, column=1, sticky='nsew')
    self.step_size_label.grid(row=row+2, column=0, sticky='nsew')
    self.step_size_entry.grid(row=row+2, column=1, sticky='nsew')
    self.warmup_label.grid(row=row+3, column=0, sticky='nsew')
    self.warmup_entry.grid(row=row+3, column=1, sticky='nsew')
    self.solve_button.grid(row=row+4, column=0, columnspan=2, pady=10, sticky='nsew')

    self.set_cell_weights(self.constants_frame)
    self.set_cell_weights(self.noise_sigmas_frame)
    self.set_cell_weights(self.init_frame)
    self.set_cell_weights(self.simsettings_frame)

def setup_ui(self):
    self.root = tk.Tk()
    self.root.title("Gridsearch Visualiser")
    self.root.geometry("1920x1080")

    self.fig_frame = ttk.Frame(self.root)
    self.control_frame = ttk.Frame(self.root)
    self.control_frame.grid(row=0, column=3, rowspan=10, sticky='nsew')
    self.fig_frame.grid(row=0, column=0, rowspan=10, columnspan=3, sticky='nsew')

    # Create plot canvas
    self.fig = Figure(figsize=(18, 9))
    self.ax = [self.fig.add_subplot(111, projection='3d')]
    self.canvas = FigureCanvasTkAgg(self.fig, master=self.fig_frame)
    self.toolbar = NavigationToolbar2Tk(self.canvas, self.fig_frame, pack_toolbar=False)
    self.toolbar.update()

    self.toolbar.grid(row=0,column=0, sticky='nw')
    self.canvas.get_tk_widget().grid(row=1, column=0, rowspan=9, sticky='nsew')

    self.resize_time = 0
    self.last_winsize = (0,0)
    self.plotsettings_frame = ttk.Labelframe(self.control_frame, text="Plot Settings")

    self.plotsettings_frame.grid(row=0, column=0, sticky='new')
    self.fill_plotsettings_frame()


    # self.set_cell_weights(self.root, weight=4)
    # self.root.columnconfigure(4,weight=)

    self.simsettings_frame = ttk.Labelframe(self.control_frame, text="Simulation Settings")
    self.simsettings_frame.grid(row=1, column=0, sticky='new')
    self.fill_simsettings_frame()

    self.set_cell_weights(self.root, 1)
    self.root.columnconfigure(0, weight=4)

    self.set_cell_weights(self.control_frame, 1)
    self.set_cell_weights(self.fig_frame)
def freqselect_menu(self):
    # Create a new Toplevel window
    self.freqselect_window = tk.Toplevel(self.root)
    self.freqselect_window.title("Select a frquency")

    # Create a Listbox with a Scrollbar
    self.frame = ttk.Frame(self.freqselect_window)
    self.frame.pack(fill=tk.BOTH, expand=True)

    self.listbox = tk.Listbox(self.frame, height=20)
    self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    self.scrollbar = ttk.Scrollbar(self.frame, orient=tk.VERTICAL, command=self.listbox.yview)
    self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    self.listbox.configure(yscrollcommand=self.scrollbar.set)

    for item in self.solved_ODE.f:
        self.listbox.insert(tk.END, item)

    # Bind the selection event
    self.listbox.bind("<<ListboxSelect>>", self.select_frequency)
