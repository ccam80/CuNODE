# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 21:14:22 2024

@author: cca79
"""

import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

def fill_plotsettings_frame(self):
    
    self.grid_or_set = tk.StringVar(value='grid')
    self.x_axis_var = tk.StringVar(value='param_1')
    self.y_axis_var = tk.StringVar(value='param_2')
    self.z_axis_var = tk.StringVar(value='fft_mag')
    self.singleplot_style_var = tk.StringVar(value='time')
    self.param1_var = tk.DoubleVar()
    self.param2_var = tk.DoubleVar()
    self.chosen_frequency_var = tk.DoubleVar()
    
    self.grid_or_set_l = ttk.Label(self.plotsettings_frame, text="Plot whole grid or single set")
    self.gridplot_b = ttk.Radiobutton(self.plotsettings_frame, variable=self.grid_or_set, value='grid', text="Grid", command=self.update_axes)
    self.singleplot_b = ttk.Radiobutton(self.plotsettings_frame, variable=self.grid_or_set, value='single', text="Single", command=self.update_axes)
    
    self.gridplot_opts_l = ttk.Label(self.plotsettings_frame, text="Axes for 3d grid plot")
    
    self.gridx_dd = ttk.Combobox(self.plotsettings_frame,
                                   textvariable=self.x_axis_var,
                                   values=['param_1', 'param_2', 'time', 'freq'])
    self.gridy_dd = ttk.Combobox(self.plotsettings_frame,
                                   textvariable=self.y_axis_var,
                                   values=['param_1', 'param_2'])
    self.gridz_dd = ttk.Combobox(self.plotsettings_frame,
                                   textvariable=self.z_axis_var,
                                   values=['fft_mag', 'fft_phase', 'time-domain'])
    
    self.gridx_dd.bind("<<ComboboxSelected>>", self.update_3d)
    self.gridy_dd.bind("<<ComboboxSelected>>", self.update_3d)
    self.gridz_dd.bind("<<ComboboxSelected>>", self.update_3d)

    self.gridx_dd.grid(row=3, column=1, padx=10, pady=10)
    
    self.singleplot_l = ttk.Label(self.plotsettings_frame,text="Single plot settings")
    self.singletime_b = ttk.Radiobutton(self.plotsettings_frame, variable=self.singleplot_style_var, value='time', text="time", command=self.update_axes)
    self.singlespec_b = ttk.Radiobutton(self.plotsettings_frame, variable=self.singleplot_style_var, value='spec', text="spectrogram", command=self.update_axes)
    
    self.grid_or_set_l.grid(row=0, column=0, columnspan=4)
    self.gridplot_b.grid(row=1, column=0, columnspan=2)
    self.singleplot_b.grid(row=1, column=2, columnspan=2)
    self.gridplot_opts_l.grid(row=2, column=0, columnspan=4)
    self.gridx_dd.grid(row=3, column=0)
    self.gridy_dd.grid(row=3, column=1)
    self.gridz_dd.grid(row=3, column=2)
    self.singleplot_l.grid(row=4, column=0, columnspan=4)
    self.singletime_b.grid(row=5, column=0, columnspan=2)
    self.singlespec_b.grid(row=5, column=2, columnspan=2)
    
    # Create dropdowns for parameter selection
    ttk.Label(self.plotsettings_frame, text="Select a:").grid(row=6, column=0, padx=10, pady=10)
    self.dropdown_a = ttk.Combobox(self.plotsettings_frame,
                                   textvariable=self.param1_var,
                                   values=sorted(list(set([param[0] for param in self.grid_values]))))
    self.dropdown_a.grid(row=6, column=1, padx=10, pady=10)
    self.dropdown_a.bind("<<ComboboxSelected>>", self.update_selected_params)

    ttk.Label(self.plotsettings_frame, text="Select b:").grid(row=7, column=0, padx=10, pady=10)
    self.dropdown_b = ttk.Combobox(self.plotsettings_frame
                                   , textvariable=self.param2_var,
                                   values=sorted(list(set([param[1] for param in self.grid_values]))))
    self.dropdown_b.grid(row=7, column=1, padx=10, pady=10)
    self.dropdown_b.bind("<<ComboboxSelected>>", self.update_selected_params)
    
    ttk.Label(self.plotsettings_frame, text="Frequency:").grid(row=8, column=0, padx=10, pady=10)
    self.button = ttk.Button(self.plotsettings_frame, text="Open Selection window", command=self.freqselect_menu)
    self.button.grid(row = 8, column = 1, padx=10, pady=10)
    
    self.dropdown_freq = ttk.Combobox(self.plotsettings_frame, 
                                      textvariable=self.chosen_frequency_var,
                                      values=self.solved_ODE.f)


     # Create Fix parameter radio buttons
    ttk.Label(self.plotsettings_frame, text="Fix:").grid(row=9, column=0, columnspan=2, padx=10, pady=10)
    self.fix_param = tk.StringVar(value="a")
    ttk.Radiobutton(self.plotsettings_frame, text="a", variable=self.fix_param,
                    value="a", command=self.set_fixed_param).grid(row=9, column=1, padx=10, pady=10)
    ttk.Radiobutton(self.plotsettings_frame, text="b", variable=self.fix_param,
                    value="b", command=self.set_fixed_param).grid(row=9, column=2, padx=10, pady=10)
    
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
    self.fs_var = tk.DoubleVar()
    self.duration_var = tk.DoubleVar()
    self.step_size_var = tk.DoubleVar()
    
    self.param1_l = tk.Label(self.simsettings_frame, text='Parameter 1')
    
    constants = list(self.solved_ODE.system.constants_dict.keys())
    self.param1_select_dd = ttk.Combobox(self.simsettings_frame, textvariable=self.current_param1_label, values=constants)
    self.param1_select_dd.bind("<<ComboboxSelected>>", self.update_param_label)
    
    self.p1start_l = tk.Label(self.simsettings_frame, text="Start")
    self.p1start_e = tk.Entry(self.simsettings_frame, textvariable=self.param1_start_var)
    
    self.p1end_l = tk.Label(self.simsettings_frame, text="End")
    self.p1end_e = tk.Entry(self.simsettings_frame, textvariable=self.param1_end_var)
    
    self.p1_n_l = tk.Label(self.simsettings_frame, text="Points")
    self.p1_n_e = tk.Entry(self.simsettings_frame, textvariable=self.param1_n_var)

    self.param1_l.grid(row=0, column=0, columnspan=7, sticky='ew')
    self.param1_select_dd.grid(row=1, column=0)
    self.p1start_l.grid(row=1, column=1)
    self.p1start_e.grid(row=1, column=2)
    self.p1end_l.grid(row=1, column=3)
    self.p1end_e.grid(row=1, column=4)
    self.p1_n_l.grid(row=1, column=5)
    self.p1_n_e.grid(row=1, column=6)


    self.param2_l = tk.Label(self.simsettings_frame, text='Parameter 2')

    self.param2_select_dd = ttk.Combobox(self.simsettings_frame, textvariable=self.current_param2_label, values=constants)
    self.param2_select_dd.bind("<<ComboboxSelected>>", self.update_param_label)
    
    self.p2start_l = tk.Label(self.simsettings_frame, text="Start")
    self.p2start_e = tk.Entry(self.simsettings_frame, textvariable=self.param2_start_var)
    
    self.p2end_l = tk.Label(self.simsettings_frame, text="End")
    self.p2end_e = tk.Entry(self.simsettings_frame, textvariable=self.param2_end_var)
    
    self.p2_n_l = tk.Label(self.simsettings_frame, text="Points")
    self.p2_n_e = tk.Entry(self.simsettings_frame, textvariable=self.param2_n_var)

    self.param2_l.grid(row=2, column=0, columnspan=7, sticky='ew')
    self.param2_select_dd.grid(row=3, column=0)
    self.p2start_l.grid(row=3, column=1)
    self.p2start_e.grid(row=3, column=2)
    self.p2end_l.grid(row=3, column=3)
    self.p2end_e.grid(row=3, column=4)
    self.p2_n_l.grid(row=3, column=5)
    self.p2_n_e.grid(row=3, column=6)
    
    row = 4
    col = 0
    for key, value in self.solved_ODE.system.constants_dict.items():
        
        label = tk.Label(self.simsettings_frame, text=key)
        
        var = tk.DoubleVar(value=value)
        entry = tk.Entry(self.simsettings_frame, textvariable=var)
        
        label.grid(row=row, column=col)
        entry.grid(row=row, column=col + 1)
        
        self.constant_vars[key] = var
        
        col += 2
        if col >= 4:  
            col = 0
            row += 1
                
                    
    row += len(self.solved_ODE.system.constants_dict)

    self.fs_label = tk.Label(self.simsettings_frame, text="fs")
    self.fs_label.grid(row=row, column=0)
    self.fs_entry = tk.Entry(self.simsettings_frame, textvariable=self.fs_var)
    self.fs_entry.grid(row=row, column=1)
    
    self.duration_label = tk.Label(self.simsettings_frame, text="Duration")
    self.duration_label.grid(row=row+1, column=0)
    self.duration_entry = tk.Entry(self.simsettings_frame, textvariable=self.duration_var)
    self.duration_entry.grid(row=row+1, column=1)
    
    self.step_size_label = tk.Label(self.simsettings_frame, text="Step Size")
    self.step_size_label.grid(row=row+2, column=0)
    self.step_size_entry = tk.Entry(self.simsettings_frame, textvariable=self.step_size_var)
    self.step_size_entry.grid(row=row+2, column=1)
    
    # Solve button
    self.solve_button = tk.Button(self.simsettings_frame, text="Solve", command=self.solve_ode)
    self.solve_button.grid(row=row+3, column=0, columnspan=2, pady=10)
    
    self.set_cell_weights(self.simsettings_frame)

def setup_ui(self): 
    self.root = tk.Tk()
    self.root.title("Gridsearch Results")
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

    # self.root.bind("<Configure>", self.resize_canvas)

    self.plotsettings_frame = ttk.Labelframe(self.control_frame, text="Plot Settings")

    self.plotsettings_frame.grid(row=0, column=0, sticky='new')
    self.fill_plotsettings_frame()


    # self.set_cell_weights(self.root, weight=4)
    # self.root.columnconfigure(4,weight=)    
    
    self.simsettings_frame = ttk.Labelframe(self.control_frame, text="Simulation Settings")
    self.simsettings_frame.grid(row=1, column=0, sticky='new')
    self.fill_simsettings_frame()
    
    self.set_cell_weights(self.root, 1)
    self.set_cell_weights(self.control_frame, 1)

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