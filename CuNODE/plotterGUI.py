import numpy as np
from diffeq_system import diffeq_system, system_constants
from CUDA_ODE import CUDA_ODE
# from CUDA_ODE import CUDA_ODE
from time import time
from cupyx.scipy.signal import stft
from _utils import round_sf, round_list_sf
from scipy.signal.windows import hann
import gui_layout
from tkinter import font as tkFont
import tkinter as tk
from tkinter import ttk
from ctypes import windll
windll.shcore.SetProcessDpiAwareness(1)

from vispy import app, scene
from vispy.util.filter import gaussian_filter




labels = {'param_1': 'a',
          'param_2': 'b',
          'time': 'Time (s)',
          'freq': 'Frequency (Hz-like)',
          'fft_mag': 'Power Spectral Magnitude (V^2-ish)',
          'fft_phase': 'Not implmented since switching to welch',
          'time-domain': 'Amplitude (RMS if no time axis)'}

class Gridplotter:
    def __init__(self, ODE, grid_values, precision=np.float64):
        self.grid_values = grid_values
        self.solved_ODE = ODE
        self.local_constants = self.solved_ODE.system.constants_dict.copy()

        self.precision=precision
        self.t = np.linspace(0,  self.solved_ODE.duration -  1/self.solved_ODE.fs, int( self.solved_ODE.duration * self.solved_ODE.fs))
        self.plotstate = 4
        self.freq_index = 0
        self.param_1_val = 0
        self.param_2_val = 0
        self.selected_index = 0
        self.sort_sf = 5 #how many sf to round to before finding a match in grid_params list

        self.min_x_index = 0
        self.max_x_index = None
        self.min_y_index = 0
        self.max_y_index = None

        self.labels = labels
        self.param1_values = np.unique([param[0] for param in self.grid_values])
        self.param2_values = np.unique([param[1] for param in self.grid_values])
        self.generate_index_map()

        self.setup_ui = gui_layout.setup_ui.__get__(self)
        self.freqselect_menu = gui_layout.freqselect_menu.__get__(self)
        self.fill_simsettings_frame = gui_layout.fill_simsettings_frame.__get__(self)
        self.fill_plotsettings_frame = gui_layout.fill_plotsettings_frame.__get__(self)

        self.setup_ui()

        # self.canvas.get_tk_widget().bind("<Configure>", self.on_resize)

        # self.freqselect_menu()
        # self.fill_simsettings_frame()
        # self.fill_plotsettings_frame()
        # self.update_3d()

    def update_param_label(self, event):
        self.labels['param_1'] = self.current_param1_label.get()
        self.labels['param_2'] = self.current_param2_label.get()
        self.grid_labels = [self.current_param1_label.get(), self.current_param2_label.get()]
        self.p1select_label.configure(text=self.current_param1_label.get())
        self.p2select_label.configure(text=self.current_param2_label.get())

    def update_sweep_params(self):
        p1_start = self.param1_start_var.get()
        p1_end = self.param1_end_var.get()
        p1_n = self.param1_n_var.get()

        p2_start = self.param2_start_var.get()
        p2_end = self.param2_end_var.get()
        p2_n = self.param2_n_var.get()


        if self.param1_mode.get() == 'lin':
            self.param1_values = np.linspace(p1_start, p1_end, p1_n,
                                             dtype=self.precision)
        else:
            if ((p1_start < 0) != (p1_end < 0)) :
                print ("can't log a pos/neg sweep my man")
            elif ((p1_start == 0) or (p1_end == 0)):
                print("Pretty hard to log a zero")
            elif ((p1_start < 0) and (p1_end < 0)):
                self.param1_values = -np.logspace(np.log10(np.abs(self.param1_start_var.get())),
                                            np.log10(np.abs(self.param1_end_var.get())),
                                            self.param1_n_var.get(),
                                            dtype=self.precision)
            else:
                self.param1_values = np.logspace(np.log10(self.param1_start_var.get()),
                                            np.log10(self.param1_end_var.get()),
                                            self.param1_n_var.get(),
                                            dtype=self.precision)

        if self.param2_mode.get() == 'lin':
            self.param2_values = np.linspace(p2_start, p2_end, p2_n,
                                             dtype=self.precision)
        else:
            if ((p2_start < 0) != (p2_end < 0)) :
                print ("can't log a pos/neg sweep my man")
            elif ((p2_start == 0) or (p2_end == 0)):
                print("Pretty hard to log a zero")
            elif ((p2_start < 0) and (p2_end < 0)):
                self.param2_values = -np.logspace(np.log10(np.abs(self.param2_start_var.get())),
                                            np.log10(np.abs(self.param2_end_var.get())),
                                            self.param2_n_var.get(),
                                            dtype=self.precision)
            else:
                self.param2_values = np.logspace(np.log10(self.param2_start_var.get()),
                                            np.log10(self.param2_end_var.get()),
                                            self.param2_n_var.get(),
                                            dtype=self.precision)

        self.grid_values = [(p1, p2) for p1 in self.param1_values for p2 in self.param2_values]
        self.generate_index_map()


    def update_local_constants(self):
        for key, var in self.constant_vars.items():
            self.local_constants[key] = var.get()

    def update_solve_params(self):
        self.fs = self.fs_var.get()
        self.duration = self.duration_var.get()
        self.step_size = self.step_size_var.get()
        self.warmup_time = self.warmup_var.get()
        self.t = np.linspace(0,  self.duration -  1/self.fs, int( self.duration * self.fs))

    def solve_ode(self):
        self.update_local_constants()
        self.update_solve_params()
        self.update_sweep_params()
        self.update_param_label(None)

        self.solved_ODE.system.update_constants(self.local_constants)
        self.solved_ODE.fs = self.fs
        self.solved_ODE.step_size = self.step_size
        self.solved_ODE.duration = self.duration
        self.solved_ODE.noise_sigmas = np.asarray([item.get() for key, item in self.sigma_vars.items()] ,dtype=self.precision)

        self.inits = np.asarray([item.get() for key, item in self.init_vars.items()] ,dtype=self.precision)

        self.solved_ODE.build_kernel()
        self.solved_ODE.euler_maruyama(self.inits,
                            self.duration,
                            self.step_size,
                            self.fs,
                            self.grid_labels,
                            self.grid_values,
                            warmup_time=self.warmup_time)
        self.solved_ODE.get_fft()

        self.update_combobox_values(self.p1select_dd, self.param1_values)
        self.update_combobox_values(self.p2select_dd, self.param2_values)


    def select_frequency(self, event):
        # Get the selected item
        selected_index = self.listbox.curselection()
        if selected_index:
            self.freq_index = selected_index
            self.update_3d()
            self.freqselect_window.destroy()

    def on_resize(self, event):
        timetoupdate = ((time() - self.resize_time) > 1)
        if ((event.width, event.height) != self.last_winsize) and (timetoupdate):
            # self.canvas.get_tk_widget().config(width=event.width, height=event.height)
            # self.last_winsize = (event.width, event.height)
            # dpi = self.root.winfo_fpixels('1i')
            # self.fig.set_figheight(event.height * 2 / dpi)
            # self.fig.set_figwidth(event.width / dpi)
            # self.canvas.draw()

            self.update_all_fonts(self.root, event.height // 100)


    def set_cell_weights(self, widget, weight=1):
        """ Set weight of all cells in tkinter grid to 1 so that they stretch """
        for col_num in range(widget.grid_size()[0]):
            widget.columnconfigure(col_num, weight=weight)
        for row_num in range(widget.grid_size()[1]):
            widget.rowconfigure(row_num, weight=weight)

    def set_fixed_param(self):
        self.fixed_param = self.fix_param.get()

    def update_selected_params(self, event):
        self.param_1_val = self.param1_var.get()
        self.param_2_val = self.param2_var.get()
        self.param_index = self.get_param_index((self.param_1_val, self.param_2_val))
        self.single_state = self.solved_ODE.output_array[:, self.param_index, :]
        self.update_single_plot()
        self.update_3d()

    def update_axes(self):
        selection = self.grid_or_set.get()

        for ax in self.ax:
            ax.cla()
            self.fig.delaxes(ax)

        self.ax = []

        if selection == 'grid':
            self.xax = scene.AxisWidget(pos=[[-0.5, -0.5], [0.5, -0.5]], tick_direction=(0, -1),
                             font_size=16, axis_color='k', tick_color='k', text_color='k',
                             parent=self.canvas.view.scene)
            self.xax.transform = scene.STTransform(translate=(0, 0, -0.2))

            self.yax = scene.AxisWidget(pos=[[-0.5, -0.5], [-0.5, 0.5]], tick_direction=(-1, 0),
                             font_size=16, axis_color='k', tick_color='k', text_color='k',
                             parent=self.view.scene)
            self.yax.transform = scene.STTransform(translate=(0, 0, -0.2))

            # Add a 3D axis to keep us oriented
            self.axis = scene.visuals.XYZAxis(parent=self.view.scene)

            self.update_3d()

        # if selection == 'single':
        #     if self.singleplot_style_var.get() == 'time':
        #         self.ax = [self.fig.add_subplot(4, 1, 1)]
        #         self.ax = [self.ax[0],
        #                    self.fig.add_subplot(4, 1, 2, sharex=self.ax[0]),
        #                    self.fig.add_subplot(4, 1, 3, sharex=self.ax[0]),
        #                    self.fig.add_subplot(4, 1, 4, sharex=self.ax[0])]
        #         self.update_single_plot()
        #     else:
        #         self.ax = [self.fig.add_subplot(111)]
        #         self.update_single_plot()

    def generate_index_map(self):
        """Set up a dict mapping paramater sets to their index in the output
        array, cutting some compute time when populating the z mesh for plotting.

        Saves results to self.param_index_map
        """
        self.param_index_map = {}
        for idx, (p1, p2) in enumerate(self.grid_values):
            p1_round = self.precision(round_sf(p1, self.sort_sf))
            p2_round = self.precision(round_sf(p2, self.sort_sf))
            self.param_index_map[(p1_round, p2_round)] = idx

    def get_param_index(self, params):
        params = (self.precision(round_sf(params[0], self.sort_sf)),
                  self.precision(round_sf(params[1], self.sort_sf)))
        return self.param_index_map.get(params, None)

    def load_grid(self, xvals, yvals, surf_type):

        z_selection = self.z_axis_var.get()
        #Get 2D array to draw z values from depending on z selection
        if z_selection == 'fft_mag':
            working_zview = np.abs(self.solved_ODE.fft_array[self.plotstate, :, :]).T
        elif z_selection == 'fft_phase':
            working_zview = np.angle(self.solved_ODE.fft_array[self.plotstate, :, :]).T
            # working_zview = np.where(working_zview < 0, working_zview+2*np.pi, working_zview)
        elif z_selection == 'time-domain':
            working_zview = self.solved_ODE.output_array[self.plotstate, :, :].T

        #Turn x and Y into grid, unsure whether this could be done in an earlier function
        X, Y = np.meshgrid(xvals, yvals)
        Z = np.zeros_like(X, dtype=self.precision)


        if surf_type == 'grid':

            #Slice or transform 2D views to 1d selection for 2-param grids
            if z_selection == 'time-domain':
                z_values = np.sqrt(np.mean(working_zview**2, axis=0))
            else:
                z_values = working_zview[self.freq_index,:].T

            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    Z[i, j] = z_values[self.get_param_index((X[i, j], Y[i, j]))]

        elif surf_type == 'progression':
            if self.fixed_var == 'param_1':
                fixed_val = self.param_1_val
                working_zview = working_zview[slice(self.min_x_index,self.max_x_index),slice(self.min_y_index,self.max_y_index)]
                for i, yval in enumerate(yvals):
                    param_index = self.get_param_index((fixed_val, yval))
                    if param_index is not None:
                        Z[i,:] = working_zview[:,param_index]

            elif self.fixed_var == 'param_2':
                fixed_val = self.param_2_val
                working_zview = working_zview[slice(self.min_x_index,self.max_x_index),slice(self.min_y_index,self.max_y_index)]
                for i, yval in enumerate(yvals):
                    param_index = self.get_param_index((yval, fixed_val))
                    if param_index is not None:
                        Z[i,:] = working_zview[:,param_index]

        return X, Y, Z

    def update_3d(self):

        self.update_plot_slice()

        if self.grid_or_set.get() == 'grid':

            valid_combo = False
            (x_request, y_request) = (self.x_axis_var.get(), self.y_axis_var.get())
            z_request = self.z_axis_var.get()

            if (x_request, y_request) == ('param_1', 'param_2'):
                xvals = round_list_sf(self.param1_values, self.sort_sf)
                yvals = round_list_sf(self.param2_values, self.sort_sf)
                self.fixed_var = 'z_slice'
                surf_type = 'grid'
                valid_combo = True

            elif (x_request, y_request) == ('param_2', 'param_1'):
                xvals = round_list_sf(self.param1_values, self.sort_sf)
                yvals = round_list_sf(self.param2_values, self.sort_sf)
                self.fixed_var = 'z_slice'
                surf_type = 'grid'
                valid_combo = True

            elif (x_request, y_request) == ('time', 'param_1'):
                xvals = self.t
                yvals = round_list_sf(self.param1_values, self.sort_sf)

                self.fixed_var = 'param_2'
                surf_type = 'progression'
                valid_combo = True

            elif (x_request, y_request) == ('time', 'param_2'):
                xvals = self.t
                yvals = round_list_sf(self.param2_values, self.sort_sf)
                self.fixed_var = 'param_1'
                surf_type = 'progression'
                valid_combo = True

            elif (x_request, y_request) == ('freq', 'param_1'):
                xvals = self.solved_ODE.f[:]
                yvals = round_list_sf(self.param1_values, self.sort_sf)
                self.fixed_var = 'param_2'
                surf_type = 'progression'
                valid_combo = True

            elif (x_request, y_request) == ('freq', 'param_2'):
                xvals = self.solved_ODE.f[:]
                yvals = round_list_sf(self.param2_values, self.sort_sf)
                self.fixed_var = 'param_1'
                surf_type = 'progression'
                valid_combo = True

            else:

                print("This x-y combo doesn't make sense man")

            if valid_combo == True:
                xvals = xvals[slice(self.min_x_index, self.max_x_index, None)]
                yvals = yvals[slice(self.min_y_index,self.max_y_index, None)]
                X, Y, Z = self.load_grid(xvals, yvals, surf_type)


                xlabel = self.labels[x_request]
                ylabel = self.labels[y_request]
                zlabel = self.labels[z_request]

                self.update_surface(X, Y, Z, xlabel, ylabel, zlabel)

    def update_plot_slice(self, *args):
        xvar = self.x_axis_var.get()
        yvar = self.y_axis_var.get()

        self.min_x_index = self.get_slice_index(self.xslice_min_var.get(), xvar)
        self.max_x_index = self.get_slice_index(self.xslice_max_var.get(), xvar)
        self.min_y_index = self.get_slice_index(self.yslice_min_var.get(), yvar)
        self.max_y_index = self.get_slice_index(self.yslice_max_var.get(), yvar)

    def get_slice_index(self, value, variable):
        if variable == 'freq':
            index = np.argmin(np.abs(self.solved_ODE.f - value))if value < np.amax(self.solved_ODE.f) else None
        elif variable == 'time':
            index = np.argmin(np.abs(self.t - value))if value < np.amax(self.t) else None
        elif variable == 'param_1':
            index = np.argmin(np.abs(self.param1_values - value))if value < np.amax(self.param1_values) else None
        elif variable == 'param_2':
            index = np.argmin(np.abs(self.param2_values - value)) if value < np.amax(self.param2_values) else None
        else:
            print("That's a bad slice")

        return index

    def update_zlim(self, *args):
       self.zslicemin = self.zslice_min_var.get()
       self.zslicemax = self.zslice_max_var.get()
       for ax in self.ax:
           ax.set_zlim((self.zslicemin, self.zslicemax))
       self.canvas.draw()

    def generate_log_ticks(self, data):


        min_data = np.min(data)
        max_data = np.max(data)
        if min_data <= 0 and max_data <= 0:
            data = -data
            min_data = np.min(data)
            max_data = np.max(data)
        if min_data <= 0:
            return data, data

        min_exp = int(np.floor(np.log10(min_data)))
        max_exp = int(np.ceil(np.log10(max_data)))
        tick_values = []

        for exp in range(min_exp, max_exp + 1):
            tick_values.extend(np.arange(1, 10) * 10**exp)

        tick_values = round_list_sf(tick_values, self.plot_sf)
        tick_values = np.array(tick_values)
        tickv_alues = tick_values[(tick_values >= min_data) & (tick_values <= max_data)]

        return tick_values, np.log10(tick_values)

    def update_surface(self, X, Y, Z, xlabel, ylabel, zlabel):
        # self.ax[0].cla()

        if self.xscale.get() == 'log':
            xticklabels, xtickvals = self.generate_log_ticks(X)
            X = np.log10(X)
            xlabel = "log10 " + xlabel

        if self.yscale.get() == 'log':
            yticklabels, ytickvals = self.generate_log_ticks(Y)
            Y = np.log10(Y)
            ylabel = "log10 " + ylabel

        if self.zscale.get() == 'log':
            zticklabels, ztickvals = self.generate_log_ticks(Z)
            Z = np.log10(Z)
            zlabel = "log10 " + zlabel

        # self.ax[0].plot_surface(X, Y, Z, cmap='viridis', rcount=X.shape[0], ccount=Y.shape[1])

        # if self.xscale.get() == 'log':
        #     self.ax[0].set_xticks(xtickvals)
        #     self.ax[0].set_xticklabels(xticklabels)

        # if self.yscale.get() == 'log':
        #     self.ax[0].set_yticks(ytickvals)
        #     self.ax[0].set_yticklabels(yticklabels)

        # if self.zscale.get() == 'log':
        #     self.ax[0].set_zticks(ztickvals)
        #     self.ax[0].set_zticklabels(zticklabels)

        # self.ax[0].set_xlabel(xlabel, labelpad=20)
        # self.ax[0].set_ylabel(ylabel, labelpad=20)
        # self.ax[0].set_zlabel(zlabel, labelpad=20)

        self.view.plot(X, Y, Z, xlabel, ylabel, zlabel)
        app.process_events()
# p1._update_data()  # cheating.
cf = scene.filters.ZColormapFilter('fire', zrange=(z.max(), z.min()))
p1.attach(cf)
#
        # self.fig_frame.canvas.show()

        # self.canvas.draw()


    def update_single_plot(self, **spectral_params):
        # if self.grid_or_set.get() == 'single':
            # if self.singleplot_style_var.get() == "time":
            #     for ax in self.ax:
            #         ax.cla()
            #     self.ax[0].plot(self.t, self.single_state[0,:], label = 'Displacement (unfiltered)')
            #     self.ax[0].plot(self.t, self.single_state[4,:], label = 'Displacement (HPF)')
            #     # self.ax[0].plot(self.t, self.reference * self.solved_ODE.csystem.onstants_dict['rhat'], label="Piezo input")
            #     self.ax[0].set_ylim([-5,5])
            #     self.ax[0].legend(loc="upper right")

            #     self.ax[1].plot(self.t, self.single_state[3,:], label = 'Control')
            #     self.ax[1].plot(self.t, self.single_state[3,:]**2, label = 'Control squared')
            #     self.ax[1].legend(loc="upper right")

            #     self.ax[2].plot(self.t, self.single_state[2,:], label='Temperature')
            #     self.ax[2].legend(loc="upper right")

            #     heat_in = self.solved_ODE.system.constants_dict['gamma']*self.single_state[3,:]**2;
            #     heat_out = self.single_state[2,:]*self.solved_ODE.system.constants_dict['beta']
            #     self.ax[3].plot(self.t, heat_in, label='Heat in')
            #     self.ax[3].plot(self.t, heat_out, label='Heat out')
            #     self.ax[3].legend(loc="upper right")

            if self.singleplot_style_var.get() == "spec":
                self.ax[0].cla()

                spectr_fs = 2*np.pi*self.solved_ODE.fs
                if 'windowlength' not in spectral_params:
                    windowlength = min(int(np.floor(30*2*np.pi * self.solved_ODE.fs)), int(len(self.t) / 8))
                if 'hop' not in spectral_params:
                    hop = int(np.floor(windowlength/4))
                if 'windowlength' not in spectral_params:
                    nfft = int(windowlength*4)
                if 'window' not in spectral_params:
                    window = hann(windowlength, sym=False)
                else:
                    window = window(windowlength, sym=False)

                nperseg = windowlength

                f, t, Sx = stft(self.single_state[self.plotstate,:],
                                self.spectr_fs, nfft=nfft, noverlap=hop,
                                nperseg = nperseg, return_onesided=True)

            #     self.ax[0].pcolormesh(t.get(), f.get(), np.abs(Sx.get()), shading='gouraud')
            #     self.ax[0].set_ylim([0,2.5])
            #     self.ax[0].set_ylabel("Frequency")
            #     self.ax[0].set_xlabel("Time")

            self.canvas.draw()

    def update_combobox_values(self, combobox, new_values):
        """
        Update the values of the given combobox.

        :param combobox: The tk.Combobox widget to update.
        :param new_values: A list of new values to set in the combobox.
        """
        if isinstance(new_values, np.ndarray):
            new_values = new_values.tolist()

        combobox['values'] = new_values
        combobox.current(0)

    def update_label_text(self, label, new_text):
        """
        Update the text of the given label.

        :param label: The tk.Label widget to update.
        :param new_text: The new text to set in the label.
        """
        label.config(text=new_text)

    def update_font_size(self, widget, new_font_size):
        """
        Update the font size of a given widget.

        :param widget: The tk widget to update.
        :param new_font_size: The new font size to set.
        """
        current_font = tkFont.nametofont(widget.cget("font"))
        current_font.config(size=new_font_size)
        widget.config(font=current_font)

    def update_all_fonts(self, parent, new_font_size):
        """
        Update the font sizes of all child widgets of a given parent widget.

        :param parent: The parent widget containing child widgets.
        :param new_font_size: The new font size to set for all child widgets.
        """
        for child in parent.winfo_children():
            try:
                self.update_font_size(child, new_font_size)
            except tk.TclError:
                pass
            if isinstance(child, tk.Frame) or isinstance(child, ttk.Frame):
                self.update_all_fonts(child, new_font_size)

# Test code
if __name__ == "__main__":

 #%%
    #Setting up grid of params to simulate with
    precision = np.float32
    a_gains = np.asarray([i * 0.001 + 1 for i in range(-100, 100)], dtype=precision)
    b_params = np.asarray([i * 0.0001 + 0.005 for i in range(-50, 50)], dtype=precision)
    grid_params = [(a, b) for a in a_gains for b in b_params]
    grid_labels = ['omega_forcing', 'rhat']
    step_size = precision(0.001)
    fs = precision(1.0)
    duration = precision(1000.0)
    sys = diffeq_system(a=5, b=-0.1, precision=precision)
    inits = np.asarray([0.0, 0, 0.0, 0, 0.0], dtype=precision)

    ODE = CUDA_ODE(sys, precision=precision)
    ODE.build_kernel()
    ODE.euler_maruyama(inits,
                        duration,
                        step_size,
                        fs,
                        grid_labels,
                        grid_params,
                        warmup_time=000.0)
    ODE.get_fft()
#%%
    plotter = Gridplotter(ODE, grid_params, precision = precision)
    plotter.root.mainloop()
