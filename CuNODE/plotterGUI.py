import numpy as np
from diffeq_system import diffeq_system, system_constants 
from CUDA_ODE import CUDA_ODE
from cupyx.scipy.signal import stft
from scipy.signal.windows import hann
import gui_layout

class Gridplotter:
    def __init__(self, ODE, grid_values):
        self.grid_values = grid_values
        self.solved_ODE = ODE
        self.local_constants = self.solved_ODE.system.constants_dict.copy()
        
        self.t = np.linspace(0,  self.solved_ODE.duration -  1/self.solved_ODE.fs, int( self.solved_ODE.duration * self.solved_ODE.fs))
        self.spectr_fs = np.floor(2*np.pi*self.solved_ODE.fs) # Correct for normalised freq (bring it to 1)
        self.plotstate = 4
        self.freq_index = 0
        self.param_1_val = 0
        self.param_2_val = 0
        self.selected_index = 0
        self.max_plot_f = 1.05
        self.min_plot_f = 0.95

        
        self.param1_values = np.unique([param[0] for param in self.grid_values])
        self.param2_values = np.unique([param[1] for param in self.grid_values])
        self.param_index_map = {param: idx for idx, param in enumerate(self.grid_values)}

        self.setup_ui = gui_layout.setup_ui.__get__(self)
        self.freqselect_menu = gui_layout.freqselect_menu.__get__(self)
        self.fill_simsettings_frame = gui_layout.fill_simsettings_frame.__get__(self)
        self.fill_plotsettings_frame = gui_layout.fill_plotsettings_frame.__get__(self)
        
        self.setup_ui()
        # self.freqselect_menu()
        # self.fill_simsettings_frame()
        # self.fill_plotsettings_frame()
        self.update_3d(None)
        


    
    def update_param_label(self, event):
        self.param1_label = self.current_param1_label.get()
        self.param2_label = self.current_param2_label.get()

    def update_sweep_params(self):
        self.param1_values = np.linspace(self.param1_start_var.get(),
                                    self.param1_end_var.get(),
                                    self.param1_n_var.get(),
                                    dtype=np.float64)
        self.param2_values = np.linspace(self.param2_start_var.get(),
                                    self.param2_end_var.get(),
                                    self.param2_n_var.get(),
                                    dtype=np.float64)
        self.param_index_map = {param: idx for idx, param in enumerate(self.grid_values)}

        self.grid_values = [(p1, p2) for p1 in self.param1_values for p2 in self.param2_values]
        self.grid_labels = [self.param1_label, self.param2_label]
        
        #TODO: add to gui, and add noise sigmas too, both varying length
        self.inits = np.asarray([1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float64)
        
        self.solved_ODE.build_kernel()
        self.solved_ODE.euler_maruyama(self.inits,
                            self.duration,
                            self.step_size,
                            self.fs,
                            self.grid_labels,
                            self.grid_params,
                            warmup_time=5000.0) ##TODO: add to GUI
        self.solved_ODE.get_fft()
    
    def update_local_constants(self):
        for key, var in self.constant_vars.items():
            self.local_constants[key] = var.get()
            
    def update_solve_params(self):
        self.fs = self.fs_var.get()
        self.duration = self.duration.get()
        self.step_size = self.step_size.get()
        
    def solve_ode(self):
        self.update_local_constants()
        self.update_sweep_params()
        
        self.solved_ODE.system.update_constants(self.local_constants)
        self.solved_ODE.fs = self.fs
        self.solved_ODE.step_size = self.step_size
        self.solved_ODE.duration = self.duration        

        print("Solving ODE with current settings")

    def select_frequency(self, event):
        # Get the selected item
        selected_index = self.listbox.curselection()
        if selected_index:
            self.freq_index = selected_index  
            self.update_3d(None)
            self.freqselect_window.destroy()

    def resize_canvas(self, event):
    # Resize the canvas to match the window size
        self.canvas.get_tk_widget().config(width=event.width * 4/5, height=event.height*0.9)
        self.canvas.draw()
        
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
        self.param_index = self.grid_values.index((self.param_1_val,
                                                              self.param_2_val))
        self.single_state = self.solved_ODE.time_friendly_array[:, self.param_index, :]
        self.update_single_plot()
        self.update_3d(None)
    
    def update_axes(self):
        selection = self.grid_or_set.get()
        
        for ax in self.ax:
            ax.cla()  
            self.fig.delaxes(ax) 
    
        self.ax = []
        
        if selection == 'grid':
            self.ax = [self.fig.add_subplot(111, projection='3d')]
            self.update_3d(None)
    
        if selection == 'single':
            if self.singleplot_style_var.get() == 'time':
                self.ax = [self.fig.add_subplot(4, 1, 1)]
                self.ax = [self.ax[0],
                           self.fig.add_subplot(4, 1, 2, sharex=self.ax[0]),
                           self.fig.add_subplot(4, 1, 3, sharex=self.ax[0]),
                           self.fig.add_subplot(4, 1, 4, sharex=self.ax[0])]
                self.update_single_plot()
            else:
                self.ax = [self.fig.add_subplot(111)]
                self.update_single_plot()

    def get_param_index(self, params):
        return self.param_index_map.get(params, None)
    
        
    def load_grid(self, xvals, yvals, surf_type):
        
        z_selection = self.z_axis_var.get()
        #Get 2D array to draw z values from depending on z selection
        if z_selection == 'fft_mag':
            if surf_type == 'progression':
                working_zview = np.log10(np.abs(self.solved_ODE.fft_array[self.plotstate, :, self.min_f_index:self.max_f_index]).T)

            else:
                working_zview = np.log10(np.abs(self.solved_ODE.fft_array[self.plotstate, :, :]).T)
        elif z_selection == 'fft_phase':
            if surf_type == 'progression':
                working_zview = np.angle(self.solved_ODE.fft_array[self.plotstate, :, self.min_f_index:self.max_f_index]).T
            else:
                working_zview = np.angle(self.solved_ODE.fft_array[self.plotstate, :, :]).T
            working_zview = np.where(working_zview < 0, working_zview+2*np.pi, working_zview)
        elif z_selection == 'time-domain':
            working_zview = self.solved_ODE.time_friendly_array[self.plotstate, :, :].T                
        
        #Turn x and Y into grid, unsure whether this could be done in an earlier function    
        X, Y = np.meshgrid(xvals, yvals)
        Z = np.zeros_like(X, dtype=np.float64)

        
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
                working_zview = working_zview[self.min_f_index:self.max_f_index,:]
                for i, yval in enumerate(yvals):
                    param_index = self.get_param_index((fixed_val, yval))
                    if param_index is not None:
                        Z[i,:] = working_zview[:,param_index]
                    
            elif self.fixed_var == 'param_2':
                fixed_val = self.param_2_val
                working_zview = working_zview[self.min_f_index:self.max_f_index,:]

                for i, yval in enumerate(yvals):
                    param_index = self.get_param_index((yval, fixed_val))
                    if param_index is not None:
                        Z[i,:] = working_zview[:,param_index]
        
        return X, Y, Z

        
    def update_3d(self, event):
        self.max_f_index = np.argmin(np.abs(self.solved_ODE.f - self.max_plot_f))
        self.min_f_index = np.argmin(np.abs(self.solved_ODE.f - self.min_plot_f))

        
        if self.grid_or_set.get() == 'grid':
            
            valid_combo = False
            (x_request, y_request) = (self.x_axis_var.get(), self.y_axis_var.get())
            z_request = self.z_axis_var.get()
            
            if (x_request, y_request) == ('param_1', 'param_2'):
                xvals = self.param1_values
                yvals = self.param2_values
                self.fixed_var = 'z_slice'
                surf_type = 'grid'
                valid_combo = True
                
            elif (x_request, y_request) == ('param_2', 'param_1'):
                xvals = self.param1_values
                yvals = self.param2_values
                self.fixed_var = 'z_slice'
                surf_type = 'grid'
                valid_combo = True
                
            elif (x_request, y_request) == ('time', 'param_1'):
                xvals = self.t
                yvals = self.param1_values
                self.fixed_var = 'param_2'
                surf_type = 'progression'
                valid_combo = True
           
            elif (x_request, y_request) == ('time', 'param_2'):
                xvals = self.t
                yvals = self.param2_values
                self.fixed_var = 'param_1'
                surf_type = 'progression'
                valid_combo = True
            
            elif (x_request, y_request) == ('freq', 'param_1'):
                xvals = self.solved_ODE.f[self.min_f_index:self.max_f_index]
                yvals = self.param1_values
                self.fixed_var = 'param_2'
                surf_type = 'progression'
                valid_combo = True
           
            elif (x_request, y_request) == ('freq', 'param_2'):
                xvals = self.solved_ODE.f[self.min_f_index:self.max_f_index]
                yvals = self.param2_values
                self.fixed_var = 'param_1'
                surf_type = 'progression'
                valid_combo = True
                
            else:
                
                print("This x-y combo doesn't make sense man")
                
            if valid_combo == True:
                X, Y, Z = self.load_grid(xvals, yvals, surf_type)
                
                self.update_surface(X, Y, Z, x_request, y_request, z_request)
        
    def update_surface(self, X, Y, Z, x_request, y_request, z_request):
        # Clear previous plot
        self.ax[0].cla() 
        self.ax[0].plot_surface(X, Y, Z, cmap='viridis')
        
        self.ax[0].set_xlabel(x_request)
        self.ax[0].set_ylabel(y_request)
        self.ax[0].set_zlabel(z_request)
        
        self.canvas.draw()
        
    def update_single_plot(self, **spectral_params):
        if self.grid_or_set.get() == 'single':
            if self.singleplot_style_var.get() == "time":   
                for ax in self.ax:
                    ax.cla()
                self.ax[0].plot(self.t, self.single_state[0,:], label = 'Displacement (unfiltered)')
                self.ax[0].plot(self.t, self.single_state[4,:], label = 'Displacement (HPF)')
                # self.ax[0].plot(self.t, self.reference * self.solved_ODE.csystem.onstants_dict['rhat'], label="Piezo input")
                self.ax[0].set_ylim([-5,5])
                self.ax[0].legend(loc="upper right")
        
                self.ax[1].plot(self.t, self.single_state[3,:], label = 'Control')
                self.ax[1].plot(self.t, self.single_state[3,:]**2, label = 'Control squared')
                self.ax[1].legend(loc="upper right")
        
                self.ax[2].plot(self.t, self.single_state[2,:], label='Temperature')
                self.ax[2].legend(loc="upper right")
        
                heat_in = self.solved_ODE.system.constants_dict['gamma']*self.single_state[3,:]**2;
                heat_out = self.single_state[2,:]*self.solved_ODE.system.constants_dict['beta']
                self.ax[3].plot(self.t, heat_in, label='Heat in')
                self.ax[3].plot(self.t, heat_out, label='Heat out')
                self.ax[3].legend(loc="upper right")
    
            if self.singleplot_style_var.get() == "spec":   
                self.ax[0].cla() 
                
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
            
                self.ax[0].pcolormesh(t.get(), f.get(), np.abs(Sx.get()), shading='gouraud')
                self.ax[0].set_ylim([0,2.5])
                self.ax[0].set_ylabel("Frequency")
                self.ax[0].set_xlabel("Time")
    
            self.canvas.draw()
        

# Test code
if __name__ == "__main__":

 #%%   
    #Setting up grid of params to simulate with
    a_gains = np.asarray([i * 0.001 + 1 for i in range(-100, 100)], dtype=np.float64)
    b_params = np.asarray([i * 0.0001 + 0.005 for i in range(-50, 50)], dtype=np.float64)
    grid_params = [(a, b) for a in a_gains for b in b_params]
    grid_labels = ['omega_forcing', 'rhat']
    step_size = 0.001
    fs = 1
    duration = np.float64(10000)
    sys = diffeq_system(a=10, b=-0.1)
    inits = np.asarray([0.0, 0, 0.0, 0, 0.0], dtype=np.float64)
    
    ODE = CUDA_ODE(sys)
    ODE.build_kernel()
    ODE.euler_maruyama(inits,
                        duration,
                        step_size,
                        fs,
                        grid_labels,
                        grid_params,
                        warmup_time=5000.0)
    ODE.get_fft()
#%% 
    plotter = Gridplotter(ODE, grid_params)
    plotter.root.mainloop()
