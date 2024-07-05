import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np
from diffeq_system import diffeq_system #Only needed for test code
from CUDA_ODE import CUDA_ODE
from cupyx.scipy.signal import stft
from scipy.signal.windows import hann


class Gridplotter:
    def __init__(self, root, solved_ODE, grid_values):
        self.root = root
        self.grid_values = grid_values
        self.solved_ODE = solved_ODE
        self.selected_a = tk.DoubleVar()
        self.selected_b = tk.DoubleVar()
        self.chosen_frequency_var = tk.DoubleVar()
        self.plot_type = tk.StringVar(value="Grid Gain @w")
        
     

        self.t = np.linspace(0,  self.solved_ODE.duration -  1/self.solved_ODE.fs, int( self.solved_ODE.duration * self.solved_ODE.fs))
        self.spectr_fs = np.floor(2*np.pi*self.solved_ODE.fs) # Correct for normalised freq (bring it to 1)
        self.plotstate = 4
        self.current_button = "Grid Gain @w"
        self.freq_index = 0
        self.current_a_val = 0
        self.current_b_val = 0
        self.selected_index = 0
        
        self.a_values = np.unique([param[0] for param in self.grid_values])
        self.b_values = np.unique([param[1] for param in self.grid_values])
        self.param_index_map = {param: idx for idx, param in enumerate(self.grid_values)}
        self.setup_ui()
        # self.update_frequency(None)
        
    def setup_ui(self): 
        self.root.title("Gridsearch Results")
        self.root.geometry("3840x2160")

        # Create plot canvas
        self.fig = Figure(figsize=(12, 8))
        self.ax = [self.fig.add_subplot(111, projection='3d')]
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().grid(row=1, column=0, columnspan=4, rowspan=9, sticky='nsew')
        self.root.bind("<Configure>", self.resize_canvas)
        
        self.toolbar = NavigationToolbar2Tk(self.canvas, root, pack_toolbar=False)
        self.toolbar.grid(row=0,column=0, columnspan=4)
        self.toolbar.update()
        
        # Create control frame on the rightmost column
        self.control_frame = ttk.Frame(self.root)
        self.control_frame.grid(row=0, column=4, rowspan=10, sticky='nsew')

        self.set_cell_weights(self.root, weight=4)
        self.root.columnconfigure(4,weight=1)
        # Create 2x2 grid of radio buttons for plot type
        ttk.Radiobutton(self.control_frame, text="Grid Gain @w", variable=self.plot_type, value="Grid Gain @w", command=self.plotselect).grid(row=0, column=0, padx=10, pady=10)
        ttk.Radiobutton(self.control_frame, text="Grid Time Domain", variable=self.plot_type, value="Grid Time Domain", command=self.plotselect).grid(row=0, column=1, padx=10, pady=10)
        ttk.Radiobutton(self.control_frame, text="Individual Spectrogram", variable=self.plot_type, value="Individual Spectrogram", command=self.plotselect).grid(row=1, column=0, padx=10, pady=10)
        ttk.Radiobutton(self.control_frame, text="Individual Time Domain", variable=self.plot_type, value="Individual Time Domain", command=self.plotselect).grid(row=1, column=1, padx=10, pady=10)
        ttk.Radiobutton(self.control_frame, text="Grid Phase @w", variable=self.plot_type, value="Grid Phase @w", command=self.plotselect).grid(row=2, column=0, padx=10, pady=10)

        # Create dropdowns for parameter selection
        ttk.Label(self.control_frame, text="Select a:").grid(row=3, column=0, padx=10, pady=10)
        self.dropdown_a = ttk.Combobox(self.control_frame,
                                       textvariable=self.selected_a,
                                       values=sorted(list(set([param[0] for param in self.grid_values]))))
        self.dropdown_a.grid(row=3, column=1, padx=10, pady=10)
        self.dropdown_a.bind("<<ComboboxSelected>>", self.update_a_b)

        ttk.Label(self.control_frame, text="Select b:").grid(row=4, column=0, padx=10, pady=10)
        self.dropdown_b = ttk.Combobox(self.control_frame
                                       , textvariable=self.selected_b,
                                       values=sorted(list(set([param[1] for param in self.grid_values]))))
        self.dropdown_b.grid(row=4, column=1, padx=10, pady=10)
        self.dropdown_b.bind("<<ComboboxSelected>>", self.update_a_b)
        
        ttk.Label(self.control_frame, text="Frequency:").grid(row=5, column=0, padx=10, pady=10)
        self.button = ttk.Button(self.control_frame, text="Open Selection window", command=self.open_dropdown)
        self.button.grid(row = 5, column = 1, padx=10, pady=10)
        
        self.dropdown_freq = ttk.Combobox(self.control_frame, 
                                          textvariable=self.chosen_frequency_var,
                                          values=self.solved_ODE.f)


         # Create Fix parameter radio buttons
        ttk.Label(self.control_frame, text="Fix:").grid(row=5, column=0, columnspan=2, padx=10, pady=10)
        self.fix_param = tk.StringVar(value="a")
        ttk.Radiobutton(self.control_frame, text="a", variable=self.fix_param,
                        value="a", command=self.set_fixed_param).grid(row=6, column=0, padx=10, pady=10)
        ttk.Radiobutton(self.control_frame, text="b", variable=self.fix_param,
                        value="b", command=self.set_fixed_param).grid(row=6, column=1, padx=10, pady=10)
        
        self.set_cell_weights(self.control_frame)
        
    def open_dropdown(self):
        # Create a new Toplevel window
        self.dropdown_window = tk.Toplevel(self.root)
        self.dropdown_window.title("Select a frquency")

        # Create a Listbox with a Scrollbar
        self.frame = ttk.Frame(self.dropdown_window)
        self.frame.pack(fill=tk.BOTH, expand=True)

        self.listbox = tk.Listbox(self.frame, height=20)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.scrollbar = ttk.Scrollbar(self.frame, orient=tk.VERTICAL, command=self.listbox.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.listbox.configure(yscrollcommand=self.scrollbar.set)

        # Populate the Listbox with the data
        for item in self.solved_ODE.f:
            self.listbox.insert(tk.END, item)

        # Bind the selection event
        self.listbox.bind("<<ListboxSelect>>", self.on_select)

    def on_select(self, event):
        # Get the selected item
        selected_index = self.listbox.curselection()
        if selected_index:
            self.freq_index = selected_index  
            self.update_frequency(None)
            # Close the dropdown window
            self.dropdown_window.destroy()

    def resize_canvas(self, event):
    # Resize the canvas to match the window size
        self.canvas.get_tk_widget().config(width=event.width * 3/4, height=event.height)
        self.canvas.draw()
        
    def set_cell_weights(self, widget, weight=1):
        """ Set weight of all cells in tkinter grid to 1 so that they stretch """
        for col_num in range(widget.grid_size()[0]):
            widget.columnconfigure(col_num, weight=weight)
        for row_num in range(widget.grid_size()[1]):
            widget.rowconfigure(row_num, weight=weight)

    
    def update_frequency(self, event):
        if self.current_button == "Grid Gain @w":
            self.update_gaingrid()
        if self.current_button == "Grid Phase @w":
            self.update_phasegrid()
            
    def set_fixed_param(self):
        self.fixed_param = self.fix_param.get()
        
    def update_a_b(self, event):
        self.current_a_val = self.selected_a.get()
        self.current_b_val = self.selected_b.get()
        self.param_index = self.grid_values.index((self.current_a_val,
                                                              self.current_b_val))
        self.single_state = self.solved_ODE.output_array[:, self.param_index, :].get()
        if self.current_button == "Grid Time Domain":
            self.update_timegrid()
        else:
            self.update_single_plot()
    

    def get_param_index(self, params):
        return self.param_index_map.get(params, None)
    
    def load_timegrid(self):
        if self.fixed_param == 'a':
            fixed_value = self.current_a_val
            A, T = np.meshgrid(self.b_values, self.t)
            Z = np.zeros_like(A, dtype=np.float64)
            for i, b_val in enumerate(self.b_values):
                param_index = self.get_param_index((fixed_value, b_val))
                if param_index is not None:
                    Z[:, i] = self.solved_ODE.time_friendly_array[self.plotstate, param_index, :]
        else:
            fixed_value = self.current_b_val
            A, T = np.meshgrid(self.a_values, self.t)
            Z = np.zeros_like(A, dtype=np.float64)
            for i, a_val in enumerate(self.a_values):
                param_index = self.get_param_index((a_val,fixed_value))
                if param_index is not None:
                    Z[:, i] = self.solved_ODE.time_friendly_array[self.plotstate, param_index, :]
            
        return A, T, Z
    
    # @timing
    def update_timegrid(self):
        self.ax[0].cla()  # Clear the previous plot
        
        A, T, Z = self.load_timegrid()

        self.ax[0].plot_surface(A, T, Z, cmap='viridis')
        self.ax[0].set_xlabel('Parameter ' + ('b' if self.fix_param.get() == 'a' else 'a'))
        self.ax[0].set_ylabel('Time')
        self.ax[0].set_zlabel('Output Value')
        self.canvas.draw()
        
    def update_gaingrid(self):
        # Clear previous plot
        self.ax[0].cla() 

        a_values = np.unique([param[0] for param in self.grid_values])
        b_values = np.unique([param[1] for param in self.grid_values])
        
        A, B = np.meshgrid(a_values, b_values)
        
        
        # Get Z values for the specific frequency and state
        z_values = np.abs(self.solved_ODE.fft_array[self.plotstate, :, self.freq_index]).T
        
        #Map (a, b) tuples to Z values
        z_dict = {self.grid_values[i]: z_values[i] for i in range(len(self.grid_values))}
        
        # Step 6: Initialize the Z matrix with the same shape as the meshgrid
        Z = np.zeros_like(A, dtype=np.float64)
        
        # Step 7: Fill the Z matrix by looking up the corresponding (a, b) tuple
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                Z[i, j] = z_dict[(A[i, j], B[i, j])]

        self.ax[0].plot_surface(A, B, Z, cmap='viridis')
        self.ax[0].set_xlabel('Parameter a')
        self.ax[0].set_ylabel('Parameter b')
        self.ax[0].set_zlabel('Magnitude at chosen frequency')
        
        self.canvas.draw()
        
    def update_phasegrid(self):
        # Clear previous plot
        self.ax[0].cla() 

        a_values = np.unique([param[0] for param in self.grid_values])
        b_values = np.unique([param[1] for param in self.grid_values])
        
        A, B = np.meshgrid(a_values, b_values)
        
        
        # Get Z values for the specific frequency and state
        z_values = np.angle(self.solved_ODE.fft_array[self.plotstate, :, self.freq_index].T)
        
        #Map (a, b) tuples to Z values
        z_dict = {self.grid_values[i]: z_values[i] for i in range(len(self.grid_values))}
        
        # Step 6: Initialize the Z matrix with the same shape as the meshgrid
        Z = np.zeros_like(A, dtype=np.float64)
        
        # Step 7: Fill the Z matrix by looking up the corresponding (a, b) tuple
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                Z[i, j] = z_dict[(A[i, j], B[i, j])]

        self.ax[0].plot_surface(A, B, Z, cmap='viridis')
        self.ax[0].set_xlabel('Parameter a')
        self.ax[0].set_ylabel('Parameter b')
        self.ax[0].set_zlabel('Magnitude at chosen frequency')
        
        self.canvas.draw()

    
    def update_single_plot(self, **spectral_params):
              
        if self.current_button == "Individual Time Domain":   
            for ax in self.ax:
                ax.cla()
            self.ax[0].plot(self.t, self.single_state[:,0], label = 'Displacement (unfiltered)')
            self.ax[0].plot(self.t, self.single_state[:,4], label = 'Displacement (HPF)')
            self.ax[0].plot(self.t, self.reference * self.solved_ODE.constants[9], label="Piezo input")
            self.ax[0].set_ylim([-5,5])
            self.ax[0].legend(loc="upper right")
    
            self.ax[1].plot(self.t, self.single_state[:,3], label = 'Control')
            self.ax[1].plot(self.t, self.single_state[:,3]**2, label = 'Control squared')
            self.ax[1].legend(loc="upper right")
    
            self.ax[2].plot(self.t, self.single_state[:,2], label='Temperature')
            self.ax[2].legend(loc="upper right")
    
            heat_in = self.solved_ODE.constants[2]*self.single_state[:,3]**2;
            heat_out = self.single_state[:,2]*self.solved_ODE.constants[1]
            self.ax[3].plot(self.t, heat_in, label='Heat in')
            self.ax[3].plot(self.t, heat_out, label='Heat out')
            self.ax[3].legend(loc="upper right")

        elif self.current_button == "Individual Spectrogram":
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
                
            nperseg = max(256, windowlength)
        
            f, t, Sx = stft(self.single_state[:,self.plotstate],
                            self.spectr_fs, nfft=nfft, noverlap=hop,
                            nperseg = nperseg, return_onesided=True)
        
            self.ax[0].pcolormesh(t.get(), f.get(), np.abs(Sx.get()), shading='gouraud')
            self.ax[0].set_ylim([0,2.5])
            self.ax[0].set_ylabel("Frequency")
            self.ax[0].set_xlabel("Time")

        self.canvas.draw()
        
    def plotselect(self):
        self.current_button = self.plot_type.get()
        for ax in self.ax:
            ax.cla()  
            self.fig.delaxes(ax) 
    
        self.ax = []
       
        if self.current_button == "Individual Time Domain":
            # setup first axis first, so that it's in the list for the others to share it's x
            self.ax = [self.fig.add_subplot(4, 1, 1)]
            self.ax = [self.ax[0],
                       self.fig.add_subplot(4, 1, 2, sharex=self.ax[0]),
                       self.fig.add_subplot(4, 1, 3, sharex=self.ax[0]),
                       self.fig.add_subplot(4, 1, 4, sharex=self.ax[0])]
            self.update_a_b(None)
            
        elif self.current_button == "Grid Time Domain":
            self.set_fixed_param()
            self.ax = [self.fig.add_subplot(111, projection='3d')]
            self.ax[0].set_xlabel('Parameter a')
            self.ax[0].set_ylabel('Parameter b')
            self.ax[0].set_zlabel('RMS Value')
            self.update_timegrid()

        elif self.current_button == "Individual Spectrogram":
            self.ax = [self.fig.add_subplot(111)]
            self.update_a_b(None)

        elif self.current_button == "Grid Gain @w":
            self.ax = [self.fig.add_subplot(111, projection='3d')]

            self.update_gaingrid()
            
        elif self.current_button == "Grid Phase @w":
            self.ax = [self.fig.add_subplot(111, projection='3d')]

            self.update_phasegrid()




# Test code
if __name__ == "__main__":

    
    # #Setting up grid of params to simulate with
    a_gains = np.asarray([i * 0.01 for i in range(-128, 128)], dtype=np.float64)
    b_params = np.asarray([i * 0.02 for i in range(-64, 64)], dtype=np.float64)
    grid_params = [(a, b) for a in a_gains for b in b_params]
    grid_labels = ['a', 'b']
    step_size = 0.001
    fs = 10
    duration = np.float64(100)
    sys = diffeq_system()
    inits = np.asarray([1.0, 0, 1.0, 0, 1.0], dtype=np.float64)
    
    ODE = CUDA_ODE(sys)
    ODE.build_kernel()
    ODE.euler_maruyama(inits,
                       duration,
                       step_size,
                       fs,
                       grid_labels,
                       grid_params,
                       warmup_time=200.0)
    ODE.get_fft(fs=10.0)

    root = tk.Tk()
    plotter = Gridplotter(root, ODE, grid_params)
    root.mainloop()
