<<<<<<< HEAD
# CuNODE
**Cuda (through Numba) ODE integrators**

ODE solvers for large parallel operations (e.g. grid searches) where the benefits of running solvers in parallel outweighs the penalty of the slightly slower runtime of a single solve.

**Requirements:**
Hardware:
NVIDIA GPU, Compute Capability > 2.* (I think)
python 3.9 + (anything after 3.7 probably OK but completely untested) environment
CUDA toolkit 12.x
numpy, numba, matplotlib, scipy, cupy

**Intended functionality:**

- You provide a dxdt function (fill in the function template), and select an integration type, step size, duration, and output sample frequency
- You provide a vector of variables to run (e.g. a 2D grid search looks like ((0, 0), (0,1), (1,0), (1,1)))
- run euler, rk4, or other integration function
- reap rewards


**Getting Started:**

CUDA: go to https://developer.nvidia.com/cuda-toolkit and download the latest CUDA toolkit - I have tested with 12.5, 12.3.
Python: if you don't have python installed, download python 3.9 from https://www.python.org/downloads/release/python-390/
- I strongly recommend using a virtual environment, because things tend to break in python when you download a new library and it updates something unintended. You can create a "venv" for each project, which has it's own set of dependencies, for a small amount of extra admin on setup.
    - To do this, you want to create a "venvs" folder somewhere handy, like on C: drive.
    - navigate to this folder in "command prompt" - type: "C:" then "cd venvs" if you put it on c drive
    - in the command prompt type "python -m venv cunode"
        - this creates a folder called cunode, with your environment in it
    - whenever you want to install stuff on this version of python, navigate to the venvs/cunode/scripts folder in command prompt and type "activate". It will say (cunode) on the left of command prompt when you've successfully activated it, which tells you you're working in this environment
- Install dependencies: pip install numpy scipy numba cupy-cuda12x matplotlib cuda-python
    - If you want to use the Spyder IDE (it's like MATLAB), then also do: "pip install PyQt5 spyder-kernels"
    - download and install Spyder from https://www.spyder-ide.org/
    - When you open Spyder, go Tools-Preferences->Python interpreter -> Use the following Python Interpreter -> navigate to wherever your "python.exe" file is (if you followed the above instructions, C:/venvs/cunode/Scripts/python.exe)

- Install git for windows if you don't have it: https://git-scm.com/download/win
- set up a folder in your documents for cunode
- right click in this folder somewhere and select "open git bash here"
- type "git init"
- type "git clone git@github.com:ccam80/CuNODE.git"
- This hopefully downloads the current version into this folder

**How to run:**
*This stuff will change a bit as I tidy things up to make it easier to use!!!*
1. Go into "CuNODE/for_deletion".
2. Open CUDA_ODE.py, diffeq_system.py, and plotterGUI.py in your IDE.
3. save diffeq_system.py as "[name of your system].py"
4. Change "defaults" under "system_constants" (currently on line 36) to the constants and parameters of your system - feel free to use as many as you want, rather than having magic numbers in your dydt code, to allow you to mess around with parameters later.
5. Enter all the mathy bits in "dxdtfunc". Annoyingly, for now, all constants _must be referred to by their position in the dictionary_, - for example, in the example in "diffeq_system", 'beta' = constants[1] (as python starts counting from 0).
6. in CUDA_ODE.py change line 18 from "from diffeq_system import diffeq_system" to "from [name of your system] import diffeq_system".
7. Open plotterGUI
8. Make the same edit in line 2 (from [name of your system])
9. change the example grid labels (currently line 563) to the names of two of your constants (any two, it's easy to change later)
10. run plotterGUI.py
11. Push buttons.

**Things you can change (but not in the GUI right now):**
plotterGUI: self.plot_downsample selects how much we downsample the grid before plotting. The default (2) plots only half of the grid points, assuming that the surface changes gradually enough that this won't affect your data. Make it 1 for no downsampling (more lag), or higher for less lag. This doesn't affect and information you're saving, just the plotting.
plotterGUI: "precision" refers to whether we're using 32-bit floating point numbers or 64-bit floating point numbers. 64 bit numbers are more precise but about 4x slower. Check one run at 32 bits against a MATLAB reference - if it's close enough, then you're probably fine to keep going in 32 bits. If not, set precision = np.float64 (currently line 560) in plotterGUI.py.


**Roadmap:**
There is a running todo list in todo.txt that will change regularly. My next batch of tasks is to:
- Rewrite GUI in a new framework (QT) so I can use a less-laggy surface plotting tool
- Add rk4 solving algorithm
- Add adaptive step size scheme
- Better separate interface, solver, and control logic to make the code less intimidating and unwieldy
- Add functionality to GUI to remove all the farting around in "How to run:" so instead you just open the program and work in it.
