# classes for graphene sample collection

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from lammps import lammps
from scipy.signal import find_peaks
import os


class GrapheneSheet:
    def __init__(self, datafile_name, x_atoms, y_atoms, makeStrengthSurface=True):
        """
        Class to store information about a graphene sheet.
        
        Parameters:
        - datafile_name (str): The path to the data file.
        - x_atoms (int): The length of the sheet in the x direction (number of atoms).
        - y_atoms (int): The length of the sheet in the y direction (number of atoms).
        - makeStrengthSurface (bool): User specifies whether or not they want strength surface plots generated and saved (default True)
        """
        self.datafile_name = datafile_name
        self.x_atoms = x_atoms
        self.y_atoms = y_atoms
        self.makeStrengthSurface = makeStrengthSurface
        self.volume = self.calcVolume()  # volume of sheet in angstroms cubed (note that this is unrelaxed)

        if self.makeStrengthSurface:
            # initialize the plots for this sheet's strength surface (if user wants)
            fig, ax = plt.subplots()
            ax.set_xlabel(r'$\sigma_1$')
            ax.set_ylabel(r'$\sigma_2$')
            ax.set_title(f'Molecular Strength Surface of Pristine Graphite ({x_atoms}x{y_atoms})')
            self.fig = fig
            self.ax = ax

    def __repr__(self):
        return f"GrapheneSheet(datafile_name='{self.datafile_name}', x_atoms={self.x_atoms}, y_atoms={self.y_atoms}, volume={self.volume})"
    
    # takes in the length and width of graphene sheet (in atoms) and gives back the volume in angstroms
    # assumes that the x-axis is the armchair edge and the y-axis is the zigzag edge
    def calcVolume(self):
        dist = 1.42
        Lz = 3.4
        if self.x_atoms % 2 == 0:
            Lx = ((self.x_atoms / 2) * dist) + (dist * 2 * (self.x_atoms / 2 - 1)) + dist
        else:
            Lx = (((self.x_atoms - 1) / 2) * dist) + (dist * 2 * ((self.x_atoms + 1) / 2 - 1)) + (dist / 2)

        Ly = self.y_atoms * dist * np.sin(np.deg2rad(60))
        vol = Lx * Ly * Lz
        return vol

    # given the three principal stresses, populate a point on the strength surface (eventually specity 2D or 3D)
    def populate_surface(self, strength, dimension=2):
        if dimension == 2:
            self.ax.scatter(strength[0], strength[1], color='g')

    def finish_surface_plot(self):
        # Get the current x and y data (do this when you have more than one point)
        # x_data = ax.collections[0].get_offsets()[:, 0]  # Extract x-coordinates
        # y_data = ax.collections[0].get_offsets()[:, 1]  # Extract y-coordinates

        # # Calculate limits with some padding
        # x_min, x_max = np.min(x_data), np.max(x_data)
        # y_min, y_max = np.min(y_data), np.max(y_data)

        # padding = 0.1  # 10% padding

        # # Set new limits with padding
        # ax.set_xlim(x_min - padding * (x_max - x_min), x_max + padding * (x_max - x_min))
        # ax.set_ylim(y_min - padding * (y_max - y_min), y_max + padding * (y_max - y_min))

        self.fig.savefig(f'figures/{self.datafile_name.split(".")[1]}__strength_surface.png')
        # plt.show()


class Simulation:
    # we basically want the whole simulation to run on initialization, then we can pull whatever we want from it for postprocessing
    def __init__(self, comm, rank, sheet, target_x=0.0, target_y=0.0, target_z=0.0, tau=0.01, sim_length=50000, timestep=200, makeplots=False, fracture_window=10):
        """
        Class to execute one simulation and store information about it.
        This essentially loads the specimen to failure.
        
        Parameters:
        - comm (???): ???
        - rank (???): ???
        - sheet (GrapheneSheet): Object that stores all necessary info about the sheet being used.
        - target_x (float): Target pressure in the x-direction.
        - target_y (float): Target pressure in the y-direction.
        - target_z (float): Target pressure in the z-direction.
        - tau (float): Scales the damping parameter (low means pull fast, high means pull slow) - basically how quickly we want to get to P_Target
        - sim_length (int): Number of timesteps before simulation is killed (max timesteps) 
        - timestep (int): Frequency of timesteps you output data (if timestep = 100, output every 100 timesteps)
        - makeplots (bool): User specifies whether or not they want plots of stress vs time generated and saved (default False)
        - fracture_window (int): Tunable parameter that says how much stress drop (GPa) is necessary to detect fracture (to eliminate noise). 10 GPa is default
        """

        self.sheet = sheet
        self.target_x = target_x
        self.target_y = target_y
        self.target_z = target_z
        self.tau = tau
        self.sim_length = sim_length
        self.timestep = timestep
        self.makeplots = makeplots
        self.fracture_window = fracture_window

        # damping parameter calculated based on the target pressures in each direction - must initialize
        damp_x, damp_y, damp_z = self.calculate_damping(first=True)

        self.csv_file = 'simulation_data/all_simulations.csv'
        self.initialize_csv()  # initialize csv file for data storage
        self.simulation_id = self.get_simid()  # returns string of simulation id (ex: '00001')

        lmp = lammps(comm=comm)  # initialize lammps

        # import all variables to lammps
        lmp.command(f"variable Volume equal {sheet.volume}")
        lmp.command(f"variable conv_fact equal 0.0001")
        lmp.command(f"variable target_x equal {target_x}")
        lmp.command(f"variable target_y equal {target_y}")
        lmp.command(f"variable target_z equal {target_z}")
        lmp.command(f"variable damp_x equal {damp_x}")  # placeholder values, will change when simulation starts
        lmp.command(f"variable damp_y equal {damp_y}")
        lmp.command(f"variable damp_z equal {damp_z}")
        lmp.command(f"variable datafile string {sheet.datafile_name}")
        lmp.command(f"variable dump_output string dumpfiles/sim{self.simulation_id}")

        lmp.file("in.tension_setup")

        self.lmp = lmp
        self.comm = comm
        self.rank = rank

        stress_tensor, pressure_tensor, step_vector = self.run_simulation()
        principal_stresses = self.compute_principal_stresses(stress_tensor)
        strength, fracture_time = self.find_fracture(principal_stresses)

        self.stress_tensor = stress_tensor  # array of all six stress values at each outputted timestep
        self.pressure_tensor = pressure_tensor  # array of all three pressure values at each outputted timestep
        self.step_vector = step_vector  # vector of each timestep where data was collected
        self.principal_stresses = principal_stresses  # array of all three principal stress values at each outputted timestep
        self.strength = strength  # vector of the three critical principal stresses at fracture (largest to smallest)
        self.fracture_time = fracture_time  # timestep in which fracture was detected

        if self.rank == 0:
            print(f'Material Strength ({sheet.x_atoms}x{sheet.y_atoms} sheet): {strength} GPa. ')
            self.append_csv()  # append all data from the simulation to the csv file. 
            self.save_detailed_data()
            if sheet.makeStrengthSurface:
                sheet.populate_surface(strength)
            if makeplots:
                # right now we really only care about the principal stresses, so only output that
                self.plot_principalStress()
                # self.plot_pressure()
                # self.plot_stressTensor()


    # if the file doesn't exits, create one. if it does exist we'll just append to the old one so we don't overwrite data
    def initialize_csv(self):
        if not os.path.exists(self.csv_file) or os.path.getsize(self.csv_file) == 0:
            df = pd.DataFrame(columns=['Simulation ID', ' Num Atoms x', ' Num Atoms y', ' Strength_1', ' Strength_2', ' Strength_3', ' Fracture Time',
                                        ' Target Pressure x (bar)', ' Target Pressure y (bar)', ' Target Pressure z (bar)',
                                        ' tau', ' Max Sim Length', ' Output Timesteps', ' Fracture Window'])
            df.to_csv(self.csv_file, index=False)

    def get_simid(self):
        if os.path.exists(self.csv_file) and os.path.getsize(self.csv_file) > 0:
            # Read the existing file and find the maximum simulation id
            df = pd.read_csv(self.csv_file)
            if "Simulation ID" in df.columns and not df.empty:
                return str(df["Simulation ID"].max() + 1).zfill(5)
        # Default to 1 if the file doesn't exist or is empty
        return str(1).zfill(5)

    # appends a new simulation to the csv file for storage
    def append_csv(self):
        new_row = pd.DataFrame({'Simulation ID':[self.simulation_id], 'Num Atoms x': [self.sheet.x_atoms], 'Num Atoms y': [self.sheet.y_atoms], 'Strength_1': [self.strength[0]],
                                'Strength_2': [self.strength[1]], 'Strength_3': [self.strength[2]], 'Fracture Time': [self.fracture_time],
                                'Target Pressure x (bar)': [self.target_x], 'Target Pressure y (bar)': [self.target_y], 'Target Pressure z (bar)': [self.target_z],
                                'tau': [self.tau], 'Max Sim Length': [self.sim_length], 'Output Timesteps': [self.timestep], 'Fracture Window': [self.fracture_window]})
        new_row.to_csv(self.csv_file, mode="a", header=False, index=False)

    def save_detailed_data(self):
        df = pd.DataFrame({'Timestep': self.step_vector, 'PrincipalStress_1': self.principal_stresses[:, 0], 'PrincipalStress_2': self.principal_stresses[:, 1], 
                           'PrincipalStress_3': self.principal_stresses[:, 2], 'Stress_xx': self.stress_tensor[:, 0], 'Stress_yy': self.stress_tensor[:, 1],
                           'Stress_zz': self.stress_tensor[:, 2], 'Stress_xy': self.stress_tensor[:, 3], 'Stress_xz': self.stress_tensor[:, 4], 
                           'Stress_yz': self.stress_tensor[:, 5], 'Pressure_x': self.pressure_tensor[:, 0], 'Pressure_y': self.pressure_tensor[:, 1],
                           'Pressure_z': self.pressure_tensor[:, 2]})
        detailed_csv_file = f'simulation_data/sim{self.simulation_id}_fulldata.csv'
        df.to_csv(detailed_csv_file, index=False)

    # Calculates the damping parameter based on the target pressure and box pressure, such that we are pulling at the same rate independently of P_target
    # Basically, if P_T is larger, we want to make the damping parameter larger, which will give the system appropriate time to ramp up (without moving too quickly)
    def calculate_damping(self, first=False):
        if not first:
            P = self.extract_pressure()
        else:
            P = [0, 0, 0]
        damp_x = self.fix_damping(self.tau * abs(P[0] - self.target_x))
        damp_y = self.fix_damping(self.tau * abs(P[1] - self.target_y))
        damp_z = self.fix_damping(self.tau * abs(P[2] - self.target_z))

        return [damp_x, damp_y, damp_z]
    
    # damping parameter must be a positive value, so if it isn't, set it to 100 (for now, after testing prob change to like 0.1)
    def fix_damping(self, damp):
        # this is basically just checking for when P_Target = 0, and setting that damping param to 0.1
        if damp <= 0:
            damp = 0.1
        return damp

    # runs lammps simulation and returns array with every stress tensor entry over time and vector of every timestep at which data was taken
    def run_simulation(self):
        stress_tensor = np.zeros((int(self.sim_length/self.timestep + 1), 6))
        step_vector = np.zeros(int(self.sim_length/self.timestep + 1))
        pressure_tensor = np.zeros((int(self.sim_length/self.timestep + 1), 3))

        # initialize stress at time = 0
        self.lmp.command(f"run 0 pre no post no")
        stress_tensor[0] = self.extract_stress()
        self.Lz = self.lmp.extract_variable("l_z", None, 0)  # get the sim box length and store
        pressure_tensor[0] = self.extract_pressure()

        iters = 0
        # run and store for desired timesteps
        for step in range(0, self.sim_length, self.timestep):
            iters, stress_tensor, step_vector, pressure_tensor = self.run_step(step, iters, stress_tensor, step_vector, pressure_tensor)

            column_sums = np.sum(np.abs(stress_tensor), axis=0)
            dominant_direction = np.argmax(column_sums)

            # checks when dominant direction stress drops to detect fracture - this is just an on the fly check because we don't have principal stresses calculated
            strength, _ = self.find_fracture(stress_tensor[:(iters+1), dominant_direction].reshape(-1, 1))

            # if we get a strength value, that means fracture was detected and we can leave the loop
            if strength[0] is not None:
                # run the simulation for a few more timesteps to visualize fracture
                for i in range(5):
                    step += self.timestep  # update timestep (cuz we are no longer in the big loop)
                    if step == self.sim_length:
                        break
                    iters, stress_tensor, step_vector, pressure_tensor = self.run_step(step, iters, stress_tensor, step_vector, pressure_tensor)
                    
                stress_tensor = stress_tensor[:(iters+1)]
                step_vector = step_vector[:(iters+1)]
                pressure_tensor = pressure_tensor[:(iters+1)]

                return stress_tensor, pressure_tensor, step_vector

        return stress_tensor, pressure_tensor, step_vector
    
    # returns a vector of length 3 representing the pressure in each direction (imposed by npt)
    def extract_pressure(self):
        xx = self.lmp.extract_variable("p_x", None, 0)
        yy = self.lmp.extract_variable("p_y", None, 0)
        zz = self.lmp.extract_variable("p_z", None, 0)
        return [xx, yy, zz]
    
    # returns a flattened np array of the stress tensor
    def extract_stress(self):
        xx = self.lmp.extract_variable("stress_xx", None, 0)
        yy = self.lmp.extract_variable("stress_yy", None, 0)
        zz = self.lmp.extract_variable("stress_zz", None, 0)
        xy = self.lmp.extract_variable("stress_xy", None, 0)
        xz = self.lmp.extract_variable("stress_xz", None, 0)
        yz = self.lmp.extract_variable("stress_yz", None, 0)

        arr = np.array([xx, yy, zz, xy, xz, yz])
        return arr
    
    # helper function for run_simulation... runs one timestep
    def run_step(self, step, iters, stress_tensor, step_vector, pressure_tensor):
        iters += 1
        # update the damping parameters to keep constant loading rate
        damp = self.calculate_damping()
        self.lmp.command('print "damp_x = ${damp_x}"')
        self.lmp.command('print "target_x = ${target_x}"')

        self.lmp.command(f"variable damp_x equal {damp[0]}")
        self.lmp.command(f"variable damp_y equal {damp[1]}")
        self.lmp.command(f"variable damp_z equal {damp[2]}")
        self.lmp.command(f"run {self.timestep} pre no post no")
        stress = self.extract_stress()  # extract stress tensor
        pressure = self.extract_pressure()
        stress_tensor[iters] = stress
        step_vector[iters] = step + self.timestep
        pressure_tensor[iters] = pressure
        return iters, stress_tensor, step_vector, pressure_tensor
    
    # input vector of principal stresses so far, outputs strength and fracture timestep (of it occurred), otherwise None for both
    # idea here is to have the previous 5 timesteps be our testcase, and the previous 10 before that be what we're testing against
    # if the average of the previous 5 is lower than the average of the 10 before that, fracture has occurred and we should find peaks
    # we know that principal_stresses[:, 0] will always be the dominant direction
    def find_fracture(self, principal_stresses):
        # Not enough timesteps to actually check
        if len(principal_stresses[:, 0]) < 25:
            strength = [None, None, None]  # must be a list of None's for later
            fracture_timestep = None
        # Now there's enough timesteps to check
        else:
            mean_last_10 = sum(principal_stresses[:, 0][-10:]) / 10
            mean_15_before = sum(principal_stresses[:, 0][-25:-10]) / 15

            # if we are still increasing on average, no fracture yet
            if mean_last_10 >= mean_15_before:
                strength = [None, None, None]
                fracture_timestep = None
            
            # we have detected a significant drop on average (fracture)
            else:
                peaks, _ = find_peaks(principal_stresses[:, 0])
                if len(peaks) == 0:
                    # No peaks found (idt this could ever happen but who knows lol)
                    strength = [None, None, None]
                    fracture_timestep = None
                
                # Now we are confident there is fracture, let's return the proper values
                else:
                    fracture_index = peaks[np.argmax(principal_stresses[:, 0][peaks])]  # find the index of the highest peak

                    # see if this "fracture" is high enough to be considered (or is it just noise?)
                    # for fracture, the (peak - window) must be greater than the minimum point
                    if (np.max(principal_stresses[fracture_index]) - self.fracture_window < np.min(principal_stresses[:, 0][-10:])):
                        strength = [None, None, None]
                        fracture_timestep = None
                    else:
                        strength = principal_stresses[fracture_index]
                        fracture_timestep = fracture_index * self.timestep
                        print("FRACTUREEEEEE")

        return strength, fracture_timestep
    
    # returns principal stresses in all three directions - ordered from which direction experienced max stress to min stress
    def compute_principal_stresses(self, stress_tensor):
        num_timesteps = stress_tensor.shape[0]
        principal_stresses = np.zeros((num_timesteps, 3))

        for i in range(num_timesteps):
            # get whole stress tensor for this timestep
            stress_matrix = np.array([[stress_tensor[i, 0], stress_tensor[i, 3], stress_tensor[i, 4]],
                                    [stress_tensor[i, 3], stress_tensor[i, 1], stress_tensor[i, 5]],
                                    [stress_tensor[i, 4], stress_tensor[i, 5], stress_tensor[i, 2]]])
            
            # compute principal stresses and principal directions - this is sorted from lowest to highest eigval every time, 
            eigvals = np.linalg.eigvalsh(stress_matrix)
            principal_stresses[i] = eigvals[::-1]

        return principal_stresses
    
    # plots value of each principal stress over time
    def plot_principalStress(self):
        princ_fig, princ_ax = plt.subplots()

        princ_ax.plot(self.step_vector, self.principal_stresses[:, 0], label=r'$\sigma_1$')
        princ_ax.plot(self.step_vector, self.principal_stresses[:, 1], label=r'$\sigma_2$')
        princ_ax.plot(self.step_vector, self.principal_stresses[:, 2], label=r'$\sigma_3$')
        if self.fracture_time is not None:
            princ_ax.axvline(x=self.fracture_time, color='r', linestyle='--', linewidth=1, label='Fracture Time')

        princ_ax.set_title(f"Principal Stresses vs. Time ({self.sheet.x_atoms}x{self.sheet.y_atoms}, tau={self.tau}, Pt={self.target_x}, {self.target_y})")
        princ_ax.set_xlabel("Timesteps")
        princ_ax.set_ylabel("Stress (GPa)")
        princ_ax.legend()
        princ_ax.grid()
        princ_fig.savefig(f'figures/sim{self.simulation_id}_princ_vs_time.png')

    # plots value of each pressure value imposed by npt over time
    def plot_pressure(self):
        pressure_fig, pressure_ax = plt.subplots()

        pressure_ax.plot(self.step_vector, self.pressure_tensor[:, 0], label=r'$p_x$')
        pressure_ax.plot(self.step_vector, self.pressure_tensor[:, 1], label=r'$p_y$')
        pressure_ax.plot(self.step_vector, self.pressure_tensor[:, 2], label=r'$p_z$')
        if self.fracture_time is not None:
            pressure_ax.axvline(x=self.fracture_time, color='r', linestyle='--', linewidth=1, label='Fracture Time')

        pressure_ax.set_title(f"Applied Pressure vs. Time ({self.sheet.x_atoms}x{self.sheet.y_atoms})")
        pressure_ax.set_xlabel("Timesteps")
        pressure_ax.set_ylabel("Pressure (Bar)")
        pressure_ax.legend()
        pressure_ax.grid()
        pressure_fig.savefig(f'figures/{self.sheet.datafile_name.split(".")[1]}__press_vs_time.png')

    # plots value of each stress tensor direction over time
    def plot_stressTensor(self):
        stress_tensor_fig, stress_tensor_ax = plt.subplots()

        stress_tensor_ax.plot(self.step_vector, self.stress_tensor[:, 0], label=r'$\sigma_{xx}$')
        stress_tensor_ax.plot(self.step_vector, self.stress_tensor[:, 1], label=r'$\sigma_{yy}$')
        stress_tensor_ax.plot(self.step_vector, self.stress_tensor[:, 2], label=r'$\sigma_{zz}$')
        stress_tensor_ax.plot(self.step_vector, self.stress_tensor[:, 3], label=r'$\sigma_{xy}$')
        stress_tensor_ax.plot(self.step_vector, self.stress_tensor[:, 4], label=r'$\sigma_{xz}$')
        stress_tensor_ax.plot(self.step_vector, self.stress_tensor[:, 5], label=r'$\sigma_{yz}$')
        if self.fracture_time is not None:
            stress_tensor_ax.axvline(x=self.fracture_time, color='r', linestyle='--', linewidth=1, label='Fracture Time')

        stress_tensor_ax.set_title(f"Stress Components vs. Time ({self.sheet.x_atoms}x{self.sheet.y_atoms})")
        stress_tensor_ax.set_xlabel("Timesteps")
        stress_tensor_ax.set_ylabel("Stress (GPa)")
        stress_tensor_ax.legend()
        stress_tensor_ax.grid()
        stress_tensor_fig.savefig(f'figures/{self.sheet.datafile_name.split(".")[1]}__tensor_vs_time.png')


class Relaxation:
    def __init__(self, comm, rank, sheet):
        """
        Class to execute one relaxation simulation and store information about it.
        
        Parameters:
        - comm (???): ???
        - rank (???): ???
        - sheet (GrapheneSheet): Object that stores all necessary info about the sheet being used.
        """

        lmp = lammps(comm=comm)

        lmp.command(f"variable Volume equal {sheet.volume}")
        lmp.command(f"variable conv_fact equal 0.0001")
        lmp.command(f"variable datafile string {sheet.datafile_name}")
        lmp.command(f"variable dump_output string dump.{sheet.x_atoms}_{sheet.y_atoms}")


        lmp.file("in.relaxation")

        self.lmp = lmp
        self.comm = comm
        self.rank = rank
        self.sheet = sheet