""" Houses all of the logic for running one simulation """

from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from lammps import lammps
from scipy.signal import find_peaks
import os
from datetime import timedelta, datetime
import time
from filelock import FileLock
import csv
import sys
import json
import local_config 
from scipy.linalg import polar


class GrapheneSheet:
    def __init__(self, datafile_name, x_atoms, y_atoms):
        """
        Class to store information about a graphene sheet.
        
        Parameters:
        - datafile_name (str): The path to the data file.
        - x_atoms (int): The length of the sheet in the x direction (number of atoms).
        - y_atoms (int): The length of the sheet in the y direction (number of atoms).
        """
        self.datafile_name = datafile_name
        self.x_atoms = x_atoms
        self.y_atoms = y_atoms
        self.volume = self.calcVolume()  # volume of sheet in angstroms cubed (note that this is unrelaxed)

    def __repr__(self):
        return f"GrapheneSheet(datafile_name='{self.datafile_name}', x_atoms={self.x_atoms}, y_atoms={self.y_atoms}, volume={self.volume})"
    
    def calcVolume(self):
        """
        - This calculates the volume of the graphene sheet in angstroms cubed, using the length and width of the sheet in number of atoms
        - It assumes the x-axis is the armchair edge and the y-axis is the zigzag edge
        """
        dist = 1.42
        Lz = 3.4
        if self.x_atoms % 2 == 0:
            Lx = ((self.x_atoms / 2) * dist) + (dist * 2 * (self.x_atoms / 2 - 1)) + dist
        else:
            Lx = (((self.x_atoms - 1) / 2) * dist) + (dist * 2 * ((self.x_atoms + 1) / 2 - 1)) + (dist / 2)

        Ly = self.y_atoms * dist * np.sin(np.deg2rad(60))
        vol = Lx * Ly * Lz

        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz

        return vol
    
    def extract_atom_positions(self):
        """
        - This extracts atom positions from a LAMMPS data file.

        Returns:
        - positions: Nx4 NumPy array with columns [id, x, y, z]
        """
        positions = []
        with open(self.datafile_name, 'r') as f:
            lines = f.readlines()

        # Find start of "Atoms" section
        atom_start = None
        for i, line in enumerate(lines):
            if line.strip().startswith("Atoms"):
                atom_start = i + 2  # Skip "Atoms" line and its header
                break

        if atom_start is None:
            raise ValueError("Could not find 'Atoms' section in file.")

        # Read until next empty line (end of atom section)
        for line in lines[atom_start:]:
            if line.strip() == '' or line.startswith('Velocities'):
                break
            parts = line.strip().split()
            atom_id = int(parts[0])
            x = float(parts[2])
            y = float(parts[3])
            z = float(parts[4])
            positions.append([atom_id, x, y, z])

        return np.array(positions)


class Simulation:
    # we basically want the whole simulation to run on initialization, then we can pull whatever we want from it for postprocessing
    def __init__(self, comm, rank, sheet, num_procs,
                 x_erate=0, y_erate=0, z_erate=0, xy_erate=0, xz_erate=0, yz_erate=0, 
                 sim_length=100000, timestep=0.0005, thermo=1000, 
                 defects="None", defect_random_seed=42,
                 detailed_data=False, fracture_window=10, theta=0,
                 storage_path=f'{local_config.DATA_DIR}/defected_data', accept_dupes=False,
                 angle_testing=False, repeat_sim=None):
        """
        Class to execute one simulation and store information about it.
        This essentially loads the specimen to failure.
        
        Parameters:
        - comm (mpi4py.MPI.Comm): MPI communicator that defines the group of processes which can talk to each other... using MPI.COMM_WORLD for all of this.
        - rank (int): The unique process ID within the communicator, ranging from 0 to n_procs-1... used mostly to control output (rank 0 prints, etc)
        - sheet (GrapheneSheet): Object that stores all necessary info about the sheet being used.
        - num_procs (int): Specifies the number of processors used for this simulation (size of comm) - just to document along with the time 
        - x_erate (float): Strain rate for fix deform (in 1/ps) (same for all other directions)
        - sim_length (int): Number of timesteps before simulation is killed (max timesteps) 
        - timestep (float): Size of one timestep in LAMMPS simulation (picoseconds)
        - thermo (int): Frequency of timesteps you output data (if thermo = 100, output every 100 timesteps)
        - defects (str): JSON-like string in the form {"type": percentage}. ex: '{"SV": 0.5}'
        - defect_random_seed (int): Sets the random seed for the defect generation
        - detailed_data (bool): User specifies whether or not they want to save the detailed timestep data (default False)
        - fracture_window (int): Tunable parameter that says how much stress drop (GPa) is necessary to detect fracture (to eliminate noise). 10 GPa is default
        - theta (float): Perscribed angle of max principal stress (for storage only, to compare with actual)
        - storage_path (str): filepath to where we want to store the data
        - accept_dupes (bool): Don't kill the simulation if we find a duplicate
        - angle_testing (bool): Execute simulation and data storage for the angle testing dataset - to generate erate_x, y, xy map to sigma_1, 2, theta. 
        - repeat_sim (int): If not None, it will find and run the exact simulation you give it (simid) and save the detailed data (but will not duplicate large data)
        """

        # set up this instances variables
        self.comm = comm
        self.rank = rank
        self.sheet = sheet
        self.num_procs = num_procs
        self.storage_path = storage_path
        self.timestep = timestep
        self.angle_testing = angle_testing
        self.repeat_sim = repeat_sim

        # if we are not repeating a sim, continue setting up with whatever was inputted
        if repeat_sim is None:
            self.x_erate = x_erate
            self.y_erate = y_erate
            self.z_erate = z_erate
            self.xy_erate = xy_erate
            self.xz_erate = xz_erate
            self.yz_erate = yz_erate
            self.sim_length = sim_length
            self.thermo = thermo
            self.defect_random_seed = defect_random_seed
            self.detailed_data = detailed_data
            self.fracture_window = fracture_window
            self.theta = theta
            self.defects = self.parse_defect_string(defects)
            self.accept_dupes = accept_dupes
        # if we are repeating a sim, ignore the inputs and find the fields from the sim we aim to repeat
        else:
            self.setup_identical_sim(repeat_sim)  # this sets x_erate, etc to the same as whatever it was in that particular simid.

        if not self.accept_dupes:
            self.check_duplicate()  # ensure that we haven't run this sim yet

        self.starttime = time.perf_counter()  # start simulation timer

        self.erate_timestep = self.calculate_erateTimestep()  # used to calculate strain rate at any given timestep (just multiply this by whatever timestep you're on)

        if self.detailed_data:
            self.create_storage_directory()  # create data storage directory (storage_path/sim{datetime})

        lmp = lammps(comm=comm)  # initialize lammps

        self.lmp = lmp  # store this instance of lammps so we can refer to it throughout the class

        self.setup_lammps()  # puts all needed variables in lammps and initializes file

        self.introduce_defects()  # put whatever defects into the graphene sheet that were specified --> does nothing if defects is None

        self.apply_fix_deform()  # basically prepares the fix deform command

        # run the simulation for the specified time period (or until fracture). Stores useful information in these tensors.
        stress_tensor, pressure_tensor, step_vector, strain_tensor, rotation_vector = self.run_simulation()

        # now we can get all of the principal information for stress and strain with this function
        # principal_angles is a list of the actual angle of dominant loading direction for each timestep
        principal_stresses, self.principal_angles = self.compute_principal_StressStrain(stress_tensor, return_theta=True)
        principalAxes_strain = self.compute_principal_StressStrain(strain_tensor)

        # store all of this useful information to use later throughout the class
        self.stress_tensor = stress_tensor  # array of all six stress values at each outputted thermo
        self.pressure_tensor = pressure_tensor  # array of all three pressure values at each outputted thermo
        self.strain_tensor = strain_tensor  # array of all six strain values at each outputted thermo
        self.step_vector = step_vector  # vector of each thermo where data was collected
        self.principal_stresses = principal_stresses  # array of all three principal stress values at each outputted thermo
        self.principalAxes_strain = principalAxes_strain  # array of strain values along all three principal directions at each outputted thermo
        self.rotation_vector = rotation_vector  # vector of the rotation of the lattice from deformation at each saved timestep

        # finally, once the sim is complete, we can get the exact point of fracture, the max strength, and the critical strain
        try:
            strength, crit_strain, fracture_time = self.find_fracture(principal_stresses, give_crit_strain=True)
        # if there was no fracture (like we just reached max sim length), just store these values as None. 
        except ValueError as e:
            print("Warning: Encountered ValueError on final find_fracture sweep. Defaulting to None.")
            print(e)
            strength = [None, None, None]
            crit_strain = [None, None, None]
            fracture_time = None

        # store all of this useful information for later
        self.strength = strength  # vector of the three critical principal stresses at fracture (largest to smallest)
        self.crit_strain = crit_strain  # vector of the three critical strains in the principal sirections at fracture (largest to smallest)
        self.fracture_time = fracture_time  # thermo in which fracture was detected

        self.sim_duration = timedelta(seconds=time.perf_counter() - self.starttime)  # record how long the simulation took

        if self.rank == 0:
            # gets simid and saves everything, also handles if we only did angle testing... this is the final data storage for this sim
            self.finalize_dataset()

            print(f'Material Strength ({sheet.x_atoms}x{sheet.y_atoms} sheet): {strength} GPa. ')


    ############ END INIT ############
    ############ BEGIN SIMULATION INPUT SETUP ############


    def setup_identical_sim(self, simid):
        """
        - We do this when we detected that a repeat_sim is requested.
        - Here, we will look through all_simulations.csv to find that simid, then essentially copy all of the 
        important input information to this simulation, running the exact simulation again.
        - This is useful if you want to now run a sim again, but this time save the detailed data.
        - Doing this will not save the simulation again in all_simulations.csv (because an identical is already there).

        Parameters:
            simid (int): The simid that we want to reproduce

        Raises:
            FileNotFoundError: If the storage CSV path (all_simulations.csv) does not exist
            ValueError: If the CSV does not contain that simid specified
            ValueError: If one of the necessary rows is empty in the csv
        """
        sim_row = None
        
        # zero'th rank must find the simulation from the csv 
        if self.rank == 0:

            csv_path = os.path.join(self.storage_path, "all_simulations.csv")
            # ensure file exists
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"Storage CSV {csv_path} does not exist.")
                
            df = pd.read_csv(csv_path)

            # ensure it contains this simid
            if "Simulation ID" not in df.columns:
                raise ValueError("CSV does not contain 'Simulation ID' column.")

            matching_rows = df[df["Simulation ID"] == simid]

            # ensure this simid has all of the rows that we want actually populated
            if matching_rows.empty:
                raise ValueError(f"No row found in CSV with Simulation ID = {simid}")

            sim_row = matching_rows.iloc[0].to_dict()
        sim_row = MPI.COMM_WORLD.bcast(sim_row, root=0)

        MPI.COMM_WORLD.Barrier()  # ensure all ranks wait for rank 0

        # set these simulation settings identical to the one we found
        self.x_erate = sim_row["Strain Rate x"]
        self.y_erate = sim_row["Strain Rate y"]
        self.z_erate = sim_row["Strain Rate z"]
        self.xy_erate = sim_row["Strain Rate xy"]
        self.xz_erate = sim_row["Strain Rate xz"]
        self.yz_erate = sim_row["Strain Rate yz"]
        self.sim_length = sim_row["Max Sim Length"]
        self.thermo = sim_row["Output Timesteps"]
        self.defect_random_seed = int(sim_row["Defect Random Seed"])
        self.detailed_data = True
        self.fracture_window = sim_row["Fracture Window"]
        self.theta = sim_row["Theta Requested"]
        self.defects = self.parse_defect_string(sim_row["Defects"])
        self.accept_dupes = True

    def parse_defect_string(self, defect_str):
        """
        - Parses a JSON string representing defects and validates key/value types.
        - This is the backbone of how our defect string input works. 

        Parameters:
            defect_str (str): JSON-formatted string, e.g. '{"SV": 0.5, "DV": 0.2}'

        Returns:
            dict[str, float]: Parsed and validated defect dictionary.

        Raises:
            ValueError: If JSON is invalid or structure does not match {str: float}
        """

        # print the defects just for debugging purposes
        print(defect_str)

        # format it if None is specified
        defect_str = defect_str.strip().upper()
        if defect_str in ("NONE", "", "NULL"):
            return {}

        # turn it into a json
        try:
            defects = json.loads(defect_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")

        # ensure its a dict
        if not isinstance(defects, dict):
            raise ValueError("Defect input must be a JSON object (dictionary).")

        # ensure formatting is correct
        for key, value in defects.items():
            if not isinstance(key, str):
                raise ValueError(f"Invalid key type: {key} (must be string)")
            if not isinstance(value, (int, float)):
                raise ValueError(f"Invalid value for '{key}': {value} (must be a number)")
            if key != "SV" and key != "DV":
                raise ValueError(f"Defect type must be either 'SV' or 'DV'. Received {key}")
            if value > 100 or value < 0:
                raise ValueError(f"Defect percentage must be between 0 and 100%. Received {value}")

        return defects

    def check_duplicate(self):
        """
        - Check to ensure that we have not run this simulation before (unless accept_dupes is true).
        - This basically just aborts the sim if it finds we have already run an identical one
        """
        if self.rank != 0:
            return  # Only rank 0 should check

        csv_path = os.path.join(self.storage_path, "all_simulations.csv")
        if not os.path.exists(csv_path):
            return

        # define these mini functions just to make life easier
        def safe_float(val):
            try:
                return float(val)
            except (ValueError, TypeError):
                return None

        def safe_int(val):
            try:
                return int(float(val))
            except (ValueError, TypeError):
                return None

        def safe_str(val):
            return str(val).strip().lower() if val is not None else ""

        def compare(col, value, is_float=True):
            if col not in row or row[col] is None:
                return True  # Column missing → ignore condition
            if is_float:
                ref = safe_float(row[col])
                return ref is not None and abs(ref - value) < 1e-10
            else:
                ref = safe_int(row[col]) if isinstance(value, int) else safe_str(row[col])
                return ref == value

        def compare_dict(col, value_dict):
            if col not in row or row[col] is None:
                return True  # Column missing → ignore condition
            try:
                row_dict = json.loads(row[col])
                return row_dict == value_dict  # direct dict comparison
            except json.JSONDecodeError:
                return False

        # utilize those mini functions and the compare to make sure we aren't running a duplicate simulation.
        with open(csv_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                match = (
                    compare("Strain Rate x", self.x_erate) and
                    compare("Strain Rate y", self.y_erate) and
                    compare("Strain Rate z", self.z_erate) and
                    compare("Strain Rate xy", self.xy_erate) and
                    compare("Strain Rate xz", self.xz_erate) and
                    compare("Strain Rate yz", self.yz_erate) and
                    compare("Num Atoms x", self.sheet.x_atoms, is_float=False) and
                    compare("Num Atoms y", self.sheet.y_atoms, is_float=False) and
                    compare("Max Sim Length", self.sim_length, is_float=False) and
                    compare("Output Timesteps", self.thermo, is_float=False) and
                    compare("Fracture Window", self.fracture_window, is_float=False) and
                    compare_dict("Defects", self.defects) and
                    compare("Defect Random Seed", self.defect_random_seed, is_float=False)
                )
                # if everything was exact, skip it
                if match:
                    print(f"[Rank 0] Already completed. Skipping: x={self.x_erate}, y={self.y_erate}, xy={self.xy_erate}")
                    sys.stdout.flush()
                    self.comm.Abort()


    ############ END SIMULATION INPUT SETUP ############
    ############ BEGIN DATA STORAGE ############


    def finalize_dataset(self):
        """
        - We call this once a simulation is complete to save all of the results. 
        - This essentially goes into the all_simulations main csv, obtains a proper simid (unless there already is one from
        a repeat sim), then saves the important information to the main csv (unless repeat sim), then saves detailed data if necessary.
        """
        # APPLY THE LOCKING MECHANISM HERE
        self.main_csv = f'{self.storage_path}/all_simulations.csv'
        lockfile = f'{self.storage_path}/all_simulations.lock'

        with FileLock(lockfile):
            self.initialize_maincsv()  # initialize csv file for data storage

            self.simulation_id = self.get_simid()  # returns string of simulation id (ex: '00001')
            
            # save main data as long as we are not repeating a sim
            if self.repeat_sim is None:    
                self.append_maincsv()  # append all data from the simulation to the csv file. 

        if self.detailed_data:
            self.save_detailed_data()

    def create_storage_directory(self):
        """
        - This creates the directory for where the detailed data is stored for this simulation (if we want the detailed data stored)
        - When this directoyr is first named, it uses the timestamp. This will update to the simid upon a successful simulation run.
        - If the simulation fails, the name will stay as the timestamp.
        - This is called in the beginning of the simulation, if detailed_data=True.
        - If this is a repeat sim, we already know the simulation id, so we automatically assign the directory name the proper simid.
        """
        if self.repeat_sim is None:
            if self.rank == 0:
                timestamp_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
            else:
                timestamp_id = None

            timestamp_id = self.comm.bcast(timestamp_id, root=0)  # must do this to sync ranks up
            self.timestamp_id = timestamp_id
            self.simulation_directory = f'{self.storage_path}/sim{self.timestamp_id}'
        # if we are repeating a sim, just make the directory the simid from the start
        else:
            self.simulation_directory = f'{self.storage_path}/sim{str(self.repeat_sim).zfill(5)}'
        
        os.makedirs(self.simulation_directory, exist_ok=True)

    def initialize_maincsv(self):
        """
        - This initializes the all_simulations.csv (main csv file that stores the data for all sims).
        - If the file doesn't exist, create one. If it does exist, append to it (do not overwrite data)
        - This function is called in 'self.finalize_dataset()' to prepare the main csv file for data storage
        """

        # we have two different csv file formats for whether we are doing simple angle testing, or the whole fracture simulation
        if not os.path.exists(self.main_csv) or os.path.getsize(self.main_csv) == 0:
            if self.angle_testing:
                df = pd.DataFrame(columns=['Simulation ID', 'Strain Rate x', 'Strain Rate y', 'Strain Rate xy', 
                                        'Sigma_x', 'Sigma_y', 'Sigma_xy', 'Sigma_1', 'Sigma_2', 'Theta', 'Rotation Angle', 'Sigma_Ratio'])
            else:
                df = pd.DataFrame(columns=['Simulation ID', 'Num Atoms x', 'Num Atoms y', 'Strength_1', 'Strength_2', 'Strength_3', 
                                       'CritStrain_1', 'CritStrain_2', 'CritStrain_3', 'Strain Rate x', 'Strain Rate y', 'Strain Rate z',
                                       'Strain Rate xy', 'Strain Rate xz', 'Strain Rate yz', 'Strength x', 'Strength y', 'Strength z', 
                                       'Strength xy', 'Strength xz', 'Strength yz', 'Fracture Time', 'Max Sim Length', 'Output Timesteps', 
                                       'Fracture Window', 'Theta Requested', 'Theta', 'Rotation Angle', 'Defects', 'Defect Random Seed', 
                                       'Simulation Time', 'Threads'])
            
            df.to_csv(self.main_csv, index=False)  # save it to the csv

    def get_simid(self):
        """
        - This gets and saves the simulation id for this simulation.
        - If it is a repeat simulation, we already have the simid so just save it as such
        - If it is not a repeat simulation, go into the main csv and get the next simid in line and assign it to this simulation.
        - This is called in 'self.finalize_dataset()' at the end of the simulation to get the simid for storage.
        """
        if self.repeat_sim is not None:
            # if repeat sim, we already have simid
            return str(self.repeat_sim).zfill(5)
        if os.path.exists(self.main_csv) and os.path.getsize(self.main_csv) > 0:
            # Read the existing file and find the maximum simulation id
            df = pd.read_csv(self.main_csv)
            if "Simulation ID" in df.columns and not df.empty:
                return str(df["Simulation ID"].max() + 1).zfill(5)
        # Default to 1 if the file doesn't exist or is empty
        return str(1).zfill(5)

    # appends a new simulation to the csv file for storage
    def append_maincsv(self):
        """
        - This appends a new simulation and all of the main data to the main csv file for storage
        - This is called in 'self.finalize_dataset()' once we initialize the main csv and after we get the simid. At this point,
        all that's left to do is simply append the new row with all of the data we have and save it.
        """
        if self.angle_testing:
            new_row = pd.DataFrame({'Simulation ID':[self.simulation_id], 'Strain Rate x': [self.x_erate], 'Strain Rate y': [self.y_erate], 'Strain Rate xy': [self.xy_erate],
                                    'Sigma_x': [self.stress_tensor[-1, 0]], 'Sigma_y': [self.stress_tensor[-1, 1]], 'Sigma_xy': [self.stress_tensor[-1, 3]],
                                    'Sigma_1': [self.principal_stresses[-1, 0]], 'Sigma_2': [self.principal_stresses[-1, 1]], 'Theta': [self.principal_angles[-1]],
                                    'Rotation Angle': [self.rotation_vector[-1]], 'Sigma_Ratio': [self.principal_stresses[-1, 1] / self.principal_stresses[-1, 0]]})
        else:
            try:
                fracture_index = int(self.fracture_time / self.thermo)
            except TypeError as e:
                print(f"Warning: self.fracture_time = {self.fracture_time} and self.thermo = {self.thermo}.\n{e}")
                fracture_index = None

            # This picks from our array, but returns None if the index is None
            def pick(arr, idx, col=None):
                if idx is None:
                    return None
                try:
                    if col is None:
                        return arr[idx]
                    else:
                        return arr[idx, col]
                except Exception as e:
                    print(e)
                    return None

            new_row = pd.DataFrame({'Simulation ID':[self.simulation_id], 'Num Atoms x': [self.sheet.x_atoms], 'Num Atoms y': [self.sheet.y_atoms], 
                                'Strength_1': [self.strength[0]], 'Strength_2': [self.strength[1]], 'Strength_3': [self.strength[2]],
                                'CritStrain_1': [self.crit_strain[0]], 'CritStrain_2': [self.crit_strain[1]], 'CritStrain_3': [self.crit_strain[2]],
                                'Strain Rate x': [self.x_erate], 'Strain Rate y': [self.y_erate], 'Strain Rate z': [self.z_erate],
                                'Strain Rate xy': [self.xy_erate], 'Strain Rate xz': [self.xz_erate], 'Strain Rate yz': [self.yz_erate],
                                'Strength x': [pick(self.stress_tensor, fracture_index, 0)],
                                'Strength y': [pick(self.stress_tensor, fracture_index, 1)],
                                'Strength z': [pick(self.stress_tensor, fracture_index, 2)],
                                'Strength xy': [pick(self.stress_tensor, fracture_index, 3)],
                                'Strength xz': [pick(self.stress_tensor, fracture_index, 4)],
                                'Strength yz': [pick(self.stress_tensor, fracture_index, 5)],
                                'Fracture Time': [self.fracture_time], 'Max Sim Length': [self.sim_length], 'Output Timesteps': [self.thermo], 
                                'Fracture Window': [self.fracture_window], 'Theta Requested': [self.theta], 'Theta': [pick(self.principal_angles, fracture_index)], 
                                'Rotation Angle': [pick(self.rotation_vector, fracture_index)], 'Defects': [json.dumps(self.defects)], 'Defect Random Seed': [self.defect_random_seed], 
                                'Simulation Time': [self.sim_duration], 'Threads': [self.num_procs]})
        new_row.to_csv(self.main_csv, mode="a", header=False, index=False)

    def save_detailed_data(self):
        """
        - This saves the detailed data csv (data for each timestep) in the same directory as the dumpfile.
        - First, it renames the directory. For the entire simulation, the directory was named as the timestamp.
        Now that we have the simid, we rename the directory to the simid, rename the dumpfile, and finally save this csv.
        - This is called in 'self.finalize_dataset()' if detailed_data is set to True.
        """
        df = pd.DataFrame({'Timestep': self.step_vector, 
                           'PrincipalStress_1': self.principal_stresses[:, 0], 'PrincipalStress_2': self.principal_stresses[:, 1], 
                           'PrincipalStress_3': self.principal_stresses[:, 2], 'Strain_1': self.principalAxes_strain[:, 0], 
                           'Strain_2': self.principalAxes_strain[:, 1], 'Strain_3': self.principalAxes_strain[:, 2], 
                           'Stress_xx': self.stress_tensor[:, 0], 'Stress_yy': self.stress_tensor[:, 1], 'Stress_zz': self.stress_tensor[:, 2], 
                           'Stress_xy': self.stress_tensor[:, 3], 'Stress_xz': self.stress_tensor[:, 4], 'Stress_yz': self.stress_tensor[:, 5],
                           'Strain_xx': self.strain_tensor[:, 0], 'Strain_yy': self.strain_tensor[:, 1], 'Strain_zz': self.strain_tensor[:, 2], 
                           'Strain_xy': self.strain_tensor[:, 3], 'Strain_xz': self.strain_tensor[:, 4], 'Strain_yz': self.strain_tensor[:, 5],
                           'Theta': self.principal_angles[:], 'Rotation Angle': self.rotation_vector[:],
                           'Pressure_x': self.pressure_tensor[:, 0], 'Pressure_y': self.pressure_tensor[:, 1], 'Pressure_z': self.pressure_tensor[:, 2]})
        
        # now we can rename the directories and filenames (because we have the actual simid)

        # rename the simulation directory and dumpfiles
        old_directory = self.simulation_directory
        new_directory = f'{self.storage_path}/sim{self.simulation_id}'
        os.rename(old_directory, new_directory)
        self.simulation_directory = new_directory  # update directory

        # rename dump files inside directory (if necessary)
        for fname in os.listdir(new_directory):
            if self.repeat_sim is None:
                if self.timestamp_id in fname:
                    new_fname = fname.replace(self.timestamp_id, self.simulation_id)
                    os.rename(os.path.join(new_directory, fname), os.path.join(new_directory, new_fname))

        # finally save this csv file in the directory.
        detailed_csv_file = f'{self.simulation_directory}/sim{self.simulation_id}.csv'
        df.to_csv(detailed_csv_file, index=False)


    ############ END DATASTORAGE ############
    ############ BEGIN SIMULATION ENGINE ############


    def setup_lammps(self):
        """
        - This sets up the LAMMPS simulation and imports all of the important variables to LAMMPS
        - This is called in 'self.__init__()' directly after this instance of LAMMPS is created
        """
        self.lmp.command(f"variable vol equal {self.sheet.volume}")
        self.lmp.command(f"variable sheet_lx equal {self.sheet.Lx}")
        self.lmp.command(f"variable sheet_ly equal {self.sheet.Ly}")
        self.lmp.command(f"variable sheet_lz equal {self.sheet.Lz}")
        self.lmp.command(f"variable timestep equal {self.timestep}")

        self.lmp.command(f"variable conv_fact equal 0.0001")
        self.lmp.command(f"variable datafile string {self.sheet.datafile_name}")

        # if we are storing the detailed data, tell LAMMPS where to put the dumpfile
        if self.detailed_data:
            # if it is repeat_sim, the name of the dumpfile will be the actual simid. Otherwise, it will be the timestamp (until simulation ends)
            if self.repeat_sim is not None:
                self.lmp.command(f"dump 1 all custom {self.thermo} {self.simulation_directory}/dump.sim{str(self.repeat_sim).zfill(5)} id type x y z")
            else:
                # make it with the timestamp id becasue we don't have simid yet
                self.lmp.command(f"dump 1 all custom {self.thermo} {self.simulation_directory}/dump.sim{self.timestamp_id} id type x y z")

        self.lmp.file("in.deform_py")

    def apply_fix_deform(self):
        """
        - This writes the fix deform command for LAMMPS and sends it to the instance of LAMMPS
        - This is called in 'self.__init__()' and it basically prepares the deform command to drive the system.
        """
        # Start building the command
        fix_command = "fix 2 all deform 1"

        # Add deformation options dynamically (depending on what the user specifies)
        if self.x_erate != 0:
            fix_command += f" x erate {self.x_erate}"
        if self.y_erate != 0:
            fix_command += f" y erate {self.y_erate}"
        if self.z_erate != 0:
            fix_command += f" z erate {self.z_erate}"
        if self.xy_erate != 0:
            fix_command += f" xy erate {self.xy_erate}"
        if self.xz_erate != 0:
            fix_command += f" xz erate {self.xz_erate}"
        if self.yz_erate != 0:
            fix_command += f" yz erate {self.yz_erate}"

        # Ensure remap is included
        fix_command += " remap x"
        self.lmp.command(fix_command)  # send it to LAMMPS

    def calculate_erateTimestep(self):
        """
        - This takes the strain rates and the simulation timestep to give a vector that holds the strain per timestep for the simulation
        - Since we are prescribing a strain rate, we can easily obtain the strain by seeing how long we have applied the strain rate for
        - This is used in 'self.__init__()' to initialize the value, but then is actually used in 'self.compute_strain(timestep)' to get
        the strain for any given timestep.
        
        Returns:
            - (np.array[float]): This essentially tells us how much strain we gain each timestep. When multiplied by the amount of timesteps that
            have passed, you get the total strain for that direction.
        """
        xx = self.x_erate * self.timestep
        yy = self.y_erate * self.timestep
        zz = self.z_erate * self.timestep
        xy = self.xy_erate * self.timestep
        xz = self.xz_erate * self.timestep
        yz = self.yz_erate * self.timestep
        return np.array([xx, yy, zz, xy, xz, yz])

    def run_simulation(self):
        """
        - This runs the entire LAMMPS simulation for the specified number of timesteps (or until failure).

        Returns:
            - stress_tensor (np.array[float]): array with every stress tensor for each timestep
            - pressure_tensor (np.array[float]): array with every pressure tensor for each timestep
            - step_vector (np.array[int]): vector with each timestep at which data was taken and stored (matches indices with other tensors)
            - strain_tensor (np.array[float]): array with every strain tensor for each timestep
            - rotation_vector (np.array[float]): vector with the rotation of the lattice for each timestep (from polar decomp of F)
        """

        # initialize output tensors
        stress_tensor = np.zeros((int(self.sim_length/self.thermo + 1), 6))
        step_vector = np.zeros(int(self.sim_length/self.thermo + 1))
        pressure_tensor = np.zeros((int(self.sim_length/self.thermo + 1), 3))
        strain_tensor = np.zeros((int(self.sim_length/self.thermo + 1), 6))
        rotation_vector = np.zeros(int(self.sim_length/self.thermo + 1))

        # initialize stress at time = 0
        self.lmp.command(f"run 0 pre yes post yes")
        stress_tensor[0] = self.extract_stress()
        pressure_tensor[0] = self.extract_pressure()
        strain_tensor[0] = self.compute_strain(0)
        rotation_vector[0] = self.compute_rotation(strain_tensor[0])

        iters = 0
        # run and store for desired timesteps
        for step in range(0, self.sim_length, self.thermo):
            iters, stress_tensor, step_vector, pressure_tensor, strain_tensor, rotation_vector = self.run_step(
                step, iters, stress_tensor, step_vector, pressure_tensor, strain_tensor, rotation_vector)

            # ensure that we are not accidentially applying compression (causing the sheet to buckle)
            # this is only a concern in angle_testing, because the applied strain tensors are randomized, so compression is possible.
            if self.angle_testing:
                self.check_buckle()

            # checks when stress drops to detect fracture - this is just an on the fly check because we don't have principal stresses calculated
            strength, _ = self.find_fracture(stress_tensor[:(iters+1)])  # note: changed from just dominant direction stress to accomidate the multi-axis check

            # if we get a strength value, that means fracture was detected and we can leave the loop
            if strength[0] is not None:
                # run the simulation for a few more thermos to visualize fracture
                for i in range(5):
                    step += self.thermo  # update thermo (cuz we are no longer in the big loop)
                    if step == self.sim_length:
                        break
                    iters, stress_tensor, step_vector, pressure_tensor, strain_tensor, rotation_vector = self.run_step(
                        step, iters, stress_tensor, step_vector, pressure_tensor, strain_tensor, rotation_vector)
                    
                stress_tensor = stress_tensor[:(iters+1)]
                step_vector = step_vector[:(iters+1)]
                pressure_tensor = pressure_tensor[:(iters+1)]
                strain_tensor = strain_tensor[:(iters+1)]
                rotation_vector = rotation_vector[:(iters+1)]

                return stress_tensor, pressure_tensor, step_vector, strain_tensor, rotation_vector

        # if we never detected fracture, return the tensors anyway
        return stress_tensor, pressure_tensor, step_vector, strain_tensor, rotation_vector
    
    def check_buckle(self, tol=5):
        """
        - This checks if the sheet is buckling/bending by finding the highest absolute value of the z-coordinate of an atom
        - This is basically a check to ensure that compression isn't being applied to the sheet by mistake - this is more
        important for when we are angle testing, because random shear applications could cause compression, giving rise 
        to ficticious stress readings, potentially confusing our ML model.

        Parameters:
            - tol (float): Default: 5 angstroms. This is just how far away an atom has to be from the z-axis to flag this check. 
        """
        # extract the highest and lowest z atom
        zmax = self.lmp.extract_variable("zmax", None, 0)
        zmin = self.lmp.extract_variable("zmin", None, 0)
        height = max(abs(zmax), abs(zmin))
        # check to make sure we stay below tolerance
        if height > tol:
            # if we didn't stay below tolerance, abort the sim.
            print(f"[Rank 0] Buckling detected, max height of {height}. Aborting this sim.")
            sys.stdout.flush()
            self.comm.Abort()  # Clean exit for parallel jobs

    def extract_pressure(self):
        """
        - This extracts the pressures in x, y, and z directions from LAMMPS

        Returns:
            - list(float): Vector of length 3 representing the pressure in each direction.
        """
        xx = self.lmp.extract_variable("p_x", None, 0)
        yy = self.lmp.extract_variable("p_y", None, 0)
        zz = self.lmp.extract_variable("p_z", None, 0)
        return [xx, yy, zz]
    
    def extract_stress(self):
        """
        - This extracts the stresses in x, y, z, xy, xz, and yz directions from LAMMPS

        Returns:
            - np.array(float): Vector of length 6 representing the stress in each direction. This is the flattened stress tensor.
        """
        xx = self.lmp.extract_variable("stress_xx", None, 0)
        yy = self.lmp.extract_variable("stress_yy", None, 0)
        zz = self.lmp.extract_variable("stress_zz", None, 0)
        xy = self.lmp.extract_variable("stress_xy", None, 0)
        xz = self.lmp.extract_variable("stress_xz", None, 0)
        yz = self.lmp.extract_variable("stress_yz", None, 0)

        arr = np.array([xx, yy, zz, xy, xz, yz])
        return arr
    
    def compute_strain(self, current_timestep):
        """
        - This computes the strain at any given timestep. It uses the self.erate_timestep, which is computed from
        'self.calculate_erateTimestep()' upon __init__. 

        Parameters:
            - current_timestep (int): This is a timestep number (ex. timestep 200) (basically whatever thermo it is).

        Returns:
            - np.array(float): Vector of length 6 representing the flattened strain tensor at that given timestep.
        """
        x_strain = self.erate_timestep[0] * current_timestep
        y_strain = self.erate_timestep[1] * current_timestep
        z_strain = self.erate_timestep[2] * current_timestep
        xy_strain = self.erate_timestep[3] * current_timestep
        xz_strain = self.erate_timestep[4] * current_timestep
        yz_strain = self.erate_timestep[5] * current_timestep
        return np.array([x_strain, y_strain, z_strain, xy_strain, xz_strain, yz_strain])
    
    # helper function for run_simulation... runs one thermo
    def run_step(self, step, iters, stress_tensor, step_vector, pressure_tensor, strain_tensor, rotation_vector):
        """
        - This is a helper function for 'self.run_simulation()'. 
        - It essentially runs one thermo. This means, if the thermo is 500, it tells LAMMPS to run 500
        timesteps then report back to us with the results. 

        Parameters:
            - step (int): Current timestep we are on (so if thermo=500, step would go 0, 500, 1000, etc.)
            - iters (int): This is essentially how many iterations we have done so far, so regardless of thermo it goes 0, 1, 2... etc.
            - stress_tensor (np.array[float]): This is an array of width 6 and length of number of maximum thermos. Each row
            is the flattened stress tensor for that timestep, and as we run steps we fill this out (it starts all zeros)
            - step_vector (np.array[int]): This is a vector that tells what actual timestep it is. for instance, running three timesteps
            of thermo=500 would yield a step_vector of [0, 500, 1000]. 
            - pressure_tensor (np.array[float]): This is an array of width 3 and length of number of maximum thermos. Each row
            represents the pressure in the x, y, z directions for that timestep, and as we run steps we fill this out (it starts all zeros)
            - strain_tensor (np.array[float]): This is an array of width 6 and length of number of maximum thermos. Each row
            is the flattened strain tensor for that timestep, and as we run steps we fill this out (it starts all zeros)
            - rotation_vector (np.array[float]): This is a vector that describes the rotation of the lattice (from deformation) at any given thermo

        Returns:
            - This returns every parameter it takes in (except for step), and it's job is to basically iterate these input parameters,
            driving the simulation one thermo further (and saving the data to the respective variables)
        """
        iters += 1

        self.lmp.command(f"run {self.thermo} pre yes post no")
        step_vector[iters] = step + self.thermo
        stress = self.extract_stress()  # extract stress tensor
        pressure = self.extract_pressure()
        strain = self.compute_strain(step_vector[iters])
        stress_tensor[iters] = stress
        pressure_tensor[iters] = pressure
        strain_tensor[iters] = strain
        rotation_vector[iters] = self.compute_rotation(strain)
        return iters, stress_tensor, step_vector, pressure_tensor, strain_tensor, rotation_vector
    
    def get_strength_info(self, stress_index, principal_stresses):
        """
        - This is a helper function for 'self.find_fracture()'. It picks up when the following criterion have already been satisfied:
            1. There were enough thermos stored to actually check for fracture in the first place
            2. It has been detected that at least one principal direction is not increasing in stress values
        - In this case, it takes in the stress tensor as well as the direction in which it was determined the stress is no longer increasing.
        - It then tests to ensure that there are peaks of the stress along this direction. If not, we return None meaning there was 
        no fracture detected.
        - Next compares with the fracture_window to ensure that this peak is not just noise, and there was a significant drop in stress. 
        - At this point, fracture is certain, so we process and return the proper values. 

        Parameters:
            - stress_index (int): This is the direction in which the stress has been deemed "not increasing". It's values can 
            only be 0 or 1. It is 0 if the "not increasing" direction is the dominant principal direction, and it is 1 if the
            "not increasing" direction is the secondly dominant principal direction. 
            - principal_stresses (np.array[float]): This is a 6 wide, n_thermos long array of the principal stresses so far. 
            principal_stresses[:, 0] will give us the dominant direction stress wrt time, and we can use this to test the dominant direction
            stress. This also works with the strain tensor, as it is the same principle. (same input as 'self.find_fracture()').
        
        Returns:
        NOTE: Just like 'self.find_fracture()', this returns None if fracture was not detected. The following explainations are for 
        the case when fracture is detected.
            - strength (np.array[float]): This is an array of length 3 that tells you the strength value for this simulation in principal stress
            space. This finds the peak stress and sets that as strength.
            - fracture_timestep (int): This is the exact timestep that corresponds with the maximum principal stress value
            - fracture_index (int): This is the index in the stress tensor that corresponds with fracture
        """

        peaks, _ = find_peaks(principal_stresses[:, stress_index])

        if len(peaks) == 0:
            # No peaks found
            strength = [None, None, None]
            fracture_timestep = None
            return strength, fracture_timestep, None
        
        else:
            fracture_index = peaks[np.argmax(principal_stresses[:, stress_index][peaks])]  # find the index of the highest peak

            # If the fracture index is calculated as one of the last values
            if fracture_index + 10 >= len(principal_stresses):
                # then the low testpoint is just the last data in the simulation
                low_testpoint = np.min(principal_stresses[:, stress_index][-10:])
            else:
                low_testpoint = np.min(principal_stresses[:, stress_index][fracture_index:fracture_index+10])

            # see if this "fracture" is high enough to be considered (or is it just noise?)
            # for fracture, the (peak - window) must be greater than the minimum point for that stress index
            if (principal_stresses[fracture_index, stress_index] - self.fracture_window < low_testpoint):
                strength = [None, None, None]
                fracture_timestep = None
            else:
                # Now we are confident there is fracture, let's return the proper values
                strength = principal_stresses[fracture_index]
                fracture_timestep = fracture_index * self.thermo
                print("FRACTUREEEEEE")  # print a note for fun

        return strength, fracture_timestep, fracture_index

    def find_fracture(self, principal_stresses, give_crit_strain=False):
        """
        - This detects the point of fracture as a sudden sustained drop in stress. A lot is in here to filter out potential noise. 
        - The idea here is to have the previous 10 thermos be our testcase, and the previous 15 before that be what we're testing against.
        If the average of the previous 10 is lower than the average of the 15 before that, fracture has occured and we should find the peaks

        Parameters:
            - principal_stresses (np.array[float]): This is a 6 wide, n_thermos long array of the principal stresses so far. 
            principal_stresses[:, 0] will give us the dominant direction stress wrt time, and we can use this to test the dominant direction
            stress. This also works with the strain tensor, as it is the same principle. 
            - give_crit_strain (bool): Default is False. If true, it will also return the critical strain value.

        Returns:
        NOTE: This only returns values when fracture is detected. If fracture is not detected, it returns a bunch of None's. This is 
        useful in 'self.run_simulation()', as we know to continue running the simulation until None is not returned. This is again used in 
        'self.__init__()' after the simulation finishes to handle any cases where fracture never occurs. 
            - strength (np.array[float]): This is an array of length 3 that tells you the strength value for this simulation in principal stress
            space. This finds the peak stress and sets that as strength.
            - fracture_timestep (int): This is the exact timestep that corresponds with the maximum principal stress value
            - crit_strain (np.array[float]): This is an array of length 3 that tells you the strain value at the maximum principal stress point. 
            This is the critical strain for this simulation. This output is optional (we don't care about it when testing for fracture every
            timestep in 'self.run_simulation()').
        """
        strength = [None, None, None]
        fracture_timestep = None
        # Not enough thermos to actually check
        if len(principal_stresses[:, 0]) < 25:
            strength = [None, None, None]  # must be a list of None's for later
            fracture_timestep = None
        # Now there's enough timesteps to check
        else:
            mean_last_10 = sum(principal_stresses[:, 0][-10:]) / 10
            mean_15_before = sum(principal_stresses[:, 0][-25:-10]) / 15

            # get the same for principal stress 2 (in case weird behavior)
            mean_last_10_2 = sum(principal_stresses[:, 1][-10:]) / 10
            mean_15_before_2 = sum(principal_stresses[:, 1][-25:-10]) / 15

            sig0_intact = mean_last_10 >= (mean_15_before * 0.9)  # slightly degrade to reduce false positives
            sig1_intact = mean_last_10_2 >= (mean_15_before_2 * 0.9)

            # if we are still increasing on average in both directions, no fracture yet
            if sig0_intact and sig1_intact:
                strength = [None, None, None]
                fracture_timestep = None
            
            # we have detected a significant drop on average (fracture)
            else:
                if not sig0_intact:
                    strength, fracture_timestep, fracture_index = self.get_strength_info(0, principal_stresses)
                
                if not sig1_intact and fracture_timestep is None:
                    strength, fracture_timestep, fracture_index = self.get_strength_info(1, principal_stresses)

                if give_crit_strain:
                    crit_strain = self.principalAxes_strain[fracture_index]
                    return strength, crit_strain, fracture_timestep
                
        return strength, fracture_timestep
    
    def compute_principal_StressStrain(self, tensor, return_theta=False):
        """
        - This computes the principal values of the stress or strain tensor, regardless of what you give it for each timestep in the tensor

        Parameters:
            - tensor (np.array[float]): This is an array with a width of 6 and length of the number of thermos saved. Each row in this array
            is the flattened tensor in which you aim to calculate the principal values of
            - return_theta (bool): Default is False. If true, it will return thetas

        Returns:
            - principal_values (np.array[float]): This is an array of width 3 and the length of the original array. Each row in this array are
            the three principal values of the original tensor. 
            - thetas (np.array[float]): This is a vector of the angle that the dominant principal direction makes with the x-axis, mirrorered
            between [0, 90] degrees, for each thermo in the original tensor given. This means 91 degrees would get flipped to 89 degrees, etc.
        """
        num_timesteps = tensor.shape[0]
        principal_values = np.zeros((num_timesteps, 3))

        if return_theta:
            thetas = np.zeros(num_timesteps)

        for i in range(num_timesteps):
            # get whole stress tensor for this timestep
            matrix = np.array([[tensor[i, 0], tensor[i, 3], tensor[i, 4]],
                                    [tensor[i, 3], tensor[i, 1], tensor[i, 5]],
                                    [tensor[i, 4], tensor[i, 5], tensor[i, 2]]])
            
            # compute principal stresses and principal directions - this is sorted from lowest to highest eigval every time, 
            eigvals, eigvecs = np.linalg.eigh(matrix)
            principal_values[i] = eigvals[::-1]
            eigvecs = eigvecs[:, ::-1]

            # if we want the angle of the dominant principal stress (for angle data generation)
            if return_theta:
                # Get dominant eigenvector (corresponding to max principal value)
                v = eigvecs[:, 0]
                # Project into x-y plane and compute angle from x-axis
                theta_deg = np.degrees(np.arctan2(v[1], v[0])) % 180
                theta_deg = min(theta_deg, 180 - theta_deg)
                thetas[i] = theta_deg
        
        if return_theta:
            return principal_values, thetas

        return principal_values
    
    def compute_rotation(self, strain_tensor):
        """
        - This computes the rotation portion of the polar decomposition of the deformation gradient, which is important 
        to obtain an accurate calculation of the angle of the state of stress with respect to the lattice. 
        - It uses the prescribed strain rate tensor and the time to get F, then takes the polar decomp to get R.

        Parameters:
            - strain_tensor (np.array[float]): This is a flattened array of length 6 that represents the strain tensor for any given step.
        
        Returns:
            - angle (float): This is the rotation of the lattice (in degrees) corresponding to the given strain tensor.
        """

        # Deformation gradient (engineering form)
        F = np.array([[1 + strain_tensor[0], strain_tensor[3], strain_tensor[4]], 
                      [0.0, 1 + strain_tensor[2], strain_tensor[5]], 
                      [0.0, 0.0, 1 + strain_tensor[3]]], dtype=float)
        
        # Right polar: F = R @ U
        R, U = polar(F, side='right')

        # Lattice rotation angle (deg) from first column of R
        angle = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
        return angle


    ############ END SIMULATION ENGINE ############
    ############ BEGIN DEFECT ADDITION ############


    def introduce_defects(self):
        """
        - This puts the defects into the sheet randomly
        - Currently only supports single vacancy and double vacancy defects
        - This basically does all of the logic to randomly remove SV and DV atoms and gives the commands directly to LAMMPS
        - Throughout these functions, we constantly update the deleted atom id's list to ensure that we do not accidentily
        choose an atom to remove that has already been removed. 
        """
        # initialize an array that will store the atom id's that have already been deleted.
        self.deleted_ids = np.array([], dtype=int)

        for defect_type, defect_percentage in self.defects.items():
            delete_ids = self.delete_atoms(defect_type, defect_percentage)
            self.deleted_ids = np.concatenate([self.deleted_ids, delete_ids])  # store which atoms we deleted

        if self.deleted_ids.size > 0:
            # Convert all IDs into a LAMMPS-compatible string
            id_str = ' '.join(str(int(id)) for id in self.deleted_ids)

            # Define the group once with all IDs and delete them at once
            # we must delete at once or else lammps will reconfigure the atom id's, screwing up our neighbor-finding algorithm
            self.lmp.command(f"group to_delete id {id_str}")
            self.lmp.command("delete_atoms group to_delete")

        print("Removed all of these atoms:", self.deleted_ids)

    # Pick atoms to delete for single/double vacancies
    def delete_atoms(self, defect_type, delete_percentage):
        """
        - This is a helper function for 'self.introduce_defects()'
        - This chooses the atoms to delete based on what defect type is chosen and how many atoms to delete.

        Parameters:
            - defect_type (str): Can be either 'SV' or 'DV'... This tells us whether to remove one atom or two adjacent atoms.
            - delete_percentage (float): Tells us what percentage of the total number of atoms should be removed from this defect type.

        Returns:
            - delete_ids (np.array[int]): This is the id's of all of the atoms deleted (so we can keep track)

        Raises:
            - ValueError: This happens if the defect type inputted does not match either 'SV' or 'DV'
        """
        # this gets the atom id, x, y, and z positions for each atom in the sheet
        atom_positions = self.sheet.extract_atom_positions()

        total_atoms = len(atom_positions)
        n_delete = int(total_atoms * (delete_percentage / 100))  # find how many atoms to delete

        # atom_positions is a Nx4 array: [id, x, y, z], so this gets the ids
        ids = atom_positions[:, 0]

        # Randomly choose atoms to delete
        np.random.seed(self.defect_random_seed)
        if defect_type == "SV":
            filtered_ids = np.setdiff1d(ids, self.deleted_ids, assume_unique=True)  # ensure we don't pick already deleted ids
            delete_ids = np.random.choice(filtered_ids, n_delete, replace=False)
            self.deleted_ids = np.concatenate([self.deleted_ids, delete_ids])  # update so we keep track of what id's we deleted
        
        elif defect_type == "DV":
            filtered_ids = np.setdiff1d(ids, self.deleted_ids, assume_unique=True)  # ensure we don't pick already deleted ids
            original_ids = np.random.choice(filtered_ids, int(round(n_delete / 2)), replace=False)
            self.deleted_ids = np.concatenate([self.deleted_ids, original_ids])  # update so we keep track of what id's we deleted
            delete_ids = self.add_neighbors(original_ids)
        
        else:
            raise ValueError(f"Unsupported defect type: '{defect_type}'")

        return delete_ids.astype(int)
    
    def add_neighbors(self, ids):
        """
        - This is a helper function for 'self.delete_atoms()', particularly when we have a DV defect type. 
        - It basically chooses a random neighbor to delete alongside each id that is already getting deleted.

        Parameters:
            - ids (np.array[int]): This is a list of randomly chosen atom id's to delete.

        Returns:
            - np.array[int]: This is a list of both the original id's that were chosen to get deleted, AND 
            the neighbor id's that will now get deleted to support the DV defect type
        """
        extra_deletions = []
        
        for id in ids:
            neighbors = self.find_neighbors(id)  # get the list of neighbors

            filtered_neighbors = np.setdiff1d(neighbors, self.deleted_ids, assume_unique=True)  # ensure we don't pick a neighbor that is already deleted
            
            if filtered_neighbors.size > 0:
                # as long as there's a valid neighbor to pick, randomly pick one
                chosen_neighbor = np.random.choice(filtered_neighbors, 1, replace=False)
                extra_deletions.append(chosen_neighbor)
                self.deleted_ids = np.concatenate([self.deleted_ids, chosen_neighbor])
                
            # if there was no valid neighbor, that means the neighbor was already removed so we just move on
            else:
                pass 
        
        flat_extras = np.concatenate(extra_deletions)

        return np.concatenate([ids, flat_extras])

    def find_neighbors(self, id):
        """
        - This is a helper function for 'self.add_neighbors()'.
        - This function finds the neighbors of an atom based on the atom id, whether there be one, two, or three neighbors.
        - This algorithm relies on the specific algorithm used to create the graphene sheet and assign the atom ids that
        is used in create_graphene.py. This will not work if the graphene sheet is not setup with that atomid format. 

        Parameters:
            - id (int): This is the atom id of the atom in question, which is the one we want to find the neighbors of. 

        Returns:
            - neighbors (np.array[int]): This is a list of all of the neighbors for that atom id. This could either be of length 
            1, 2, or 3 depending on where the original atom is situated (and if any of the neighbors had already been removed)
        """
        neighbors = []
        # get the id of the top atom
        top_id = ((id + self.sheet.y_atoms - 1) // self.sheet.y_atoms) * self.sheet.y_atoms
        block_num = top_id / self.sheet.y_atoms  # see where we are

        # if we are an odd column
        if (block_num % 2 == 1):
            # as long as we are not on bottom edge, we can add id below us as a neighbor
            if ((id - 1) % self.sheet.y_atoms) != 0:
                neighbors.append(id - 1)

            # as long as we are not on the top edge, we can add id above us as a neighbor
            if (id % self.sheet.y_atoms) != 0:
                neighbors.append(id + 1)

            if (id % 2) == 0:
                # since we are in an odd column, an even id means we are on the right
                # check to ensure that we aren't fully on the right (no neighbor there)
                if top_id != (self.sheet.y_atoms * self.sheet.x_atoms - self.sheet.y_atoms):
                    neighbors.append(id + 3 * self.sheet.y_atoms)
            
            # if our id is odd, we will never be fully on boundary in a column, so we are good
            if (id % 2) == 1:
                    neighbors.append(id + self.sheet.y_atoms)

        # if we are an even column
        else:
            # as long as we are not on bottom edge, we can add id below us as a neighbor
            if ((id - 1) % self.sheet.y_atoms) != 0:
                neighbors.append(id - 1)

            # as long as we are not on the top edge, we can add id above us as a neighbor
            if (id % self.sheet.y_atoms) != 0:
                neighbors.append(id + 1)

            if (id % 2) == 0:
                # since we are in an even column, an even id means we are on the left
                # check to ensure that we aren't fully on the left (no neighbor there)
                if top_id != (2 * self.sheet.y_atoms):
                    neighbors.append(id - 3 * self.sheet.y_atoms)
                
            # if our id is odd, we will never be fully on boundary in a column, so we are good
            if (id % 2) == 1:
                    neighbors.append(id - self.sheet.y_atoms)

        return neighbors

        
class Relaxation:
    def __init__(self, comm, rank, sheet_path, x_atoms, y_atoms, 
                 sim_length=120000, timestep=0.0005, thermo=1000, nvt_percentage=0.2, detailed_data=False):
        self.comm = comm
        self.rank = rank
        self.sheet_path = sheet_path
        self.x_atoms = x_atoms
        self.y_atoms = y_atoms
        self.sim_length = sim_length
        self.timestep = timestep
        self.thermo = thermo
        self.nvt_percentage = nvt_percentage
        self.detailed_data = detailed_data

        self.sheet = GrapheneSheet(sheet_path, x_atoms, y_atoms)

        lmp = lammps(comm=comm)  # initialize lammps
        self.lmp = lmp  # store this instance of lammps so we can refer to it throughout the class

        self.setup_lammps()  # puts all needed variables in lammps and initializes file
        self.lmp.command('velocity all create 100.0 20 mom yes rot yes dist gaussian')
        nvt_command = "fix 1 all nvt temp 100.0 273.0 10.0"
        nvt_time = int(self.sim_length * self.nvt_percentage)
        self.run_command(nvt_command, nvt_time, cleanup_command="unfix 1")

        npt_command = "fix 2 all npt temp 273.0 273.0 0.1 x 0.0 0.0 0.1 y 0.0 0.0 0.1"
        self.run_command(npt_command, int(self.sim_length - nvt_time))
        self.lmp.command(f"write_data {sheet_path}")  # overwrite the stiff data with the relaxed data

    def setup_lammps(self):
        """
        - This sets up the LAMMPS simulation and imports all of the important variables to LAMMPS
        - This is called in 'self.__init__()' directly after this instance of LAMMPS is created
        """
        self.lmp.command(f"variable vol equal {self.sheet.volume}")
        self.lmp.command(f"variable sheet_lx equal {self.sheet.Lx}")
        self.lmp.command(f"variable sheet_ly equal {self.sheet.Ly}")
        self.lmp.command(f"variable sheet_lz equal {self.sheet.Lz}")
        self.lmp.command(f"variable timestep equal {self.timestep}")

        self.lmp.command(f"variable datafile string {self.sheet.datafile_name}")

        if self.detailed_data:
            path, name = split_path(self.sheet_path)
            self.lmp.command(f"dump 1 all custom {self.thermo} {path}dump.{name} id type x y z")

        self.lmp.file("in.relax_py")

    def run_command(self, fix_command, num_timesteps, cleanup_command=None):
        self.lmp.command(fix_command)
        thermos_to_run = int(num_timesteps / self.thermo)
        for steps in range(thermos_to_run):
            self.lmp.command(f"run {self.thermo} pre yes post no")

        if cleanup_command is not None:
            self.lmp.command(cleanup_command)

# can get the raw filepath as well as the filename, where the file name is something like data.name
def split_path(s):
    if '/' in s:
        before, after = s.rsplit('/', 1)
        before += '/'
    else:
        before, after = '', s

    if '.' in after:
        after = after.split('.', 1)[1]

    return before, after
