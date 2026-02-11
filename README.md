# Stochastic, Anisotropic Strength Surface of Monocrystalline Graphene

## 0) Environment Setup:
A conda environment makes this setup the simplest, and here I have the following packages installed:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `lammps`
  - `scipy`
  - `sklearn`
  - `mpi4py`
  - `filelock`
  - `plotly`
  - `joblib`
  - `pysr`
  - `pickle`
  Only necessary for the notebooks:
  - `ipywidgits`
  - `jupyter`
Only necessary for the copulas scripts:
  - `seaborn`
  - `copulas`

## 1) LAMMPS Input Files:
  - `CH.airebo`: AIREBO potential function definition
  - `in.relaxation`: Toy LAMMPS input file to understand how to run the relaxation step for lammps
  - `in.sandbox`: Toy LAMMPS input file to play around with deforming sheet, adding defects, etc.
  - **`in.relax_py`: LAMMPS input file that `deform_graphene.py` uses to drive relaxation simulations**
  - **`in.deform_py`: LAMMPS input file that `deform_graphene.py` uses to drive fracture simulations**

## 2) Creating Graphene:
  - **`create_graphene.py`: This is a simple script that writes a LAMMPS data file, used for input into the LAMMPS input files. Creates the data file for specified graphene sheet size.**

## 3) Driving Simulations:
  - **`deform_graphene.py`: This is the main driver of all simulations for this project.**
    - The `GrapheneSheet` class allows us to store graphene sheets simply in the code, storing their filepath, size, volume, etc.
    - The `Relaxation` class allows us to run automated relaxation simulations to bring any graphene sheet to equillibrium, then store that for subsequent simulations.
    - **The `Simulation` class hosts everything needed to run one fracture simulation, including setup, fracture logic, data storage, etc.**
  - Simulation results are stored in some directory (defined locally, in a file called `local_config.py` and again in `local_config.lmp`), in csv format. Here, all input and output results are stored, along with a simulation id number. This simulation id number is used to link csv results to "detailed data" results, such as full .dump files of simulations or timestep-level grainularity of variables. This is usually stored in a file called `all_simulations.csv`.

## 4) Running Large-Scale Simulations:
  - The bash scripts listed below all orchestrate some level of large-scale simulation pipeline, where scripts such as `run_all.sh`, `run_manual.sh`, and `run_specific.sh` handle user inputs, then throttle jobs to other scripts.
  - **`run_surface.sh` takes inputs from `run_all.sh` to run all principal stress ratios required for any particular strength surface.**
  - **All of these bash scripts feed into `run_one.sh`, which actually submits a SLURM job by taking in the necessary inputs, and passing them to `one_sim.py`.**
  - `one_sim.py` takes in command line arguments to run one specified simulation. This works well for parallelization, which is why it is used here.
  - `run_relaxation.sh` works in a very similar way as `run_all.sh`, but it instead takes in the user parameters for relaxation, then throttles jobs to `one_relaxation.sh`, which launches SLURM jobs to `one_relaxation.py`.
  - `serial_MPI_datacollect.py` is a largely outdated file that allows for direct class definitions in `deform_graphene.py`. This was mostly used for early-stage testing.

## 5) Data Upkeep:
  - **`filter_csv.py`: Filters the `all_simulations.csv` (which stores simulation results) in whatever way the user defines. The function `filter_data()` is used many times in other scripts.**
  - `check_sims.py`: Utilizes `filter_data()` to ensure that all simulations that we thought completed actually completed and are stored.
  - `plot_StrengthSurface.py`: Can plot 2D and 3D strength surfaces, uses `filter_data()` to allow the user to plot exactly what they want, all you need is to specify the dataset.
  - `plot_timestep_data.py`: Admittedly, this script is not the best, but `plot_many_detailed()` allows us to plot the detailed data as a function of time (or whatever we want) during one single simulation, and overlay simulations if desired.

## 6) Stress-Strain Neural Network:
  - `run_angle_tests.sh`: This automates the data creation for shorter, "angle testing" simulations, where we ran the simulation just long enough to gather meaningful stresses, then returned the results. This triggered certain flags in `deform_graphene.py` to setup `all_simulations.csv` differently for this purpose.
  - `stress_strain_map.py`: Once the dataset creation was complete, this script was used to fit a MLP Regressor to the data, mapping strain rate tensor to stress tensor. Note some commented out parts -- this was from testing which included GradientBoosting, Symbolic Regression, using different input/output features, etc. The bottom commented out area holds some grid search logic as well as some plotting to visualize performance.
  - `stress_strain.ipynb`: This is just a simple sandbox to probe the results of the angle testing dataset, ensuring proper coverage across all stresses.
  - `ss_model_tester.py`: This tests the performance of a stress-strain mapping model without the need for retraining.
  - `inverse_strain.py`: Once the nerual network holds acceptable accuracy, this takes in the model and a desired stress ratio and $\theta$, and it employs the model to give a strain rate tensor that will keep us at the desired strain rates, while also giving that ratio and $\theta$.
  - `create_straintable.sh`: This uses `inverse_strain.py` to call the model for target ratios and $\theta$'s that are most common (ex: angles every 10$\circ$, ratios every 0.1. It gathers these resulting strain tensors and stores them in strain_table.csv, allowing for faster access when desired (while running simulations).

## 7) Strength Surface Model Fitting:
  - `DP_model.py`: This fits the 2-parameter Drucker-Prager model to a set of strength surface data and stores the $\alpha$, $k$, and seed in the csv. It also is now expanded to fit $\alpha$ and $k$ as functions of $\theta$, but that expansion is pretty old and likely outdated. Use `fit_3D_DP.py` to fit an angular surface. This is mostly just used to fit the 2D surfaces for the pristine data.
  - **`fit_3D_DP.py`: This fits the 2-parameter Drucker-Prager model to a set of strength surface data, where $\alpha$ and $k$ are functions of $\theta$. It fits $\alpha$ and $k$ as $n^{th}$ order fourier series, and saves all of the coefficients in the csv. Allows the user to input lambda, fourier_order, and defects directly from the command line, working well with bash scripts for mass fitting.**
  - `DP_model_probing.py`: Given a csv with Drucker Prager parameters (for 2D or 3D), this generates plots and calculates the errors without having to refit the least squares model. This is a bit hard-coded from testing, but the logic is there if we care to improve it in the future.
  - **`fit_all.sh`: We can use this to call `fit_3D_DP.py` many times, making it way easier to fit in multiple ways (like using different regularization parameters, fourier orders, etc), and uses MPI to make this quickly.**
  - This gives us nice strength surface parameters in a csv format.

## 8) Copula Fitting:
  - This portion of the project is outdated. We no longer fit copula models to the strength surface parameters. It could still be used, but simply for the scaler parameters (aka, $\alpha$ and $k$ do not vary with $\theta$).
  - `copulas.R`: This does basically all of the work of fitting the copulas.
  - `copula_analysis.py`: This takes in the params saved from fitting the copulas in R, verifies that the distributions look good, then does JS Divergence to compare the distributions.
  - `copulas.ipynb`: Simple notebook that I used to learn about copulas, particularly the `copulas` package in python, and to test whether it would work for this application.

## 9) Stochastic Modeling of Strength Surface Parameters
  - **`probabilistic_results.py`: This houses everything we need to take in a dataset of strength surface fit parameters, reduce dimensionality, fit, sample, and project back. This includes generating confidence intervals and means in strength space and parameter space. It also allows for intermediate inturruption to view metrics like PCA and the latent space. This is how to perform all of the stochastic results.**

## 10) Misc.
  - `sandbox.py` and `sandbox.ipynb` were exactly that... sandboxes. `sandbox.py` ended up by turning into a bit of a misc. plotting script used for the paper.

## Notes:
  - If you are getting errors based on `matplotlib`, just comment out the following code... it shows up in a lot of scripts where plotting occurs. All it does is make the matplotlib text LaTeX, but if you don't have that installed it will throw errors:
  ```
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Computer Modern Roman']

    mpl.rcParams['text.latex.preamble'] = r"""
    \usepackage{amsmath}
    \usepackage{amssymb}
    \usepackage{xcolor}
    """

    mpl.rcParams['axes.unicode_minus'] = False
    plt.rcParams.update({
        'font.size': 10,      # match normalsize
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8, # slightly smaller like LaTeX
        'text.usetex': True,
    })
  ```


