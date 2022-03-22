# 2D VTOL Trajectory Planning Demo

The script solves the problem of planning of the tic-toc maneuver for a 2D VTOL helicopter:

![image](./fig/anim.gif)

## VTOL dynamics
VTOL dynamics is represented in symbolic with the help of Casadi symbolic algebra library. The quadrotor is assumed to have two configurable parameters (the dataclass `Parameters` in `dynamics.py`)
 * `epsilon` is the motors dihedral angle; the left motor inclined ccw by the angle `epsilon`, while the right motor inclined cw by `epsilon`.
 * `gravity` is the gravity acceleration

## Reduced dynamics solver
The script `construct_singular_trajectory.py` implements methods of solving singular reduced dynamics. It also has finds a hardcoded periodic trajectory (TODO: make this configurable).

## Construct transverse coordinates and find linearized transverse dynamics
For the given dynamics object and evaluated trajectory the function `get_trans_lin` constructs transvrese coordinates and corresponding LTV system. (TODO: split into two algos?)

## Construct feedback controller for the linearized system
The function `linsys_feedback.main` finds time varying LQR for the given linearized transverse dynamics.

## Transverse Linearization Feedback
The class `TransversePeriodicFeedback` implements the transverse linearization feedback.

## Simulation
The function `run_simulation` computes the closed loop system and saves results into a datafile.

## Plots
The function `plot_trajectory_projections` displays the change in the phase coordinates of the system over time.

## Animation
The class `AnimateQuadrotor`  makes animation based on the given simulation results.
