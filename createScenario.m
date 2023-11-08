function [X, Y, fin_grid, cap_ind, cap_ind_occl, store_vec] = createScenario(N, x_step, mu, sigma, num_cap, ...
    D, tau, R, f, A, phi_C, phi_V, n_iter, cons_rate, firing_rate, ratio_neur, ...
    radius, boundary, seed, center, epsilon, ip, inf_firing_rate)

%   Inputs: 
%       N (scalar) - Number of grid points
%       x_step (scalar) - Distance between grip points (um)
%       mu (scalar) - Mean (Gaussian) distance between capillaries
%       sigma (scalar) - Standard deviation (Gaussian) of the distance between capillaries
%       num_cap (scalar) - Number of capillaries
%       D (scalar) - Diffusivity Coefficient (um^2/seconds)
%       tau (scalar) - Time step for iteration (seconds)
%       R (scalar) - Proportion of venules occupying tissue (A.u)
%       f (scalar) - Frequency of oscillations (Hz)
%       A (scalar) - Amplitude of oscillations (mmHg)
%       phi_C (scalar) - Oxygen pressure at capillaries (mmHg)
%       phi_V (scalar) - Oxygen pressure at venules (mmHg)
%       n_iter (scalar) - Number of iterations for algorithm
%       cons_rate (scalar) - Oxygen consumption rate (mmHg/Hz)
%       firing_rate (scalar) - Average firing rate of neurons (Hz)
%       ratio_neur (scalar) - Proportion of neuronal volume in tissue (A.u)
%       radius (scalar) - Radius of infarction (um^3)
%       boundary (Boolean) - If true then set boundaries equal to capillary
%       pressure (phi_C), otherwise set no flux boundary conditions
%       seed (scalar) - Set random seed
%       center (1x2 matrix) - Grid point of center of infarction
%       epsilon (scalar) - Error value in successive iterations until
%       convergence
%       ip (scalar) - Initial oxygen pressure in tissue
%       inf_firing_rate (scalar) - Firing rate within infarction
%
%   Outputs:
%       Note: N_c = number of capillaries in tissue.
% 
%       X (N x N matrix) - X values over N by N tissue grid
%       Y (N x N matrix) - Y values over N by N tissue grid
%       fin_grid (N x N matrix) - Oxygen concentration over entire tissue
%       in steady state
%       cap_ind (N_c x 2 matrix) - Indices of capillaries in tissue, X and Y
%       coordinate, respectively
%       cap_ind_occl (? x 2 matrix) - Indices of occluded capillaries
%       store_vec (? x 1 matrix) - O2 pressure at center of Area of
%       Potential infarction (API) up until convergence to steady state

rng(seed)

% Generate randomized grid with capillary sources using Monte Carlo method
[X, Y, cap_ind] = genGrid(N, x_step, mu, sigma, num_cap);

% Generate the indices of occluded capillaries
cap_ind_occl = createInfarction(N, x_step, cap_ind, radius);

% Iterate the grid of PO2 until reaching steady state (or max iterations)
[fin_grid, store_vec] = DiffADI(N, cap_ind_occl, D, x_step, tau, R, f, A, phi_C, ...
        phi_V, n_iter, cons_rate, firing_rate, ratio_neur, 0, boundary, center, epsilon, ip, radius, inf_firing_rate);

end