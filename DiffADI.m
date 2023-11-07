function [grid, store_vec] = DiffADI(N, cap_ind, D, x_step,...
    tau, R, f, A, phi_C, phi_V, num_iter, cons_rate, firing_rate, ratio_neur,...
    perc_occl, boundary, center_index, epsilon, ip, radius, inf_firing_rate)

%   Inputs: 
%       Note: N_c = number of capillaries
% 
%       N (scalar) - Number of grid points
%       cap_ind (N_c x 2 matrix) - Indices of capillaries in grid
%       D (scalar) - Diffusivity Coefficient (um^2/seconds)
%       x_step (scalar) - Distance between grip points (um)
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
%       perc_occl - Percentage of total capillaries occluded (Note: outside 
%       of occlusion from raidus of API)
%       boundary (Boolean) - If true then set boundaries equal to capillary
%       pressure (phi_C), otherwise set no flux boundary conditions
%       center_index (1x2 matrix) - Grid point of center of infarction
%       epsilon (scalar) - Error value in successive iterations until
%       convergence
%       ip (scalar) - Initial oxygen pressure in tissue
%       radius (scalar) - Radius of infarction (um^3)
%       inf_firing_rate (scalar) - Firing rate within infarction

Diff_grid = (zeros(N,N)+1)*ip;

store_vec = [];

infarction_units = radius/x_step;

% Assumed square grid
maxN = size(Diff_grid, 1);
Rs = R/(1-R);

source_ind = cap_ind;
num_sources_raw = size(source_ind,1);

num_draw = floor((1-perc_occl)*num_sources_raw);
perm_ind = randperm(num_sources_raw, num_draw);
source_ind_filtered = source_ind(perm_ind,:);

for i = 1:size(source_ind_filtered,1)
    Diff_grid(source_ind_filtered(i,1), source_ind_filtered(i,2)) = phi_C;
end

Psi_T = cons_rate*firing_rate*ratio_neur*sqrt(D);
Psi_I = cons_rate*inf_firing_rate*ratio_neur*sqrt(D);

if (tau == 0)
    tau = (x_step^2)/(D);
end

tau_step = tau/2;

lambda = (tau_step*D)/x_step^2;
alpha = (tau_step*Rs)/2;

if boundary
    Diff_grid(:,1) = phi_V;
    Diff_grid(:,maxN) = phi_V;
    Diff_grid(1,:) = phi_V;
    Diff_grid(maxN,:) = phi_V;
end

prev_grid = Diff_grid;
current_grid = Diff_grid;

A1 = diag((1 + 2*lambda + alpha)*ones(1,maxN)) + diag(-lambda*ones(1,maxN-1),1) + diag(-lambda*ones(1,maxN-1),-1);
A2 = diag((1 + 2*lambda)*ones(1,maxN)) + diag(-lambda*ones(1,maxN-1),1) + diag(-lambda*ones(1,maxN-1),-1);
A1(1,1) = 1 + lambda + alpha;
A1(maxN, maxN) = 1 + lambda + alpha;
A2(1,1) = 1 + lambda + alpha;
A2(maxN, maxN) = 1 + lambda + alpha;

B1 = diag((1 - 2*lambda - alpha)*ones(1,maxN)) + diag(lambda*ones(1,maxN-1),1) + diag(lambda*ones(1,maxN-1),-1);
B2 = diag((1 - 2*lambda)*ones(1,maxN)) + diag(lambda*ones(1,maxN-1),1) + diag(lambda*ones(1,maxN-1),-1);
B3 = zeros(maxN, maxN);
B1(1,1) = 1 - lambda - alpha;
B1(maxN, maxN) = 1 - lambda - alpha;
B2(1, 1) = 1 - lambda - alpha;
B2(maxN, maxN) = 1 - lambda - alpha;

% disp(B2);

d1 = ones(maxN, 1)*(-Psi_T*tau_step + 2*alpha*phi_V);
d2 = ones(maxN, 1)*(-Psi_T*tau_step);

d1_inf = d1;
d2_inf = d2;

d1_inf((center_index(1) - infarction_units):(center_index(1) + infarction_units))...
    = -Psi_I*tau_step + 2*alpha*phi_V;
d2_inf((center_index(1) - infarction_units):(center_index(1) + infarction_units))...
    = -Psi_I*tau_step;

for iter = 1:num_iter
    
%     disp(iter);
    
    % Calculate indices of sub-regions
    sub1 = current_grid >= phi_V;
    sub2 = current_grid < phi_V & current_grid >= Psi_T;
    sub3 = current_grid < Psi_T & current_grid >= 0;
    sub4 = current_grid < 0;
    
    % Initialize transformation matrices
    B_temp = zeros(maxN, maxN);
    A_temp = zeros(maxN, maxN);
    d_temp = zeros(maxN, 1);
    
    % Iterate explicitly in the x direction
    for k = 1:maxN
        B_temp(sub1(:,k), :) = B1(sub1(:,k), :);
        B_temp(sub2(:,k), :) = B2(sub2(:,k), :);
        B_temp(sub3(:,k), :) = B2(sub3(:,k), :);
        if (sum(sub4(:,k)) > 0)
            B_temp(sub4(:,k), :) = B3(sub4(:,k), :);
        end
        
        if (abs(k-center_index(1)) < infarction_units)
            d_temp(sub1(:,k)) = d1_inf(sub1(:,k), :);
            d_temp(sub2(:,k)) = d2_inf(sub2(:,k), :);
        else
            d_temp(sub1(:,k)) = d1(sub1(:,k), :);
            d_temp(sub2(:,k)) = d2(sub2(:,k), :);
        end
        
        current_grid(:,k) = B_temp*prev_grid(:,k) + d_temp;
    end
    
%     disp(sub1(:,k))

    % Transpose data to iterate in y direction
    current_grid = current_grid';
    
    sub1 = current_grid >= phi_V;
    sub2 = current_grid < phi_V & current_grid >= Psi_T;
    sub3 = current_grid < Psi_T & current_grid >= 0;
    sub4 = current_grid < 0;
    
    % Iterate implicitly in the y direction
    for j = 1:maxN        
        A_temp(sub1(:,j), :) = A1(sub1(:,j), :);
        A_temp(sub2(:,j), :) = A2(sub2(:,j), :);
        A_temp(sub3(:,j), :) = A2(sub3(:,j), :);
        A_temp(sub4(:,j), :) = A2(sub4(:,j), :);
        
        current_grid(:,j) = A_temp\current_grid(:,j);
    end
    
    % Note, must use the transposed coordinates
    for i = 1:size(source_ind_filtered,1)
        current_grid(source_ind_filtered(i,2), source_ind_filtered(i,1)) = phi_C + A*sin(f*2*pi*(tau*(iter-1) + tau_step));
    end
    
    if boundary
        current_grid(:,1) = phi_V;
        current_grid(:,maxN) = phi_V;
        current_grid(1,:) = phi_V;
        current_grid(maxN,:) = phi_V;
    end
    
    % Recalculate indices of sub-regions for next half time step
    sub1 = current_grid >= phi_V;
    sub2 = current_grid < phi_V & current_grid >= Psi_T;
    sub3 = current_grid < Psi_T & current_grid >= 0;
    sub4 = current_grid < 0;
    
    % Re-initialize transformation matrices
    B_temp = zeros(maxN, maxN);
    A_temp = zeros(maxN, maxN);
    d_temp = zeros(maxN, 1);
    
    % Iterate explicitly in y direction
    for j = 1:maxN        
        B_temp(sub1(:,j), :) = B1(sub1(:,j), :);
        B_temp(sub2(:,j), :) = B2(sub2(:,j), :);
        B_temp(sub3(:,j), :) = B2(sub3(:,j), :);
        if (sum(sub4(:,j)) > 0)
            B_temp(sub4(:,j), :) = B3(sub4(:,j), :);
        end
        
        if (abs(j-center_index(1)) < infarction_units)
            d_temp(sub1(:,j)) = d1_inf(sub1(:,j), :);
            d_temp(sub2(:,j)) = d2_inf(sub2(:,j), :);
        else
            d_temp(sub1(:,j)) = d1(sub1(:,j), :);
            d_temp(sub2(:,j)) = d2(sub2(:,j), :);
        end
        
        current_grid(:,j) = B_temp*current_grid(:,j) + d_temp;
    end
    
    % Transpose data to iterate in x direction
    current_grid = current_grid';
    
    sub1 = current_grid >= phi_V;
    sub2 = current_grid < phi_V & current_grid >= Psi_T;
    sub3 = current_grid < Psi_T & current_grid >= 0;
    sub4 = current_grid < 0;
    
    % Iterate implicitly in the x direction
    for k = 1:maxN        
        A_temp(sub1(:,k), :) = A1(sub1(:,k), :);
        A_temp(sub2(:,k), :) = A2(sub2(:,k), :);
        A_temp(sub3(:,k), :) = A2(sub3(:,k), :);
        A_temp(sub4(:,k), :) = A2(sub4(:,k), :);
        
        current_grid(:,k) = A_temp\current_grid(:,k);
    end
    
%     disp(current_grid);
    
    for i = 1:size(source_ind_filtered,1)
        current_grid(source_ind_filtered(i,1), source_ind_filtered(i,2)) = phi_C + A*sin(f*2*pi*(tau*(iter)));
    end
    
    if boundary
        current_grid(:,1) = phi_V;
        current_grid(:,maxN) = phi_V;
        current_grid(1,:) = phi_V;
        current_grid(maxN,:) = phi_V;
    end
    
    difference_grid = current_grid - prev_grid;
    max_error = max(max(abs(difference_grid)));
   
    if (max_error < epsilon)
        break    
    end
    
    prev_grid = current_grid;
    
    store_vec(iter) = current_grid(center_index(1), center_index(2));
end

disp(max_error);
grid = current_grid;

end