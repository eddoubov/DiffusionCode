N = 150;
x_step = 20;
sigma = 60;

D = 1000;
tau = .5;
f = 0;
A = 15;
phi_A = 30;
phi_V = phi_A-10;
n_iter = 10000;
cons_rate = .0026;
perc_occl = 0;
boundary = 0;
seed = 100;
ip = 6;
epsilon = 1e-10;

center = [floor(N/2), floor(N/2)];

neuron_radius = 8.7;
neuron_volume = 4/3*pi*neuron_radius^3;

venule_radius = 5;

% Thalamus override
% Create scenario for cortex with diameter 1500;

radius = 750;

% neur_per_mm3 = 200000;
% num_cap_per_mm2 = 385;
% num_cap = num_cap_per_mm2*4;
% firing_rate = 3;

neur_per_mm3 = 35000;
num_cap_per_mm2 = 385;
num_cap = num_cap_per_mm2*(N*x_step/1000)^2;
firing_rate = 4;
% 

total_area = (N*x_step)^2;

ratio_neur = (neuron_volume/1000^3)*neur_per_mm3;
mu = sqrt(4/sqrt(3)*total_area/num_cap);

R = 1 - (pi*venule_radius^2/(N*x_step)^2)*num_cap;

[X, Y, fin_grid1, art_ind, art_ind_occl, ~] = createScenario(N, x_step, mu, sigma, num_cap, D, 0, R, ...
    f, A, phi_A, phi_V, n_iter, cons_rate, firing_rate, ratio_neur, ...
    radius, boundary, seed, center, epsilon, ip, firing_rate);

disp(size(art_ind, 1))
disp(size(art_ind_occl, 1))

figure(1)
clf()
surf(X*x_step,Y*x_step,fin_grid1)
colormap(jet)
shading interp
xlabel("Microns")
ylabel("Microns")
zlabel("PO2 (mmHg)")
colorbar
caxis([0 40])
view(2);

% Create scenario for cortex with diameter 1500;
radius = 1000;

[X, Y, fin_grid2, art_ind2, art_ind_occl2,store_vec] = createScenario(N, x_step, mu, sigma, num_cap, D, tau, R, ...
    f, A, phi_A, phi_V, n_iter, cons_rate, firing_rate, ratio_neur, ...
    radius, boundary, seed, center, epsilon, ip, firing_rate);

disp(size(art_ind2, 1))
disp(size(art_ind_occl2, 1))

figure(2)
clf()
surf(X*x_step,Y*x_step,fin_grid2)
colormap(jet)
shading interp
xlabel("Microns")
ylabel("Microns")
zlabel("PO2 (mmHg)")
colorbar
caxis([0 40])
view(2);

% Scenario for cortex, diameter 1mm with stimulation
radius = 750;
firing_rate = 8;

phi_A = 40;
phi_V = phi_A-10;

[X, Y, fin_grid3, art_ind3, art_ind_occl3,~] = createScenario(N, x_step, mu, sigma, num_cap, D, tau, R, ...
    f, A, phi_A, phi_V, n_iter, cons_rate, firing_rate, ratio_neur, ...
    radius, boundary, seed, center, epsilon, ip, firing_rate);

disp(size(art_ind3, 1))
disp(size(art_ind_occl3, 1))

figure(3)
clf()
surf(X*x_step,Y*x_step,fin_grid3)
colormap(jet)
shading interp
xlabel("Microns")
ylabel("Microns")
zlabel("PO2 (mmHg)")
colorbar
caxis([0 40])
view(2);

% Scenario for cortex, diameter 1500 with stimulation
radius = 1000;

[X, Y, fin_grid4, art_ind4, art_ind_occl4,~] = createScenario(N, x_step, mu, sigma, num_cap, D, tau, R, ...
    f, A, phi_A, phi_V, n_iter, cons_rate, firing_rate, ratio_neur, ...
    radius, boundary, seed, center, epsilon, ip, firing_rate);

disp(size(art_ind4, 1))
disp(size(art_ind_occl4, 1))

figure(4)
clf()
surf(X*x_step, Y*x_step, fin_grid4)
colormap(jet)
shading interp
xlabel("Microns")
ylabel("Microns")
zlabel("PO2 (mmHg)")
colorbar
caxis([0 40])
view(2);

% % Scenario for vestibular nuclei, diameter 1mm
% radius = 500;
% 
% phi_A = 30;
% phi_V = phi_A-10;
% 
% neur_per_mm3 = 80000;
% num_cap_per_mm2 = 583;
% num_cap = num_cap_per_mm2*4;
% firing_rate = 100;
% 
% ratio_neur = (neuron_volume/1000^3)*neur_per_mm3;
% mu = sqrt(4/sqrt(3)*total_area/num_cap);
% 
% R = 1 - (pi*venule_radius^2/(N*x_step)^2)*num_cap;
% 
% [X, Y, fin_grid5, art_ind5, art_ind_occl5, ~] = createScenario(N, x_step, mu, sigma, num_cap, D, tau, R, ...
%     f, A, phi_A, phi_V, n_iter, cons_rate, firing_rate, ratio_neur, ...
%     radius, boundary, seed, center, epsilon);
% 
% disp(size(art_ind5, 1))
% disp(size(art_ind_occl5, 1))
% 
% figure(5)
% clf()
% surf(X*x_step,Y*x_step,fin_grid5)
% colormap(jet)
% shading interp
% xlabel("Microns")
% ylabel("Microns")
% zlabel("PO2 (mmHg)")
% colorbar
% caxis([0 40])
% view(2);