% This program simulates the propagation of viscoacoustic waves in a 2D gas chimney model using the WBT-GGQ SOE algorithm.
% Author: [Feng Wang]
% Date: November 9, 2026
clear
close all

Q   = load ('gas_chimney_Q.txt'); 
v   = load ('gas_chimney_vel.txt'); 
v = v(41:end,:);
Q = Q(41:end,:);

[Nz,Nx]  = size(v);

[idx_Q, ia, ib] = unique(Q);

[idx_v, ia1, ib1] = unique(v);

cgamma = zeros(length(idx_Q),1);

for ii = 1:length(idx_Q)

cgamma(ii,:) = (1/pi)*atan(1/idx_Q(ii));
end

%---------Riker wavelet------------
fp     = 15;
fp_ref = 100;
dt     = 1e-3;
T      = 1.5;
Nstep  = round(T/dt); 
assert(abs(Nstep*dt-T)<1e-10,'T must be an integer multiple of dt.');

s  = wavelet(fp,dt,T);

W0 = 2*pi*fp_ref;

C  = zeros(Nz,Nx);

for ii = 1:length(idx_Q)

IDX = find(Q == idx_Q(ii) );

C(IDX) = v(IDX).^2.*(cos(pi*cgamma(ii)./2)).^2*W0.^(-2*cgamma(ii));

clear IDX

end

C_gamma = zeros(Nz,Nx);

for ii = 1:length(idx_Q)

IDX = find(Q == idx_Q(ii) );


C_gamma(IDX) = C(IDX)./gamma(1-2*cgamma(ii));

clear IDX

end


reps = 1e-4;

basisTimer = tic;
[Y,WT,My,SOE] = WBTGGQ_basis(Q,idx_Q,cgamma,reps,dt,T);
basisSeconds = toc(basisTimer);

v1    =  zeros( Nz,Nx );
v3    =  zeros( Nz,Nx ); 
phi   =  zeros( My,Nz,Nx);
sigma =  zeros( Nz,Nx);
rou   = ones(Nz,Nx);

%------------meshing---------

dx      = 12.5;
dz      = 12.5;
LX      = dx*(Nz-1);
LZ      = dz*(Nx-1);
 
%-------------- PML--------------
 N_PML = 20;

[aaz,bbz,aax,bbx] = cpml(1e-2,[Nz Nx],N_PML,max(v(:)),dt,dx);
sigma_z = -log(bbz)/dt;
sigma_x = -log(bbx)/dt;


sigma_z(1:N_PML,:) = 0;

%-----------Set receiver position-----------------

sx  = (Nx-1)/2*dx;
sz  = 0;
gx  = (0:2:(Nx-1))*dx;

HZ  = dx*20;

gz  = HZ*ones(size(gx));
ng  = numel(gx);
g   = 1:ng;


seis = zeros(Nstep ,numel(gx));%Synthetic seismic data

nbc = 0;

[isx,isz,igx,igz] = adjust_sr(sx,sz,gx,gz,dx,nbc);

igz = ones(1,length(igx));

%--------------Finite difference-----------
nn   = 6;
dxd  =  FDcoeffDx(nn);
ddz0 =  dxd';
ddz1 =  [dxd 0]';
ddx0 =  ddz0';
ddx1 =  ddz1';

pic_num = 1;   


%Dm=FDMx_fft2d(p,M,N,LX,LZ,f,axis_class)


stepTimer = tic;

for k = 1:Nstep 

%------------------------Stress excitation source loading------------------------ 

F1 = 0;
F2 = 0;

%----------------velocity--------------

  dsigmax = imfilter(sigma,ddx0)./dx;
  dsigmaz = imfilter(sigma,ddz0)./dx;


  v1 = update_velocity_PML(v1, dsigmax, dt,rou,sigma_x);
  v3 = update_velocity_PML(v3, dsigmaz, dt,rou,sigma_z);

 %----------------stress--------------


 dv1 = imfilter(v1,ddx1)./dx;
 dv3 =  imfilter(v3,ddz1)./dx;


[sigma, phi] = update_stress_response_2D_WBTGGQ( phi, SOE,WT, dv1,dv3,Nx,Nz, dt, C_gamma,rou,F1,F2,sigma_x,sigma_z);

 sigma(2,199)  =  sigma(2,199) + s(k);


sigma_TT = sigma(1:end,:);

for ig = 1:ng

seis(k,ig)= sigma_TT(igz(ig),igx(ig));

end


end

solveSeconds = toc(stepTimer);
fprintf('WBT-GGQ: basis %.3f s, time stepping %.3f s, My=%d\n', ...
 basisSeconds,solveSeconds,My);

z1 = 0:dz:dz*(Nz-1);
x1 = 0:dx:dx*(Nx-1);

figure(1)
pcolor(x1./1000, z1./1000, sigma_TT)
shading interp;
colormap("gray")
axis equal;
axis tight;
set(gca, 'Box', 'on', ...                                 
         'LineWidth', 1, 'GridLineStyle', '-',...          
         'Layer','top',...                                  
         'XGrid', 'off', 'YGrid', 'off', ...               
         'TickDir', 'in', 'TickLength', [.01 .01], ...    
         'XMinorTick', 'off', 'YMinorTick', 'off',...       
         'XColor', [.1 .1 .1],  'YColor', [.1 .1 .1])     
xlabel('Distance (km)'); 
ylabel('Depth (km)'); 
set(gca, 'YDir', 'reverse');
box on
set(gca, 'FontName', 'Times New Roman', 'FontSize', 14, 'TickDir', 'in');

%% Figure 2: Seismic shot gather (symmetric amplitude clipping)

x1 = 0:2*dx:dx*(Nx-1);
T1 = 0:dt:dt*(Nstep-1);

% 1. Robust clipping threshold based on absolute amplitudes
amp = abs(seis(:));
amp = sort(amp(isfinite(amp)));
clipIdx = max(1, min(numel(amp), round(0.99 * numel(amp))));
clipAmp = amp(clipIdx);

% Avoid a degenerate color range
if isempty(amp) || clipAmp <= 0
    error('seis does not contain valid non-zero amplitudes.');
end

figure(2);
clf;

% 2. Plot the seismic record
pcolor(x1./1000, T1, seis);
shading interp;
colormap(gray(256));
clim([-clipAmp, clipAmp]);
% 3. Axes and frame
set(gca, ...
    'Box', 'on', ...
    'LineWidth', 1.0, ...
    'Layer', 'top', ...
    'XGrid', 'off', ...
    'YGrid', 'off', ...
    'TickDir', 'in', ...
    'TickLength', [0.01 0.01], ...
    'XMinorTick', 'off', ...
    'YMinorTick', 'off', ...
    'XColor', [0.1 0.1 0.1], ...
    'YColor', [0.1 0.1 0.1], ...
    'FontName', 'Times New Roman', ...
    'FontSize', 16);

set(gca, 'YDir', 'reverse');

xlabel('Distance (km)', 'FontName', 'Times New Roman', ...
    'FontSize', 16);
ylabel('Times (s)', 'FontName', 'Times New Roman', ...
    'FontSize', 16);

axis tight;
box on;









