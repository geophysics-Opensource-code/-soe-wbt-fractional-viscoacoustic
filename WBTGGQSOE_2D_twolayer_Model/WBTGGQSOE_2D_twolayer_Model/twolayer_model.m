% This program simulates the propagation of viscoacoustic waves in a 
% 2D twolayer model using the WBT-GGQ SOE algorithm.
% Author: [Feng Wang]
% Date: November 9, 2026

clear
close all


%----------Same wave velocity but different quality factors---------
Q1   = 32*ones(100,200); 
Q2   = 10*ones(100,200); 
Q    = [Q1;Q2];
v1   = 1800*ones(100,200); 
v2   = 1800*ones(100,200); 
   v = [v1 ;v2 ];

%----------Different wave velocities but identical quality factors---------
% Q1   = 32*ones(100,200); 
% Q2   = 32*ones(100,200); 
% Q    = [Q1;Q2];
% v1   = 3200*ones(100,200); 
% v2   = 1800*ones(100,200); 
%    v = [v1 ;v2 ];

%----------Both the wave velocities and the quality factors differ---------
% Q1   = 32*ones(100,200); 
% Q2   = 10*ones(100,200); 
% Q    = [Q1;Q2];
% v1   = 3200*ones(100,200); 
% v2   = 1800*ones(100,200); 
%    v = [v1 ;v2 ];

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
T      = 0.5;
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

dx      = 10;
dz      = 10;
LX      = dx*(Nz-1);
LZ      = dz*(Nx-1);
 
%-------------- PML--------------
 N_PML = 20;

[aaz,bbz,aax,bbx] = cpml(1e-2,[Nz Nx],N_PML,max(v(:)),dt,dx);
sigma_z = -log(bbz)/dt;
sigma_x = -log(bbx)/dt;

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

 sigma(3*Nz/8,Nx/2)  =  sigma(3*Nz/8,Nx/2) + s(k);


sigma_TT = sigma(1:end,:);

for ig = 1:ng

seis(k,ig)= sigma_TT(igz(ig),igx(ig));

end

% pcolor(sigma_TT)
% shading interp;
% colormap("gray")
% axis equal;
% axis tight;
% set(gca, 'YDir', 'reverse');
% drawnow

end

solveSeconds = toc(stepTimer);
fprintf('WBT-GGQ: basis %.3f s, time stepping %.3f s, My=%d\n', ...
 basisSeconds,solveSeconds,My);

z1 = 0:dz:dz*(Nz-1);
x1 = 0:dx:dx*(Nx-1);

figure(100)
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













