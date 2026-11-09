function [sigma_post, phi_post] = update_stress_response_2D_WBTGGQ(phi_pre, SOE, wt, dv1, dv3, M, N, dt, C_gamma, rou, F1, F2, sigma_x, sigma_z)
%UPDATE_STRESS_RESPONSE_2D_WBTGGQ WBT-GGQ 历史递推和应力更新。

My = size(phi_pre,1);
if dt ~= SOE.dt || size(phi_pre,2) ~= N || size(phi_pre,3) ~= M || ...
        size(SOE.decay,1) ~= My
    error('WBTGGQ:StateSize', '历史状态尺寸或时间步长与 WBTGGQ_basis 的预计算结果不一致。');
end

phi_post = zeros(My,N,M);
D = 1-dt*(sigma_x+sigma_z);
div_v = dv1+dv3;

history_integral = SOE.local.*div_v;
for i = 1:My
    
    old = reshape(phi_pre(i,:,:),N,M);
    E = reshape(SOE.decay(i,:,:),N,M);
    B = reshape(SOE.step(i,:,:),N,M);
    weight = reshape(wt(i,:,:),N,M);
    history_integral = history_integral + weight.*E.*old;
    phi_post(i,:,:) = D.*(E.*old+B.*div_v);
end


sigma_post = (C_gamma.*rou).*D.^2.*history_integral ...
    + D.*dt.*(F1+F2);
end
