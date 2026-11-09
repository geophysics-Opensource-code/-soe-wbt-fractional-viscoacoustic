function [Y, WT, My, SOE] = WBTGGQ_basis(Q, idx_Q, cgamma, reps, dt, T)

% WBTGGQ_BASIS Generates the WBT-GGQ sum-of-exponentials basis based on partitioned Q values.
validateattributes(Q, {'numeric'}, {'2d','real','finite','nonempty','positive'}, mfilename, 'Q');
validateattributes(idx_Q, {'numeric'}, {'vector','real','finite','nonempty','positive'}, mfilename, 'idx_Q');
validateattributes(cgamma, {'numeric'}, {'vector','real','finite','nonempty'}, mfilename, 'cgamma');
validateattributes(reps, {'numeric'}, {'scalar','real','finite','>',0,'<',1}, mfilename, 'reps');
validateattributes(dt, {'numeric'}, {'scalar','real','finite','positive'}, mfilename, 'dt');
validateattributes(T, {'numeric'}, {'scalar','real','finite','>=',dt}, mfilename, 'T');

idx_Q = idx_Q(:);
alpha = 2*cgamma(:);
if numel(idx_Q) ~= numel(alpha) || any(alpha <= 0 | alpha >= 1)
    error('WBTGGQ:FractionalOrder', ...
        'cgamma 必须与 idx_Q 一一对应，且实际导数阶数 2*cgamma 必须在 (0,1) 内。');
end
if numel(unique(idx_Q)) ~= numel(idx_Q) || ...
        ~isequal(sort(double(idx_Q)), unique(double(Q(:))))
    error('WBTGGQ:QPartition', 'idx_Q 必须恰好包含 Q 中的全部不同品质因子，且不能重复。');
end

nQ = numel(idx_Q);
rates = cell(nQ,1);
weights = cell(nQ,1);
counts = zeros(nQ,1);
for iq = 1:nQ
   
    [si, wi, nexp] = sumofexpappr_wbtgg(alpha(iq), reps, dt, T);
    si = si(:);
    wi = wi(:);
    if ~isscalar(nexp) || nexp < 1 || nexp ~= fix(nexp) || ...
            numel(si) ~= nexp || numel(wi) ~= nexp || ...
            ~isreal(si) || ~isreal(wi) || any(~isfinite(si)) || ...
            any(~isfinite(wi)) || any(si <= 0) || any(wi <= 0)
        error('WBTGGQ:InvalidBasis', 'Q=%g 对应的 SOE 节点/权重必须是有限正实数。', idx_Q(iq));
    end
    rates{iq} = si;
    weights{iq} = wi;
    counts(iq) = nexp;
end

[Nz, Nx] = size(Q);
My = max(counts);
Y = zeros(My,Nz,Nx);
WT = zeros(My,Nz,Nx);
SOE.decay = zeros(My,Nz,Nx);
SOE.step = zeros(My,Nz,Nx);
SOE.local = zeros(Nz,Nx);

for iq = 1:nQ
    IDX = find(Q == idx_Q(iq));
    si = rates{iq};
    wi = weights{iq};
    decay = exp(-si*dt);
    step = -expm1(-si*dt)./si;
    SOE.local(IDX) = dt^(1-alpha(iq))/(1-alpha(iq));
    for kk = 1:counts(iq)
     
        Y(kk,IDX) = si(kk);
        WT(kk,IDX) = wi(kk);
        SOE.decay(kk,IDX) = decay(kk);
        SOE.step(kk,IDX) = step(kk);
    end
end
SOE.dt = dt;
SOE.T = T;
SOE.reps = reps;
SOE.Q = idx_Q;
SOE.alpha = alpha;
SOE.counts = counts;
SOE.rates = rates;
SOE.raw_weights = weights;
end
