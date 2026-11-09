function [si,wi,nexp] = sumofexpappr_wbtgg(beta,reps,tau,T)
%SUMOFEXPAPPR_WBTGG 与 sumofexpappr2new 相同
%   [si,wi,nexp] = sumofexpappr_wbtgg(beta,reps,tau,T)
%
%   beta : 要逼近的幂核正指数，
%   reps : 核相对误差容差，5e-13 <= reps < 1。
%   tau  : 最小时间滞后，tau > 0。
%   T    : 最大时间滞后，T >= tau。
%   si   : 正衰减率列向量，物理单位为时间的倒数。
%   wi   : 正权重列向量，未额外乘除 Caputo Gamma 归一化因子。
%   nexp : 指数项数，等于 length(si) 和 length(wi)。
%
narginchk(4,4);
opts = struct(); % 算法内部采用默认设置。
validateattributes(beta,{'numeric'},{'real','scalar','finite','positive'});
validateattributes(reps,{'numeric'},{'real','scalar','finite','positive','<',1});
validateattributes(tau,{'numeric'},{'real','scalar','finite','positive'});
validateattributes(T,{'numeric'},{'real','scalar','finite','>=',tau});
if reps < 5e-13
    error('soe:Precision','Double precision version requires reps >= 5e-13.');
end
delta = tau/T;
if delta == 0 || ~isfinite(beta*log(T))
    error('soe:Scale','The requested time scaling is not representable.');
end
if delta == 1
    s = 1; w = exp(1);
else
    [s,w] = soe_initial_quad(beta,delta,reps*0.04,opts);
    [s,w] = soe_wbt_reduce(beta,delta,s,w,reps*0.65,opts);
    [s,w] = soe_positive_refine(beta,delta,s,w,reps*0.85,opts);
end
ncheck = 10001;
[err,~] = soe_check_error(beta,delta,s,w,ncheck);
if err > reps
    error('soe:Accuracy','SOE validation failed: %.4g > requested %.4g.',err,reps);
end
% Restore physical units only once. Column outputs match the original API.
si = s(:)/T;
wi = exp(log(w(:))-beta*log(T));
if any(~isfinite(si) | si<=0) || any(~isfinite(wi) | wi<=0)
    error('soe:Scale','Positive physical SOE coefficients overflow/underflow.');
end
% Check the returned coefficients, including rounding from unit conversion.
[errphysical,~] = soe_check_error(beta,delta,si*T, ...
    exp(log(wi)+beta*log(T)),ncheck);
if errphysical > reps
    error('soe:Accuracy','Unit conversion failed the requested tolerance.');
end
nexp = numel(wi);
end

% -------------------------------------------------------------------------
function [s,w,stats] = soe_initial_quad(beta,delta,tol,opts)
% Positive adaptive Gauss-Legendre quadrature of Eq. (3.24):
% t^-beta = 1/Gamma(1+beta) * integral_0^infinity exp(-t*u^(1/beta)) du.
% The omitted tail has exact relative size Q(beta,t*S), largest at delta.
order = getoption(opts,'InitialOrder',12);
validateattributes(order,{'numeric'},{'scalar','integer','>=',4,'<=',64});
zmax = max(1,beta);
while gammainc(zmax,beta,'upper') > tol/4
    zmax = 1.5*zmax + 1;
end
S = zmax/delta;
if ~isfinite(S) || beta*log(S)>log(realmax)-5
    error('soe:Range','Initial quadrature exceeds double precision range.');
end
% Geometric panels in rate s, mapped to u=s^beta; integrate all the way
% to u=0 so a small beta never requires truncation at exp(-C/beta).
logs = unique([log(0.01):log(2):log(S),log(S)]);
edges = [0,exp(beta*logs)];
edges = unique(edges);
tfit = exp(linspace(log(delta),0,max(201,ceil(30*log10(1/delta)+1))))';
[x1,q1] = gauss_rule(order);
[x2,q2] = gauss_rule(2*order);
s = []; w = []; splits = 0;
paneltol = tol/(4*(numel(edges)-1));
for k=1:numel(edges)-1
    [sk,wk,nk] = panel(edges(k),edges(k+1),paneltol,0);
    s = [s;sk]; w = [w;wk]; %#ok<AGROW>
    splits = splits+nk;
end
% Tiny-beta underflows are near-static modes; keep a positive representable
% rate. Equal positive rates can be combined without approximation error.
[s,~,ix] = unique(s);
w = accumarray(ix,w);
[err,~] = soe_check_error(beta,delta,s,w,2001);
if err > max(tol,100*eps)
    error('soe:InitialAccuracy','Initial quadrature error %.3e exceeds %.3e.',err,tol);
end
stats = struct('nexp',numel(s),'max_rel_error',err,'tail_rel_bound', ...
    gammainc(zmax,beta,'upper'),'rate_cutoff',S,'splits',splits, ...
    'representation','Guo et al. (2024), Eq. (3.24)');

    function [sp,wp,nsplit] = panel(a,b,budget,depth)
        u1 = (a+b)/2+(b-a)*x1/2;
        u2 = (a+b)/2+(b-a)*x2/2;
        s1 = exp(max(log(realmin),log(u1)/beta));
        s2 = exp(max(log(realmin),log(u2)/beta));
        w1 = q1*((b-a)/2)*exp(-gammaln(1+beta));
        w2 = q2*((b-a)/2)*exp(-gammaln(1+beta));
        f1 = exp(beta*log(tfit)-tfit*s1')*w1;
        f2 = exp(beta*log(tfit)-tfit*s2')*w2;
        discrepancy = max(abs(f2-f1));
        if discrepancy <= max(budget,20*eps*max(abs(f2)))
            sp=s2; wp=w2; nsplit=0;
        elseif depth>=40 || (a+b)/2==a || (a+b)/2==b
            error('soe:Quadrature','Adaptive quadrature could not resolve a panel.');
        else
            [sl,wl,nl] = panel(a,(a+b)/2,budget/2,depth+1);
            [sr,wr,nr] = panel((a+b)/2,b,budget/2,depth+1);
            sp=[sl;sr]; wp=[wl;wr]; nsplit=nl+nr+1;
        end
    end
end

function [x,w] = gauss_rule(n)
k=(1:n-1)'; a=k./sqrt(4*k.^2-1);
[V,D]=eig(diag(a,1)+diag(a,-1));
[x,p]=sort(diag(D)); w=2*(V(1,p)').^2;
end
function v=getoption(o,n,d)
if isfield(o,n), v=o.(n); else, v=d; end
end

% -------------------------------------------------------------------------
function [s,w,stats] = soe_wbt_reduce(beta,delta,s0,w0,tol,opts)
%SOE_WBT_REDUCE Positive square-root weighted balanced truncation.
% [s,w,stats] = soe_wbt_reduce(beta,delta,s0,w0,tol,opts) compresses
% sum_j w0(j)*exp(-s0(j)*t) for t^(-beta), delta <= t <= 1.
% The input and output rates and weights are real, positive column vectors.
% uses a quadrature square-root factor, instead of forming and factoring an ill-conditioned Gramian.
% No Control System or Optimization Toolbox is required.
%
% Options (all optional):
%   ShiftFraction          .9    shift = ShiftFraction*delta, in [0,1]
%   WeightPower            .5    rho(u) = (u+delta)^(-WeightPower)
%   WBTQuadratureOrder      16    positive Gauss rule on geometric panels
%   WBTCheckPoints        2000    independent error-check grid size
%   WBTOrderStep             4   increase in tested reduction order
%   MaxWBTOrder           Inf    optional limit on reduction order

if nargin < 6 || isempty(opts), opts = struct(); end
s0 = s0(:); w0 = w0(:);
if numel(s0) ~= numel(w0) || isempty(s0) || ...
        ~isreal(s0) || ~isreal(w0) || ...
        any(~isfinite(s0)) || any(~isfinite(w0)) || ...
        any(s0 <= 0) || any(w0 <= 0)
    error('soe_wbt_reduce:InvalidSOE','Initial rates and weights must be finite and positive.');
end
if ~(isscalar(beta) && isfinite(beta) && beta > 0 && ...
        isscalar(delta) && isfinite(delta) && delta > 0 && delta < 1 && ...
        isscalar(tol) && isfinite(tol) && tol > 0)
    error('soe_wbt_reduce:InvalidInput','Require beta>0, 0<delta<1 and tol>0.');
end
shiftFraction = wbt_option(opts,'ShiftFraction',.9);
weightPower = wbt_option(opts,'WeightPower',.5);
quadOrder = wbt_option(opts,'WBTQuadratureOrder',16);
ncheck = wbt_option(opts,'WBTCheckPoints',2000);
orderStep = wbt_option(opts,'WBTOrderStep',4);
maxOrder = wbt_option(opts,'MaxWBTOrder',Inf);
if ~(isscalar(shiftFraction) && isfinite(shiftFraction) && ...
        shiftFraction >= 0 && shiftFraction <= 1 && ...
        isscalar(weightPower) && isfinite(weightPower) && ...
        isscalar(quadOrder) && isfinite(quadOrder) && quadOrder >= 4 && quadOrder == fix(quadOrder) && ...
        isscalar(ncheck) && isfinite(ncheck) && ncheck >= 10 && ncheck == fix(ncheck) && ...
        isscalar(orderStep) && isfinite(orderStep) && orderStep >= 1 && orderStep == fix(orderStep) && ...
        isscalar(maxOrder) && maxOrder >= 1 && (isinf(maxOrder) || maxOrder == fix(maxOrder)))
    error('soe_wbt_reduce:InvalidOptions','Invalid WBT option value.');
end
[s0,ix] = sort(s0); w0 = w0(ix);
s = s0; w = w0;
initialError = soe_check_error(beta,delta,s,w,ncheck);
shift = shiftFraction*delta;
H = (1-shift)/2;
stats = struct('initialTerms',numel(s0),'wbtTerms',numel(s0), ...
    'accepted',false,'initialError',initialError,'sampledError',initialError, ...
    'shift',shift,'horizon',H,'weightPower',weightPower, ...
    'quadraturePoints',0,'numericalRank',0,'singularValues',[], ...
    'testedOrders',[],'testedErrors',[],'status','initial_kept');
if numel(s0) < 2, return; end

% Scale the shifted input vector before constructing the feature matrix.
% A common scalar factor changes singular values but not the projection.
logb = (log(w0)-shift*s0)/2;
logScale = max(logb);
b = exp(logb-logScale);
[u,q] = geometric_gauss(H,max(s0),quadOrder);
logRow = .5*log(q)-weightPower*log(u+delta);
logRow = logRow-max(logRow);
F = bsxfun(@times,exp(-u*s0.'),b.');
F = bsxfun(@times,F,exp(logRow));
[~,D,U] = svd(F,'econ');
d = diag(D);
% Keep the directions resolved in the square-root factor.  Unlike an
% eigendecomposition of F'*F, this does not square its condition number.
nr = sum(d > max(size(F))*eps*d(1));
stats.quadraturePoints = numel(u);
stats.numericalRank = nr;
stats.singularValues = d/d(1); % normalized square-root Gramian singular values
lastOrder = min([numel(s0)-1,nr,maxOrder]);
if lastOrder < 1, return; end

% Each order is tested against the actual power kernel.  Coarse growth is
% followed by testing every omitted order in the final bracket.  This is
% an economical search, not a claim of globally minimal SOE length.
orders = unique([1:orderStep:lastOrder,lastOrder]);
previousOrder = 0;
for r = orders
    [sr,wr,er] = candidate(r);
    stats.testedOrders(end+1,1) = r;
    stats.testedErrors(end+1,1) = er;
    if er <= tol
        for rr = previousOrder+1:r-1
            [st,wt,et] = candidate(rr);
            stats.testedOrders(end+1,1) = rr;
            stats.testedErrors(end+1,1) = et;
            if et <= tol
                sr = st; wr = wt; er = et;
                break;
            end
        end
        s = sr; w = wr;
        stats.wbtTerms = numel(s);
        stats.sampledError = er;
        stats.accepted = true;
        stats.status = 'reduced';
        return;
    end
    previousOrder = r;
end

    function [sr,wr,er] = candidate(r)
        Ur = U(:,1:r);
        % Orthogonal projection preserves symmetry and positive rates.
        Sr = Ur.'*bsxfun(@times,s0,Ur);
        Sr = (Sr+Sr.')/2;
        [V,L] = eig(Sr);
        sr = real(diag(L));
        amp = V.'*(Ur.'*b);
        wr = [];
        er = Inf;
        if any(~isfinite(sr)) || any(sr <= 0) || ~isreal(amp)
            return;
        end
        % Undo the kernel shift in logarithmic form.  A candidate with an
        % overflowing unshifted representation is rejected, not clipped.
        logw = 2*logScale+2*log(abs(amp))+shift*sr;
        if any(logw > log(realmax)) || any(isnan(logw))
            return;
        end
        wr = exp(logw);
        keep = wr > 0 & isfinite(wr);
        sr = sr(keep); wr = wr(keep);
        if isempty(sr), return; end
        [sr,jj] = sort(sr); wr = wr(jj);
        er = soe_check_error(beta,delta,sr,wr,ncheck);
        if ~isscalar(er) || ~isfinite(er), er = Inf; end
    end
end

function value = wbt_option(opts,name,default)
if isfield(opts,name) && ~isempty(opts.(name))
    value = opts.(name);
else
    value = default;
end
end

function [u,q] = geometric_gauss(H,maxRate,n)
% Resolve the narrowest exponential near zero, then grow panels by two.
% The Jacobi-matrix rule is Gauss-Legendre with strictly positive weights.
k = (1:n-1).';
off = k./sqrt(4*k.^2-1);
[V,D] = eig(diag(off,1)+diag(off,-1));
[x,jj] = sort(diag(D));
weights = 2*(V(1,jj).').^2;
edge = min(H,1/maxRate);
edges = [0;edge];
while edges(end) < H
    edges(end+1,1) = min(H,2*edges(end)); %#ok<AGROW>
end
nPanels = numel(edges)-1;
u = zeros(nPanels*n,1); q = u;
for j = 1:nPanels
    a = edges(j); h = (edges(j+1)-a)/2;
    ids = (j-1)*n+(1:n);
    u(ids) = a+h*(x+1);
    q(ids) = h*weights;
end
end

% -------------------------------------------------------------------------
function [s,w,stats] = soe_positive_refine(beta,delta,s,w,tol,opts)
%SOE_POSITIVE_REFINE Positive SOE refinement and GGQ-style node deletion.
%   [S,W,STATS] = SOE_POSITIVE_REFINE(BETA,DELTA,S,W,TOL,OPTS) improves
%   sum_j W(j)*exp(-S(j)*t) as an approximation of t^(-BETA), DELTA<=t<=1.
%   S and W are UNSHIFTED, strictly positive, real column vectors
%   Options (all optional):
%     NumFit         default max(400,80*ceil(log10(1/DELTA)))
%     MaxIter        default 100, per nonlinear solve
%     MaxDeleteTrials default 6, candidates tried at each SOE order;
%                     use Inf to try every candidate
%     Optimize       default true; false only checks/returns the input
%     NumCheck       default 2001, independent error-check grid size
%
%   STATS records term counts, validation errors, nonlinear iterations,
%   deletion attempts, accepted deletions, and final training error.

if nargin < 6 || isempty(opts), opts = struct(); end
if ~isstruct(opts), error('soe_positive_refine:Options','opts must be a struct.'); end
if ~isscalar(beta) || ~isreal(beta) || ~isfinite(beta) || beta <= 0
    error('soe_positive_refine:Beta','beta must be finite and positive.');
end
if ~isscalar(delta) || ~isreal(delta) || ~isfinite(delta) || delta <= 0 || delta > 1
    error('soe_positive_refine:Delta','delta must lie in (0,1].');
end
if ~isscalar(tol) || ~isreal(tol) || ~isfinite(tol) || tol <= 0
    error('soe_positive_refine:Tolerance','tol must be finite and positive.');
end
s = s(:); w = w(:);
if isempty(s) || numel(s) ~= numel(w) || ~isreal(s) || ~isreal(w) || ...
        any(~isfinite(s)) || any(~isfinite(w)) || any(s <= 0) || any(w <= 0)
    error('soe_positive_refine:Seed','s and w must be finite positive real vectors of equal length.');
end
[s,ix] = sort(s); w = w(ix);
numFit = refine_option(opts,'NumFit',max(400,80*ceil(-log10(delta))));
maxIter = refine_option(opts,'MaxIter',100);
maxTrials = refine_option(opts,'MaxDeleteTrials',6);
numCheck = refine_option(opts,'NumCheck',2001);
optimize = refine_option(opts,'Optimize',true);
check_integer(numFit,2,'NumFit',false);
check_integer(maxIter,0,'MaxIter',false);
check_integer(maxTrials,0,'MaxDeleteTrials',true);
check_integer(numCheck,3,'NumCheck',false);
if ~isscalar(optimize) || ~(islogical(optimize) || isnumeric(optimize)) || ...
        ~isreal(optimize) || ~isfinite(optimize) || ~ismember(optimize,[0 1])
    error('soe_positive_refine:Options','Optimize must be true or false.');
end

t = unique([delta; exp(linspace(log(delta),0,numFit)).'; 1]);
[err,tWorst] = soe_check_error(beta,delta,s,w,numCheck);
stats = struct('initial_terms',numel(s),'final_terms',numel(s), ...
    'initial_error',err,'final_error',err,'t_worst',tWorst, ...
    'attempts',0,'accepted_deletions',0,'lm_iterations',0, ...
    'training_error',NaN,'stop_reason','','history',[numel(s),err]);

if ~optimize
    stats.training_error = max(abs(relative_system(log([w;s]),beta,t)));
    stats.stop_reason = 'Optimization disabled.';
    return
end

% Keep the seed unless refinement improves its independent validation error.
[sNew,wNew,it] = fit_positive(s,w,beta,t,tol,maxIter);
stats.lm_iterations = stats.lm_iterations + it;
[newErr,newWorst] = soe_check_error(beta,delta,sNew,wNew,numCheck);
if isfinite(newErr) && newErr < err
    s = sNew; w = wNew; err = newErr; tWorst = newWorst;
end
stats.history(end+1,:) = [numel(s),err];

% A seed that still misses tolerance is not safe for greedy order reduction.
if err > tol || ~isfinite(err)
    stats.stop_reason = 'Refinement did not reach the requested tolerance; no deletion accepted.';
elseif maxTrials == 0
    stats.stop_reason = 'Node deletion disabled by MaxDeleteTrials=0.';
else
    while numel(s) > 1
        order = rank_candidates(s,w,beta,t,maxTrials);
        accepted = false;
        for kk = 1:min(numel(order),maxTrials)
            k = order(kk);
            keep = true(numel(s),1); keep(k) = false;
            stats.attempts = stats.attempts + 1;
            fitT = t;
            [sTry,wTry,it] = fit_positive(s(keep),w(keep),beta,fitT,tol,maxIter);
            stats.lm_iterations = stats.lm_iterations + it;
            [tryErr,tryWorst] = soe_check_error(beta,delta,sTry,wTry,numCheck);

            % A narrow missed peak can be supplied to a second fit. This is
            % deliberately bounded so failed deletions have bounded cost.
            trainErr = max(abs(relative_system(log([wTry;sTry]),beta,fitT)));
            if tryErr > tol && isfinite(tryErr) && trainErr < 0.8*tol && ...
                    isfinite(tryWorst) && tryWorst >= delta && tryWorst <= 1
                fitT = unique([fitT;tryWorst]);
                [sTry,wTry,it] = fit_positive(sTry,wTry,beta,fitT,tol,maxIter);
                stats.lm_iterations = stats.lm_iterations + it;
                [tryErr,tryWorst] = soe_check_error(beta,delta,sTry,wTry,numCheck);
            end
            if isfinite(tryErr) && tryErr <= tol
                s = sTry; w = wTry; err = tryErr; tWorst = tryWorst;
                t = fitT;
                stats.accepted_deletions = stats.accepted_deletions + 1;
                stats.history(end+1,:) = [numel(s),err];
                accepted = true;
                break
            end
        end
        if ~accepted
            stats.stop_reason = 'No tested single-node deletion passed independent validation.';
            break
        end
    end
    if isscalar(s), stats.stop_reason = 'One positive exponential remains.'; end
end

[s,ix] = sort(s); w = w(ix);
stats.final_terms = numel(s);
stats.final_error = err;
stats.t_worst = tWorst;
stats.training_error = max(abs(relative_system(log([w;s]),beta,t)));
end

function order = rank_candidates(s,w,beta,t,maxTrials)
% Rank by a regularized linearized compensation after deleting both columns
n = numel(s);
[r,J,F] = relative_system(log([w;s]),beta,t);
columnScale = max(sqrt(sum(J.^2,1)),sqrt(eps));
Js = bsxfun(@rdivide,J,columnScale);
importance = max(F,[],1).';
[~,coarseOrder] = sort(importance);
if isinf(maxTrials)
    pool = coarseOrder;
else
    pool = coarseOrder(1:min(n,max(32,2*maxTrials)));
end
score = zeros(numel(pool),1);
for jj = 1:numel(pool)
    k = pool(jj);
    keep = true(1,2*n); keep([k,n+k]) = false;
    Jr = Js(:,keep);
    rhs = F(:,k)-r;
    mu = 1e-4;
    d = [Jr;mu*eye(2*n-2)] \ [rhs;zeros(2*n-2,1)];
    score(jj) = norm(Jr*d-rhs)/sqrt(numel(t)) + ...
        mu*norm(d)/sqrt(max(1,numel(d)));
end
[~,ix] = sort(score);
order = pool(ix);
end

function [s,w,iterations] = fit_positive(s,w,beta,t,tol,maxIter)
n = numel(s);
x = log([w;s]);
lambda = 1e-3;
iterations = 0;
for it = 1:maxIter
    [r,J] = relative_system(x,beta,t);
    cost = sum(r.^2)/2;
    if ~isfinite(cost) || max(abs(r)) <= max(0.1*tol,20*eps), break; end
    scale = max(sqrt(sum(J.^2,1)),sqrt(eps));
    Js = bsxfun(@rdivide,J,scale);
    accepted = false;
    for retry = 1:8
        % Augmentation uses a QR least-squares solve, never J'*J.
        dz = [Js;sqrt(lambda)*eye(2*n)] \ [-r;zeros(2*n,1)];
        dx = dz./scale.';
        % A log parameter step is a multiplicative change. Limit runaway
        % near-null directions and let subsequent iterations travel further.
        dx = dx/max(1,max(abs(dx))/2);
        xTry = x+dx;
        if any(~isfinite(xTry)) || any(xTry > log(realmax)-4) || ...
                any(xTry < log(realmin)+4)
            lambda = min(1e16,lambda*10);
            continue
        end
        rTry = relative_system(xTry,beta,t);
        costTry = sum(rTry.^2)/2;
        if isfinite(costTry) && costTry < cost
            x = xTry;
            lambda = max(1e-14,lambda/3);
            accepted = true;
            break
        end
        lambda = min(1e16,lambda*10);
    end
    iterations = it;
    if ~accepted, break; end
    if norm(dx,inf) <= 1e-11 || cost-costTry <= 1e-12*max(cost,realmin)
        break
    end
end
w = exp(x(1:n)); s = exp(x(n+1:end));
[s,ix] = sort(s); w = w(ix);
end

function [r,J,F] = relative_system(x,beta,t)
n = numel(x)/2;
p = x(1:n); q = x(n+1:end);
st = t*exp(q).';
% Evaluate relative contributions directly; this avoids forming a large
% t^(-beta) target or unnecessarily large unscaled exponential weights.
F = exp(bsxfun(@plus,beta*log(t),p.')-st);
r = sum(F,2)-1;
if nargout > 1, J = [F,-F.*st]; end
end

function value = refine_option(opts,name,fallback)
if isfield(opts,name) && ~isempty(opts.(name)), value = opts.(name);
else, value = fallback;
end
end

function check_integer(value,minimum,name,allowInf)
ok = isnumeric(value) && isreal(value) && isscalar(value) && ...
    value >= minimum && ((isfinite(value) && value == floor(value)) || ...
    (allowInf && isinf(value) && value > 0));
if ~ok, error('soe_positive_refine:Options','%s has an invalid value.',name); end
end

% -------------------------------------------------------------------------
function [err,tWorst,details] = soe_check_error(beta,delta,s,w,ncheck)
% Numerical relative L-infinity check on [delta,1]. No rigorous certificate.
% Uses independent log and linear grids and roots of d(relative error)/dlog(t).
if nargin<5, ncheck=4001; end
validateattributes(ncheck,{'numeric'},{'scalar','integer','>=',3});
s=s(:); w=w(:);
if numel(s)~=numel(w) || isempty(s) || any(~isfinite(s)|s<=0) || any(~isfinite(w)|w<=0)
    err=Inf; tWorst=NaN; details=struct(); return
end
z=unique([linspace(log(delta),0,ncheck), ...
    log(linspace(delta,1,max(3,ceil(ncheck/4))))])';
[r,d]=evaluate(z);
[err,k]=max(abs(r)); tWorst=exp(z(k));
crossings=find(d(1:end-1).*d(2:end)<0 & ...
    max(abs(d(1:end-1)),abs(d(2:end)))>1e-13);
rootsFound=0;
for k=crossings'
    try
        zstar=fzero(@derivative,[z(k),z(k+1)],optimset('TolX',1e-12,'Display','off'));
        rstar=evaluate(zstar); rootsFound=rootsFound+1;
        if abs(rstar)>err, err=abs(rstar); tWorst=exp(zstar); end
    catch
        % Grid value remains valid evidence; a failed root search is recorded.
    end
end
details=struct('grid_points',numel(z),'stationary_brackets',numel(crossings), ...
    'stationary_points',rootsFound,'certified',false);
    function [res,der] = evaluate(zz)
        tt=exp(zz(:)); res=zeros(size(tt)); der=zeros(size(tt));
        for start=1:512:numel(tt)
            ids=start:min(start+511,numel(tt)); ts=tt(ids)*s';
            C=exp(beta*zz(ids)+log(w')-ts);
            res(ids)=sum(C,2)-1;
            if nargout>1, der(ids)=sum(C.*(beta-ts),2); end
        end
    end
    function value=derivative(zz)
        [~,value]=evaluate(zz);
    end
end
