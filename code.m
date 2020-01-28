%Univariate ivx_wald test for presistence of prediction, based on Kostakis, Magdalinos and Stamatogiannis (2015).
function [A_ivx,W_ivx]= ivx_wald_univ_1(X,Y,varargin)
default_c = -1;
default_beta = 0.95;
default_alpha = 0;
default_dMn = 0.0001;
default_bwmethod = 1;
%default_window_var = 60;
p = inputParser;
   %validScalarPosNum = @(x) isnumeric(x) && isscalar(x);
   addRequired(p,'X');
   addRequired(p,'Y');
   addParameter(p,'c',default_c); % the optional input: c, for equation (4)
   addParameter(p,'beta',default_beta); % the optional input: beta, for equation (4)
   addParameter(p,'alpha',default_alpha); % the optional input: alpha, for P(iv)
   addParameter(p,'dMn',default_dMn); % the critical value for bandwidth choice according to Newey-West
   addParameter(p,'bwmethod',default_bwmethod); % the method to calculate NW bandwidth, default is 1 (rule of thumb)
   %addParameter(p,'window_var',Default_window_var); %setup the window size for variance estimation
   parse(p,X,Y,varargin{:});
c = p.Results.c;
alpha = p.Results.alpha;
beta = p.Results.beta;
dMn = p.Results.dMn;
bwmethod = p.Results.bwmethod;
%% 0. Setup the parameters
n = length(X);
r = size(X,2);

C = c*ones(r,1);

%% the restrictions of H and h, we simple test
% H_0: A_ivx = 0
h_r = zeros(r,1); 
H = ones(r,1);

if r > 1
   warning('X must be Univariate')
end

%% 1. Prepare the IVX variable z
%% 1.1 Calculate AR(1) of x, and record the corresponding residuals
EstMdl = ar(X,1);
R_n = -1*EstMdl.a(2);

u = nan(n,1);
u(1) = 0;
for t = 2: n
    u(t) = X(t) - R_n*X(t-1);
end

%% 1.2 Calculate delta_x
delta_x = nan(n,1);
for t = 2:n
    delta_x(t) = u(t) + (C/(n^alpha))*X(t-1);
end

%% 1.3 Calculate equation (4) and (5)
R_z = 1 + (C/(n^beta));
z = nan(n,1);
z(1) = 0;
for t = 2:n
    z(t) = R_z*z(t-1) + delta_x(t);
end

%% 2. Estimate A_ivx and Wald test
%% 2.1 A_ivx
%% demean X and Y
y_ = Y-mean(Y);
x_ = X-mean(X);
A_ivx = (y_'*z)*(x_'*z)^(-1);

e_ivx = nan(n,1);
for t = 1: n
    e_ivx(t) = y_(t) - A_ivx*x_(t);
end


%% 2.2 W_ivx
% 2.2.1 Equation (13), (14), (15)
%
%% OLS of equation (1) to obtain the residuals
beta = (x_'*x_)^(-1)*x_'*y_;
e_ols = nan(n,1);
for t = 1: n
    e_ols(t) = Y(t) - beta*X(t);
end

Sigma_ee = 0;
Sigma_eu = 0;
Sigma_uu = 0;
for t = 1:n
    Sigma_ee = e_ols(t)^2+Sigma_ee;
    Sigma_eu = e_ols(t)*u(t)+Sigma_eu;
    Sigma_uu = u(t)^2+Sigma_uu;
end
Sigma_ee = (1/n)*Sigma_ee;
Sigma_eu = (1/n)*Sigma_eu;
Sigma_uu = (1/n)*Sigma_uu;

% Selecting bandwith according to Newey-West-type estimators
if bwmethod == 0;
    numEstimates = 100;
    CoeffNW = zeros(numEstimates,1);
    MSENW = zeros(numEstimates,1);
    for bw = 1:numEstimates
        [~,~,Coeff] = hac(x_,y_,'bandwidth',bw,'display','off','intercept',false); % Newey-West
        CoeffNW(bw) = Coeff;
        MSENW(bw) = mean((y_- CoeffNW(bw)*x_).^2);
        if bw == 1
            dMSE = MSENW(bw)-0;
        else
            dMSE = MSENW(bw)-MSENW(bw-1);
        end
        if dMSE <=dMn
            Mn=bw;
            break;
        end
    end
elseif bwmethod == 1
    Mn = floor(0.75*n^(1/3)-1); % rule of thumb
end


Lambda_uu = 0;
Lambda_ue = 0;
first_part = 0;
Lambda_uu = u(t)*u(t) + Lambda_uu;
Lambda_ue = u(t)*e_ols(t) + Lambda_ue;
for h = 1: Mn
    first_part = (1- h/(Mn+1))+ first_part;
    for t = Mn+1:n
        Lambda_uu = first_part*u(t)*u(t-h) + Lambda_uu;
        Lambda_ue = first_part*u(t)*e_ols(t-h) + Lambda_ue;
    end
end

Lambda_uu = (1/n)*Lambda_uu;
Lambda_ue = (1/n)*Lambda_ue;
Omega_uu = Sigma_uu + Lambda_uu + Lambda_uu';
Omega_eu = Sigma_eu + Lambda_ue';

%  Equation (20) & (21)
pho = Omega_eu/sqrt(Sigma_ee*Omega_uu);
First =0;
for t = 1:n-1
    First = z(t)^2+First;
end
MM = (First-n*mean(z(1:t-1))^2*(1-pho^2))*Sigma_ee;
Q = H*((z'*x_)^(-1))*MM*((x_'*z)^(-1))*H';
W_ivx = (H*A_ivx-h_r)'*Q^(-1)*(H*A_ivx-h_r);


end
