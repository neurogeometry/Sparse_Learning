% This function implements the sparse learning rule based on Eqs. (47) of 
% Supplementary Materials. It produces the results for the models that include 
% h, κ, ℓ1, sign, and gap constraints. 
% The notation used matches the manuscript.

% INPUT PARAMETERS:
% X: binary input associations, number of inputs (N) x number of associations (m)
% y: binary outputs associations, 1 x m
% h: firing threshold, scalar
% kappa: robustness parameter, scalar >=0
% g: signs of input connections, N x 1 array of +1 and -1 
% w: mean absolute connection weight (ℓ1 norm is N*w), scalar >0
% delta: gap or minimum absolute non-zero connection weight, N x 1 array of >=0 numbers

% OUTPUTS PARAMETERS:
% J: input connection strengths, N x 1
% exitflag: 1 if the solution is found, 0 if not

% IMPROTANT INSTRUCTIONS
% use h = 0 or [] to remove threshold
% use kappa = 0 or [] to remove robustness
% use delta = zeros(N,1) or [] to remove gap constraints
% use g = [] to remove sign constraints 

% HOW TO RUN EXAMPLE 
% N = 100;
% m = 20;
% f_in=0.5.*ones(N,1);
% f_out=0.5;
% h=0;
% kappa=10;
% w=1;
% inhibitory_fraction=0.2;
% g=[-ones(round(inhibitory_fraction.*N),1);ones(N-round(inhibitory_fraction.*N),1)];
% delta=2.5.*ones(N,1);
% Nsteps=10^5;
%
% X=rand(N,m)<repmat(f_in,1,m);
% y=rand(1,m)<f_out;
% [J,exitflag] = Sparse_Learning_Rule(X,y,h,kappa,g,w,delta,Nsteps);

function [J,exitflag] = Sparse_Learning_Rule(X,y,h,kappa,g,w,delta,Nsteps)

beta_mu = 0.1*w;
beta_w=0.1;
p_delta=0.98;

tol_m=0;
tol_delta=0;
tol_w=0.001;
plt=0;

N = size(X,1);
m = size(X,2); 

if isempty(h)
    h=0;
end

if isempty(kappa)
    kappa=0;
end

if isempty(g)
    g=nan(N,1);
end

if isempty(delta)
    delta=zeros(N,1);
end

y=2*y-1;
XX=X.*repmat(y,N,1);
if ~isnan(sum(g))
    J = (w+delta).*g;
else
    J=(w+delta).*(2.*rand(N,1)-1);
end

iteration=0;
out=(kappa-J'*XX+h.*y)>0;
bad_associations=nnz(out);

if plt==1
    wind_plt=1000;
    m_errors_plt=zeros(1,Nsteps);
    m_errors_0_plt=zeros(1,Nsteps);
    gap_errors_plt=zeros(1,Nsteps);
    l1_norm_plt=zeros(1,Nsteps);
    figure(100), clf
    subplot(2,2,1), axis square, xlabel('iteration'), ylabel('m error rate'), xlim([0 Nsteps]), ylim([0 1]), hold on
    subplot(2,2,2), axis square, xlabel('iteration'), ylabel('gap error rate'), xlim([0 Nsteps]), ylim([0 1]), hold on
    subplot(2,2,3), axis square, xlabel('iteration'), ylabel('l1 norm'), xlim([0 Nsteps]), ylim([0 2]), hold on
    subplot(2,2,4), axis square, xlabel('J'), ylabel('count'), xlim([-10 10]), hold on
end

while (bad_associations>tol_m || any(J.*g<0) || nnz(J~=0 & abs(J)<delta)>tol_delta || abs(mean(abs(J))/w-1)>tol_w) && iteration<Nsteps
    iteration=iteration+1;
    
    % perceptron + sign step
    ind_mu=find(out);
    if ~isempty(ind_mu)
        ind_mu = ind_mu(randi(length(ind_mu)));
        J = J + beta_mu.*mean(y(ind_mu).*X(:,ind_mu),2);
        J(J.*g<0)=0;
    end

    % l1 + sign step
    ind_W=(J~=0 & abs(J)~=delta);
    if nnz(ind_W)>0
        J(ind_W) = J(ind_W) - beta_w.*sign(J(ind_W)).*(mean(abs(J))-w);
        J(J.*g<0)=0;
    end
    
    % gap step 
    ind_delta=(J~=0 & abs(J)<delta);
    if nnz(ind_delta)>0
        
        rnd=rand(nnz(ind_delta),1);
        J(ind_delta)=delta(ind_delta).*sign(J(ind_delta)).*((rnd>p_delta).*1+(rnd<(1-p_delta)).*(-1));
    end

    temp=J'*XX;
    out=(kappa-temp+h.*y)>0;
    bad_associations=nnz(out);

    if plt==1 && mod(iteration,wind_plt)==0
        m_errors_plt(iteration)=bad_associations/m;
        gap_errors_plt(iteration)=nnz(J~=0 & abs(J)<delta)/N;
        l1_norm_plt(iteration)=mean(abs(J));
        out_0=(-temp+h.*y)>0;
        bad_associations_0=nnz(out_0);
        m_errors_0_plt(iteration)=bad_associations_0/m;

        figure(100)
        subplot(2,2,1), plot(iteration,mean(m_errors_plt(max(1,iteration-wind_plt+1):iteration)),'k.'),
        plot(iteration,mean(m_errors_0_plt(max(1,iteration-wind_plt+1):iteration)),'r.')
        subplot(2,2,2), plot(iteration,mean(gap_errors_plt(max(1,iteration-wind_plt+1):iteration)),'k.')
        subplot(2,2,3), plot(iteration,mean(l1_norm_plt(max(1,iteration-wind_plt+1):iteration)),'k.')
        subplot(2,2,4), hist(J,100); hold(gca,'off')
        drawnow
    end
end

exitflag=(bad_associations<=tol_m & all(J.*g>=0 | isnan(g)) & nnz(J~=0 & abs(J)<delta)<=tol_delta & abs(mean(abs(J))/w-1)<=tol_w);