% This function uses linear and mixed-integer linear programming to find 
% the most robust numerical solutions for the constrained perceptron learning 
% models described in the manuscript. It produces the results for the following 
% 4 cases based on Eqs. (41-46) of Supplementary Materials.   
% Case 1: h + κ + ℓ1 
% Case 2: h + κ + sign + ℓ1 
% Case 3: h + κ + ℓ0 + ℓ1
% Case 4: h + κ + gap + ℓ1
% The notation used matches the manuscript.

% INPUT PARAMETERS:
% X: binary input associations, number of inputs (N) x number of associations (m)
% y: binary outputs associations, 1 x m
% h_tilde: normalized firing threshold, scalar
% g: signs of input connections, N x 1 array of +1 and -1 
% p: fraction of non-zero-weight connections (l0 norm is N*p), scalar in (0,1] range
% delta_tilde: normalized gap or minimum absolute non-zero connection weight, N x 1 array of >=0 numbers
% case_number: 1, 2, 3, or 4 as defined above 

% OUTPUTS PARAMETERS:
% J_tilde: normalized input connection strengths, N x 1
% max_kappa_tilde: maximal normalized robustness parameter, scalar

% IMPROTANT INSTRUCTIONS
% use h_tilde = 0 or [] to remove threshold 
% use g = [], p = [], delta_tilde = [] if the related constraints are not used 

% HOW TO RUN EXAMPLE FOR CASE 2 
% N = 100;
% m = 60;
% f_in=0.5.*ones(N,1);
% f_out=0.5;
% h_tilde=0;
% inhibitory_fraction=0.2;
% g=[-ones(round(inhibitory_fraction.*N),1);ones(N-round(inhibitory_fraction.*N),1)];
% p=[];
% delta_tilde=[];
% case_number=2;
% 
% X=rand(N,m)<repmat(f_in,1,m);
% y=rand(1,m)<f_out;
% [J_tilde,max_kappa_tilde] = Numerical_Results(X,y,h_tilde,g,p,delta_tilde,case_number);

function [J_tilde,max_kappa_tilde] = Numerical_Results(X,y,h_tilde,g,p,delta_tilde,case_number)

N = size(X,1);
m = size(X,2);

if isempty(h_tilde)
    h_tilde=0;
end

J_tilde=nan(N,1); max_kappa_tilde=nan;  

% Case 1: h + κ + ℓ1 
if case_number==1
    % x=[kappa_tilde; J1_tilde; J2_tilde]
    f = [-1;zeros(N,1);zeros(N,1)];

    A = [ones(m,1)./N^0.5,-((ones(N,1)*(2*y-1)).*X)'./N,((ones(N,1)*(2*y-1)).*X)'./N];
    b = -(2*y'-1).*h_tilde;
    
    Aeq = [0,ones(1,N),ones(1,N)];
    beq = N;
    
    lb=[-inf;zeros(N,1);zeros(N,1)];
    ub=[inf;inf(N,1);inf(N,1)];

    opts = optimoptions('linprog','Algorithm','interior-point','Display','off','MaxIterations',10^4);
    [x,~,exitflag] = linprog(f,A,b,Aeq,beq,lb,ub,opts);
    
    if exitflag==1
        max_kappa_tilde=x(1);
        J_tilde = x(2:N+1)-x(N+2:end);
    end    
end

% Case 2: h + κ + sign + ℓ1 
if case_number==2
    % x=[kappa_tilde; J_tilde]
    f = [-1;zeros(N,1)];
    
    A = [ones(m,1)./N^0.5,-((ones(N,1)*(2*y-1)).*X)'./N];
    b = -(2*y'-1).*h_tilde;

    Aeq = [0,g'];
    beq = N;
    
    lb=zeros(size(g));
    lb(g==-1)=-inf;
    ub=zeros(size(g));
    ub(g==1)=inf;
    lb=[-inf;lb];
    ub=[inf;ub];

    opts = optimoptions('linprog','Algorithm','interior-point','Display','off','MaxIterations',10^4);
    [x,~,exitflag] = linprog(f,A,b,Aeq,beq,lb,ub,opts);
    
    if exitflag==1
        max_kappa_tilde=x(1);
        J_tilde = x(2:end);
    end
end

% Case 3: h + κ + ℓ0 + ℓ1
if case_number==3  
    % x=[kappa_tilde; c; y1; y2]
    f = [-1;zeros(N,1);zeros(N,1);zeros(N,1)];
    intcon=2:N+1;
    
    A = [zeros(N,1),-N.*diag(ones(1,N)),diag(ones(1,N)),-diag(ones(1,N));...
        zeros(N,1),-N.*diag(ones(1,N)),-diag(ones(1,N)),diag(ones(1,N));...
        ones(m,1)./N^0.5,zeros(m,N),-((ones(N,1)*(2*y-1)).*X)'./N,((ones(N,1)*(2*y-1)).*X)'./N];
    b = [zeros(N,1);zeros(N,1);-(2*y'-1).*h_tilde];

    Aeq = [0,ones(1,N),zeros(1,N),zeros(1,N);...
           0,zeros(1,N),ones(1,N),ones(1,N)];
    beq = [N*p;N];
    
    lb=[-inf;zeros(N,1);zeros(N,1);zeros(N,1)];
    ub=[inf;ones(N,1);inf(N,1);inf(N,1)];
    
    x0=[];

    options = optimoptions('intlinprog','Display','iter','MaxTime',600);
    [x,~,exitflag] = intlinprog(f,intcon,A,b,Aeq,beq,lb,ub,x0,options);
    
    if exitflag==1 && nnz(((x(N+2:2*N+1)+x(2*N+2:end))==0) & (x(2:N+1)==1))==0
        max_kappa_tilde=x(1);
        J_tilde=(x(N+2:2*N+1)-x(2*N+2:end)).*x(2:N+1);
    end
end

% Case 4: h + κ + gap + ℓ1
if case_number==4
    % x=[kappa_tilde; c; (s+1)/2; J; w]
    f = [-1;zeros(N,1);zeros(N,1);zeros(N,1);zeros(N,1)];
    intcon=2:2*N+1;
    
    A = [zeros(N,1),zeros(N,N),2*N.*diag(ones(1,N)),-diag(ones(1,N)),diag(ones(1,N));...
        zeros(N,1),zeros(N,N),-2*N.*diag(ones(1,N)),diag(ones(1,N)),diag(ones(1,N));...
        
        zeros(N,1),zeros(N,N),zeros(N,N),diag(ones(1,N)),-diag(ones(1,N));...
        zeros(N,1),zeros(N,N),zeros(N,N),-diag(ones(1,N)),-diag(ones(1,N));...
        
        zeros(N,1),-N.*diag(ones(1,N)),zeros(N,N),zeros(N,N),diag(ones(1,N));...
        zeros(N,1),diag(delta_tilde),zeros(N,N),zeros(N,N),-diag(ones(1,N));...
        
        ones(m,1)./N^0.5,zeros(m,N),zeros(m,N),-((ones(N,1)*(2*y-1)).*X)'./N,zeros(m,N)];
    
    b = [2*N.*ones(N,1);zeros(N,1);zeros(N,1);zeros(N,1);zeros(N,1);zeros(N,1);-(2*y'-1).*h_tilde];

    Aeq = [0,zeros(1,N),zeros(1,N),zeros(1,N),ones(1,N)];
    beq = N;
    
    lb=[-inf;zeros(N,1);zeros(N,1);-inf(N,1);zeros(N,1)];
    ub=[inf;ones(N,1);ones(N,1);inf(N,1);inf(N,1)];
    
    x0=[];

    options = optimoptions('intlinprog','Display','iter','MaxTime',600);
    [x,~,exitflag] = intlinprog(f,intcon,A,b,Aeq,beq,lb,ub,x0,options);
    
    if exitflag==1
        max_kappa_tilde=x(1);
        J_tilde=x(2*N+2:3*N+1);
    end
end
