% This function generates replica theory results for the constrained
% perceptron learning models described in the manuscript. It produces the results 
% for the following 4 cases based on Eqs. (35-40) of Supplementary Materials.   
% Case 1: h + κ + ℓ1 
% Case 2: h + κ + sign + ℓ1 
% Case 3: h + κ + ℓ0 + ℓ1
% Case 4: h + κ + gap + ℓ1
% The notation used matches the manuscript.

% INPUT PARAMETERS:
% f_in: input firing probabilities, N x 1 array of numbers in (0,1) range
% f_out: output firing probability, scalar in (0,1) range
% h_tilde: normalized firing threshold, scalar
% kappa_tilde: normalized robustness parameter, scalar >=0
% g: signs of input connections, N x 1 array of +1 and -1 
% p: fraction of non-zero-weight connections (l0 norm is N*p), scalar in (0,1] range
% delta_tilde: normalized gap or minimum absolute non-zero connection weight, N x 1 array of >=0 numbers
% case_number: 1, 2, 3, or 4 as defined above 

% OUTPUTS PARAMETERS:
% alpha: critical memory storage capacity, scalar
% S: sparsity or fraction of zero-weight connections, scalar
% exitflag: 1 if the solution is found, 0 if not

% IMPROTANT INSTRUCTIONS
% use h_tilde = 0 or [], kappa_tilde = 0 or [] to remove threshold and/or robustness 
% use g = [], p = [], delta_tilde = [] if the related constraints are not used 

% HoW To RUN EXAMPLE FOR CASE 2 
% N=100;
% f_in=0.5.*ones(N,1);
% f_out=0.5;
% h_tilde=0;
% kappa_tilde=0.1;
% inhibitory_fraction=0.2;
% g=[-ones(round(inhibitory_fraction.*N),1);ones(N-round(inhibitory_fraction.*N),1)];
% p=[];
% delta_tilde=[];
% case_number=2;
% [alpha,S,exitflag]=Replica_Theory_Results(f_in,f_out,h_tilde,kappa_tilde,g,p,delta_tilde,case_number)

function [alpha,S,exitflag]=Replica_Theory_Results(f_in,f_out,h_tilde,kappa_tilde,g,p,delta_tilde,case_number)

alpha=nan; S=nan; exitflag=0;

E = @(x) (1+erf(x))/2;
F = @(x) exp(-x.^2)./pi^0.5+x.*(1+erf(x));
D = @(x) x.*F(x)+E(x);

if isempty(h_tilde)
    h_tilde=0;
end
    
if isempty(kappa_tilde)
    kappa_tilde=0;
end

ff=1./(f_in.*(1-f_in)).^0.5;
fff=f_in.*ff;
if ~isempty(g)
    fg=fff.*g;
end

options = optimset('Display','off','MaxIter',10^5,'MaxFunEvals',10^5,'TolX',10^-8,'TolFun',10^-8);

% Cases 1: h + κ + ℓ1 
if case_number==1
    if kappa_tilde>0
        % x=[u+,u-,z,eta]
        x=[0.1,0.1,0,0];

        Eqs = @(x) [f_out*F(x(2))-(1-f_out)*F(x(1));...
            mean(ff.*(F(fff.*x(3)-ff.*x(4)./2)+F(-fff.*x(3)-ff.*x(4)./2)))-...
            (2*h_tilde*x(3)+x(4))*(x(1)+x(2))/kappa_tilde^2*(f_out*E(x(2))+(1-f_out)*E(x(1)))/(f_out*F(x(2))+(1-f_out)*F(x(1)));...
            mean(fff.*(F(fff.*x(3)-ff.*x(4)./2)-F(-fff.*x(3)-ff.*x(4)./2)))+...
            h_tilde*(2*h_tilde*x(3)+x(4))*(x(1)+x(2))/kappa_tilde^2*(f_out*E(x(2))+(1-f_out)*E(x(1)))/(f_out*F(x(2))+(1-f_out)*F(x(1)));...
            mean(D(fff.*x(3)-ff.*x(4)./2)+D(-fff.*x(3)-ff.*x(4)./2))-...
            (2*h_tilde*x(3)+x(4))^2/kappa_tilde^2*(f_out*E(x(2))+(1-f_out)*E(x(1)))^2/(f_out*F(x(2))+(1-f_out)*F(x(1)))^2];

        [x,~,exitfl] = fsolve(Eqs, x, options);
        if exitfl==1 && (x(1)+x(2))>=0
            alpha = (f_out*D(x(2))+(1-f_out)*D(x(1)))/(f_out*F(x(2))+(1-f_out)*F(x(1)))^2*(2*h_tilde*x(3)+x(4))^2/kappa_tilde^2;
            S = 1-mean(E(fff.*x(3)-ff.*x(4)./2)+E(-fff.*x(3)-ff.*x(4)./2));
            exitflag=1;
        end

    elseif kappa_tilde==0
        % x=[u+,z]
        x=zeros(1,2);
        
        Eqs = @(x) [f_out*F(-x(1))-(1-f_out)*F(x(1));...
            mean((fff+ff.*h_tilde).*F((fff+ff.*h_tilde).*x(2))+(-fff+ff.*h_tilde).*F((-fff+ff.*h_tilde).*x(2)))];
        
        [x,~,exitfl] = fsolve(Eqs, x, options);
        if exitfl==1
            S = 1-mean(E((fff+ff.*h_tilde).*x(2))+E((-fff+ff.*h_tilde).*x(2)));
            alpha = (1-S)/(f_out*E(-x(1))+(1-f_out)*E(x(1)));
            exitflag=1;
        end
    end
end

% Case 2: h + κ + sign + ℓ1 
if case_number==2
    if kappa_tilde>0
        % x=[u+,u-,z,eta]
        x=[0.1,0.1,0,0];
        
        Eqs = @(x) [f_out*F(x(2))-(1-f_out)*F(x(1));...
            mean(ff.*(F(-fg.*x(3)-ff./2.*x(4))))-...
            (2*h_tilde*x(3)+x(4))*(x(1)+x(2))/kappa_tilde^2*(f_out*E(x(2))+(1-f_out)*E(x(1)))/(f_out*F(x(2))+(1-f_out)*F(x(1)));...
            mean(fg.*(F(-fg.*x(3)-ff./2.*x(4))))-...
            h_tilde*(2*h_tilde*x(3)+x(4))*(x(1)+x(2))/kappa_tilde^2*(f_out*E(x(2))+(1-f_out)*E(x(1)))/(f_out*F(x(2))+(1-f_out)*F(x(1)));...
            mean(D(-fg.*x(3)-ff./2.*x(4)))-...
            (2*h_tilde*x(3)+x(4))^2/kappa_tilde^2*(f_out*E(x(2))+(1-f_out)*E(x(1)))^2/(f_out*F(x(2))+(1-f_out)*F(x(1)))^2];
        
        [x,~,exitfl] = fsolve(Eqs, x, options);
        if exitfl==1 && (x(1)+x(2))>=0
            alpha = (f_out*D(x(2))+(1-f_out)*D(x(1)))/(f_out*F(x(2))+(1-f_out)*F(x(1)))^2*(2*h_tilde*x(3)+x(4))^2/kappa_tilde^2;
            S = 1-mean(E(-fg.*x(3)-ff./2.*x(4)));
            exitflag=1;
        end
        
    elseif kappa_tilde==0
        % x=[u+,z]
        x=zeros(1,2);
        
        Eqs = @(x) [f_out*F(-x(1))-(1-f_out)*F(x(1));...
            mean((-fg+ff.*h_tilde).*F((-fg+ff.*h_tilde).*x(2)))];
        
        [x,~,exitfl] = fsolve(Eqs, x, options);
        if exitfl==1
            S = 1-mean(E((-fg+ff.*h_tilde).*x(2)));
            alpha = (1-S)/(f_out*E(-x(1))+(1-f_out)*E(x(1)));
            exitflag=1;
        end
    end
end

% Case 3: h + κ + ℓ0 + ℓ1
if case_number==3
    if kappa_tilde>0
        % x=[u+,u-,x,z,eta]
        x=[0.1,0.1,0,0,0];
        
        Eqs = @(x) [f_out*F(x(2))-(1-f_out)*F(x(1));...
            mean(E(fff.*x(4)-ff./2.*x(5)-x(3))+E(-fff.*x(4)-ff./2.*x(5)-x(3)))-p;...
            mean(ff.*(F(fff.*x(4)-ff./2.*x(5)-x(3))+F(-fff.*x(4)-ff./2.*x(5)-x(3))+2.*x(3).*(E(fff.*x(4)-ff./2.*x(5)-x(3))+E(-fff.*x(4)-ff./2.*x(5)-x(3)))))-...
            (2*h_tilde*x(4)+x(5))*(x(1)+x(2))/kappa_tilde^2*(f_out*E(x(2))+(1-f_out)*E(x(1)))/(f_out*F(x(2))+(1-f_out)*F(x(1)));...
            mean(fff.*(-F(fff.*x(4)-ff./2.*x(5)-x(3))+F(-fff.*x(4)-ff./2.*x(5)-x(3))+2.*x(3).*(-E(fff.*x(4)-ff./2.*x(5)-x(3))+E(-fff.*x(4)-ff./2.*x(5)-x(3)))))-...
            h_tilde*(2*h_tilde*x(4)+x(5))*(x(1)+x(2))/kappa_tilde^2*(f_out*E(x(2))+(1-f_out)*E(x(1)))/(f_out*F(x(2))+(1-f_out)*F(x(1)));...
            mean(D(fff.*x(4)-ff./2.*x(5)-x(3))+D(-fff.*x(4)-ff./2.*x(5)-x(3))+2.*x(3).*(F(fff.*x(4)-ff./2.*x(5)-x(3))+F(-fff.*x(4)-ff./2.*x(5)-x(3)))+2.*x(3)^2.*(E(fff.*x(4)-ff./2.*x(5)-x(3))+E(-fff.*x(4)-ff./2.*x(5)-x(3))))-...
            (2*h_tilde*x(4)+x(5)).^2./kappa_tilde^2*(f_out*E(x(2))+(1-f_out)*E(x(1)))^2/(f_out*F(x(2))+(1-f_out)*F(x(1)))^2];
        
        [x,~,exitfl] = fsolve(Eqs, x, options);
        if exitfl==1 && (x(1)+x(2))>=0 && x(3)>=0
            alpha = (f_out*D(x(2))+(1-f_out)*D(x(1)))/(f_out*F(x(2))+(1-f_out)*F(x(1)))^2*(2*h_tilde*x(4)+x(5))^2/kappa_tilde^2;
            S = 1-p;
            exitflag=1;
        end
        
    elseif kappa_tilde==0
        % x=[u+,x,z]
        x=zeros(1,3);
        
        Eqs = @(x) [f_out*F(-x(1))-(1-f_out)*F(x(1));...
            mean(E((fff+ff.*h_tilde).*x(3)-x(2))+E((-fff+ff.*h_tilde).*x(3)-x(2)))-p;...
            mean((fff+ff.*h_tilde).*F((fff+ff.*h_tilde).*x(3)-x(2))+(-fff+ff.*h_tilde).*F((-fff+ff.*h_tilde).*x(3)-x(2))+2.*x(2).*((fff+ff.*h_tilde).*E((fff+ff.*h_tilde).*x(3)-x(2))+(-fff+ff.*h_tilde).*E((-fff+ff.*h_tilde).*x(3)-x(2))))];
        
        [x,~,exitfl] = fsolve(Eqs, x, options);
        if exitfl==1
            alpha = 1/(f_out*E(-x(1))+(1-f_out)*E(x(1)))*mean(D((fff+ff.*h_tilde).*x(3)-x(2))+D((-fff+ff.*h_tilde).*x(3)-x(2))+2.*x(2).*(F((fff+ff.*h_tilde).*x(3)-x(2))+F((-fff+ff.*h_tilde).*x(3)-x(2)))+2*x(2)^2*p);
            S = 1-p;
            exitflag=1;
        end
    end
end

% Case 4: h + κ + gap + ℓ1
if case_number==4
    if kappa_tilde>0
        % x=[u+,u-,z,eta,Q]
        x=[0.1,0.1,0,0,1];
        
        Eqs = @(x) [f_out*F(x(2))-(1-f_out)*F(x(1));...
            mean(ff.*(F(-fff.*x(3)-ff./2.*x(4)-delta_tilde./ff.*x(5))+F(fff.*x(3)-ff./2.*x(4)-delta_tilde./ff.*x(5))+2*delta_tilde./ff.*x(5).*(E(-fff.*x(3)-ff./2.*x(4)-delta_tilde./ff.*x(5)./2)+E(fff.*x(3)-ff./2.*x(4)-delta_tilde./ff.*x(5)./2))))-2*x(5);...
            mean(fff.*(F(-fff.*x(3)-ff./2.*x(4)-delta_tilde./ff.*x(5))-F(fff.*x(3)-ff./2.*x(4)-delta_tilde./ff.*x(5))+2*delta_tilde./ff.*x(5).*(E(-fff.*x(3)-ff./2.*x(4)-delta_tilde./ff.*x(5)./2)-E(fff.*x(3)-ff./2.*x(4)-delta_tilde./ff.*x(5)./2))))-2*h_tilde*x(5);...
            mean(D(-fff.*x(3)-ff./2.*x(4)-delta_tilde./ff.*x(5))+D(fff.*x(3)-ff./2.*x(4)-delta_tilde./ff.*x(5))+2*delta_tilde./ff.*x(5).*(F(-fff.*x(3)-ff./2.*x(4)-delta_tilde./ff.*x(5))+F(fff.*x(3)-ff./2.*x(4)-delta_tilde./ff.*x(5)))+2*(delta_tilde./ff.*x(5)).^2.*(E(-fff.*x(3)-ff./2.*x(4)-delta_tilde./ff.*x(5)./2)+E(fff.*x(3)-ff./2.*x(4)-delta_tilde./ff.*x(5)./2)))-...
            (2*kappa_tilde*x(5)/(x(1)+x(2)))^2;...
            mean(delta_tilde./ff.*x(5).*(F(-fff.*x(3)-ff./2.*x(4)-delta_tilde./ff.*x(5))+F(fff.*x(3)-ff./2.*x(4)-delta_tilde./ff.*x(5))-F(-fff.*x(3)-ff./2.*x(4)-delta_tilde./ff.*x(5)./2)-F(fff.*x(3)-ff./2.*x(4)-delta_tilde./ff.*x(5)./2))+(delta_tilde./ff.*x(5)).^2.*(E(-fff.*x(3)-ff./2.*x(4)-delta_tilde./ff.*x(5)./2)+E(fff.*x(3)-ff./2.*x(4)-delta_tilde./ff.*x(5)./2)))-...
            (2*h_tilde*x(3)+x(4)-2*kappa_tilde^2*x(5)/(x(1)+x(2))*(f_out*F(x(2))+(1-f_out)*F(x(1)))/(f_out*E(x(2))+(1-f_out)*E(x(1))))*x(5)];
        
        [x,~,exitfl] = fsolve(Eqs, x, options);
        if exitfl==1 && (x(1)+x(2))>=0
            alpha = (f_out*D(x(2))+(1-f_out)*D(x(1)))/(f_out*E(x(2))+(1-f_out)*E(x(1)))^2*(2*kappa_tilde*x(5)/(x(1)+x(2)))^2;
            S = 1-mean(E(-fff.*x(3)-ff./2.*x(4)-delta_tilde./ff.*x(5)./2)+E(fff.*x(3)-ff./2.*x(4)-delta_tilde./ff.*x(5)./2));
            exitflag=1;
        end
        
    elseif kappa_tilde==0
        % x=[u+,z,eta,Q]
        x=[0,0,0,1];
        
        Eqs = @(x) [f_out*F(-x(1))-(1-f_out)*F(x(1));...
            mean(ff.*(F(-fff.*x(2)-ff./2.*x(3)-delta_tilde./ff.*x(4))+F(fff.*x(2)-ff./2.*x(3)-delta_tilde./ff.*x(4))+2.*(delta_tilde./ff.*x(4)).*(E(-fff.*x(2)-ff./2.*x(3)-delta_tilde./ff.*x(4)./2)+E(fff.*x(2)-ff./2.*x(3)-delta_tilde./ff.*x(4)./2))))-2*x(4);...
            mean(fff.*(F(-fff.*x(2)-ff./2.*x(3)-delta_tilde./ff.*x(4))-F(fff.*x(2)-ff./2.*x(3)-delta_tilde./ff.*x(4))+2.*(delta_tilde./ff.*x(4)).*(E(-fff.*x(2)-ff./2.*x(3)-delta_tilde./ff.*x(4)./2)-E(fff.*x(2)-ff./2.*x(3)-delta_tilde./ff.*x(4)./2))))-2*h_tilde*x(4);...
            mean(delta_tilde./ff.*x(4).*(F(-fff.*x(2)-ff./2.*x(3)-delta_tilde./ff.*x(4))+F(fff.*x(2)-ff./2.*x(3)-delta_tilde./ff.*x(4)))-(delta_tilde./ff.*x(4)).*(F(-fff.*x(2)-ff./2.*x(3)-delta_tilde./ff.*x(4)./2)+F(fff.*x(2)-ff./2.*x(3)-delta_tilde./ff.*x(4)./2))+(delta_tilde./ff.*x(4)).^2.*(E(-fff.*x(2)-ff./2.*x(3)-delta_tilde./ff.*x(4)./2)+E(fff.*x(2)-ff./2.*x(3)-delta_tilde./ff.*x(4)./2)))-...
            (2*h_tilde*x(2)+x(3))*x(4)];
        
        [x,~,exitfl] = fsolve(Eqs, x, options);
        if exitfl==1
            alpha = (f_out*D(-x(1))+(1-f_out)*D(x(1)))/(f_out*E(-x(1))+(1-f_out)*E(x(1)))^2*mean(D(-fff.*x(2)-ff./2.*x(3)-delta_tilde./ff.*x(4))+D(fff.*x(2)-ff./2.*x(3)-delta_tilde./ff.*x(4))+2.*delta_tilde./ff.*x(4).*(F(-fff.*x(2)-ff./2.*x(3)-delta_tilde./ff.*x(4))+F(fff.*x(2)-ff./2.*x(3)-delta_tilde./ff.*x(4)))+2.*(delta_tilde./ff.*x(4)).^2.*(E(-fff.*x(2)-ff./2.*x(3)-delta_tilde./ff.*x(4)./2)+E(fff.*x(2)-ff./2.*x(3)-delta_tilde./ff.*x(4)./2)));
            S = 1-mean(E(-fff.*x(2)-ff./2.*x(3)-delta_tilde./ff.*x(4)./2)+E(fff.*x(2)-ff./2.*x(3)-delta_tilde./ff.*x(4)./2));
            exitflag=1;
        end
    end
end

