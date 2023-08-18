%function [val] = solveLPP2(0.8,1,Q_r,Q_c,B,d_mat, P, cost, gamma)  
p= rand ;
q= rand;
B = 15;
pi = [p,  1-p;
      q,    1-q];
d_mat = pi;
a = 0.7;
p1 = optimvar('p1');
e1 = optimvar('e1');
%p2 = optimvar('p2');

obj =p1*Q_r(st,1) + (1-p1)*Q_r(st,2) + 0*e1 ;
prob = optimproblem('Objective',obj,'ObjectiveSense','max');

s0 = 2; % state with constraint


if st==1
    j=2;
else
    j=1;
end

% tmp1(1:2)=0;
% tmp2(1:2)=0;
% 
% for s0=1:2
% tmp1(s0) = d_mat(s0,1)*(cost(s0,1) + gamma*P(s0,1,j)*(d_mat(j,1)*Q_c(j,1) + d_mat(j,2)*Q_c(j,2)) + gamma* P(s0,1,st)*(p1*Q_c(st,1) + p2*Q_c(st,2)));
% tmp2(s0)= d_mat(s0,2)*(cost(s0,2) + gamma*P(s0,2,j)*(d_mat(j,1)*Q_c(j,1) + d_mat(j,2)*Q_c(j,2)) + gamma* P(s0,2,st)*(p1*Q_c(st,1) + p2*Q_c(st,2)));
% end
% 
% tmp_val = tmp1+tmp2;
% constr = alpha(1)*tmp_val(1) +alpha(2)*tmp_val(2);
p2=1-p1;
s0=1;
tmp1 = d_mat(s0,1)*(cost(s0,1) + gamma*P(s0,1,j)*(d_mat(j,1)*Q_c(j,1) + d_mat(j,2)*Q_c(j,2)) + gamma* P(s0,1,st)*(p1*Q_c(st,1) + p2*Q_c(st,2)));
tmp2 = d_mat(s0,2)*(cost(s0,2) + gamma*P(s0,2,j)*(d_mat(j,1)*Q_c(j,1) + d_mat(j,2)*Q_c(j,2)) + gamma* P(s0,2,st)*(p1*Q_c(st,1) + p2*Q_c(st,2)));
constr1 = tmp1 +tmp2;

s0=2;
tmp1 = d_mat(s0,1)*(cost(s0,1) + gamma*P(s0,1,j)*(d_mat(j,1)*Q_c(j,1) + d_mat(j,2)*Q_c(j,2)) + gamma* P(s0,1,st)*(p1*Q_c(st,1) + p2*Q_c(st,2)));
tmp2 = d_mat(s0,2)*(cost(s0,2) + gamma*P(s0,2,j)*(d_mat(j,1)*Q_c(j,1) + d_mat(j,2)*Q_c(j,2)) + gamma* P(s0,2,st)*(p1*Q_c(st,1) + p2*Q_c(st,2)));
constr2 = tmp1 +tmp2;

constr=a*constr1+(1-a)*constr2;

prob.Constraints.c1 = constr - e1  <= B;
prob.Constraints.c2 = p1 +p2 == 1;
prob.Constraints.c3 = p1  >= 0;
prob.Constraints.c4 = p2  >= 0;
prob.Constraints.c5 = e1  >= 0;


problem = prob2struct(prob);


[sol,~] = linprog(problem)