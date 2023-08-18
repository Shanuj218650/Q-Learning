function [val] = solveLPP2(a,st,Q_r,Q_c,B,d_mat, P, cost, gamma)  
a =0.7;
st =2;
Q_r =[41.8466   51.0166
   50.5458   43.3414];
Q_c =[37.8804   54.6743
   50.5183   40.0769];
B = 80;
d_mat =[1 0
    0 1];
P(:,:,1) = [0.2 0.8; 0.3 0.7]; 
P(:,:,2) = [0.8 0.2; 0.7 0.3];
cost= [11 30  ; 24 15 ];
gamma = 0.6;
p1 = optimvar('p1');
e1 = optimvar('e1');
%p2 = optimvar('p2');

obj =p1*Q_r(st,1) + (1-p1)*Q_r(st,2) + 0*e1  ;
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

prob.Constraints.c1 = constr - e1   <= B;
prob.Constraints.c2 = p1 +p2 == 1;
prob.Constraints.c3 = p1  >= 0;
prob.Constraints.c4 = p2  >= 0;
prob.Constraints.c5 = e1  >= 0;
prob.Constraints.c6 = e1  <= 1;


problem = prob2struct(prob);

options = optimoptions('linprog','Display','none');
problem.options=options;
[sol,~] = linprog(problem);

val(1) = sol(1);
val(2) = 1-sol(1);
val(3) = sol(2);
newval = [val(1), val(2)]; 
newval;


end