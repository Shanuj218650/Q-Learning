%random initial  distribution, 
%decreasing bounnd only if feasible is found
cost= [11 30  ; 24 15 ];
B = 35;

eps=10;
n_epochs=10;
% P is a threee dimension matrix, where P(s,a,s') denotes the probability
% of going to state s' from state s when action a is chosen


P(:,:,1) = [0.2 0.8; 0.3 0.7]; 
P(:,:,2) = [0.8 0.2; 0.7 0.3];

% rewards is a matrix of size (s x a) where s is states and a is actions
rewards = [15 26; 24 18];



% setting initial parameters

gamma = 0.6;
delta = 0.0001;
%alpha = 0.9;

inf_counter=0;
inf_time=0;
% Initialising Qtables
Q_r=rand(2,2);
Q_c=rand(2,2);




Q_r_new=Q_r;
Q_c_new=Q_c;

a_init=0.7;

alpha =  0.5; 
i=0;
counter = 0; 
% Initialize D
 %p= 1;
 %q= 0.9463;

p= rand ;
q= rand;

pi = [p,  1-p;
      q,    1-q];
D_mat = pi;
D_mat_init =pi;
alpha_d = 0.8;

while(1)
    
    i=i+1;
    B_hat = B  +  eps/(i) ;
    
 for st=1:2
   
    % D_mat(st, :) = alpha_d * D_mat(st, :) + (1-alpha_d) * solveLPP2(st, Q_r, Q_c, B , D_mat, P, cost, gamma) ;

    temp= solveLPP2(a_init,st, Q_r, Q_c, B_hat , D_mat, P, cost, gamma) ;
    temp1 = temp(1,2);
    

     if isempty(temp1)
          %fprintf('\n LP is infeasible, randomising')
          inf_counter=inf_counter+1;
            inf_time =i;
          %random actions
         % temp=D_mat_init(st,:);
         temp1(1)=rand;
         temp1(2)=1-temp1(1);
     
         
          
     end
     output_prob=D_mat(st,:);
     tmp2 = (temp1(1,:)-output_prob);
     output_prob=output_prob+0.4*(temp1(1,:)-output_prob);

     D_mat(st, :) =  output_prob;

     d_1=D_mat(1,:);
     d_2= D_mat(2,:);
     for a = 1:2
         tmp = rewards(st,a) + gamma*(P(st,a,1)*(d_1(1)*Q_r(1,1)+d_1(2)*Q_r(1,2)) + P(st,a,2)*(d_2(1)*Q_r(2,1)+d_2(2)*Q_r(2,2)) );
         Q_r_new(st,a) = Q_r(st,a) + alpha*(tmp - Q_r(st,a));

         tmp1 = cost(st,a) + gamma*(P(st,a,1)*(d_1(1)*Q_c(1,1)+d_1(2)*Q_c(1,2)) + P(st,a,2)*(d_2(1)*Q_c(2,1)+d_2(2)*Q_c(2,2)) );
         Q_c_new(st,a) = Q_c(st,a) + alpha*(tmp1 - Q_c(st,a));
     end
     
 
 end
 max_diff_r = max(max(abs(Q_r - Q_r_new)./Q_r_new ));
 max_diff_c = max(max(abs(Q_c - Q_c_new)./ Q_c_new ));

 if mod(i,30)==9
     
     fprintf('\n iteration = %d, (d_1, d_2) = %f %f Q factors reward , B_hat =%f infeasible =%d times, last infeasible at %d \n',i, d_1(1), d_2(1),B_hat,inf_counter,inf_time)
     disp(Q_r_new)
     fprintf('\n Q factors cost\n')
     disp(Q_c_new)
 end
 Q_r = Q_r_new;
 Q_c = Q_c_new;
 if (max_diff_r < delta) && (max_diff_c < delta) && B_hat-B<0.0005
        if counter > 500
            fprintf('Stopping Time : %d \n', i);
            break
        end
        counter = counter +  1;
    else
        counter = 0;        
 end

 
end
disp(Q_r);
disp(Q_c);
fprintf('policy');
disp(D_mat)
fprintf('\n value of states under this policy')
temp_reward = sum((D_mat.*Q_r)')
fprintf('\n cost of states under this policy')
temp_cost = sum((D_mat.*Q_c)')
init_d=[a_init, 1-a_init];
final_rew= temp_reward*init_d';
final_cost= temp_cost*init_d';
fprintf('\n values of reward= %f and cost =%f (B= %f, B_hat= %f) under initial distribution alpha',final_rew, final_cost,  B, B_hat)





% function [val] = solveLPP(st,qr,qc,Bound)  
% 
%     A = [qc(st,1) qc(st,2) ];        
%     b = Bound;
%     lb = [0 0 ];
%     ub = [1 1 ];
%     
%     Aeq = [1 1 ];
%     beq = 1;
%     
%     f = [-qr(st,1) -qr(st,2) ];        
%     options = optimoptions('linprog','Display','none');
%     [x,~] = linprog(f,A,b,Aeq,beq,lb,ub,options);
%     
%     prob1 = x(1);
%     prob2 = x(2);
%     val=[prob1, prob2]; 
% end




function [val] = solveLPP2(a,st,Q_r,Q_c,B,d_mat, P, cost, gamma)  

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
prob.Constraints.c2 = p1  >= 0;
prob.Constraints.c4 = p2  >= 0;
prob.Constraints.c3 = e1  >= 0;
%prob.Constraints.c6 = e1  <= 1;


problem = prob2struct(prob);

options = optimoptions('linprog','Display','none');
problem.options=options;
[sol,~] = linprog(problem);
val = [sol(1), sol(2)];
% val(1) = sol(1);
% val(2) = 1-sol(1);
% val(3) = sol(2);
% newval = [val(1), val(2)]; 
% newval;


end


