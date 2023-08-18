%cost matrix C of size (s x a) for each state and action pair
%C= [11 30  ; 24 15 ];
cost= [11 30  ; 24 15 ];
B = 50;

eps=0.4;
n_epochs=100000;
% P is a threee dimension matrix, where P(s,a,s') denotes the probability
% of going to state s' from state s when action a is chosen


P(:,:,1) = [0.2 0.8; 0.3 0.7]; 
P(:,:,2) = [0.8 0.2; 0.7 0.3];

% rewards is a matrix of size (s x a) where s is states and a is actions
rewards = [15 26; 24 18];



% setting initial parameters

gamma = 0.6;
delta = 0.00001;
%alpha = 0.9;

% Initialising Qtables
Q_r = [1.0 1.0  ; 1.0 1.0];
Q_r_new = [1.0 1.0  ; 1.0 1.0];
Q_c = [1.0 1.0  ; 1.0 1.0  ];
Q_c_new = [1.0 1.0  ; 1.0 1.0];


alpha =  0.5; 
i=0;
counter = 0; 
% Initialize D

D_mat = [0 1 ; 1 0];
alpha_d = 0.6;
while(1)
    
    i=i+1;
 for st=1:2
     alpha_d = alpha_d * 0.88;
     outprob = solveLPP2(st, Q_r, Q_c, B , D_mat, P, cost, gamma);
     if length(outprob) ~= 2
         disp ('In Feas');
         outprob = rand(1, 2);
     end
     D_mat(st, :) = alpha_d * D_mat(st, :) + (1-alpha_d) *  outprob;
     d_1 = D_mat (1, :);
     d_2 = D_mat (2, :);
    
     for a = 1:2
         tmp = rewards(st,a) + gamma*(P(st,a,1)*(d_1(1)*Q_r(1,1)+d_1(2)*Q_r(1,2)) + P(st,a,2)*(d_2(1)*Q_r(2,1)+d_2(2)*Q_r(2,2)) );
         Q_r_new(st,a) = Q_r(st,a) + alpha*(tmp - Q_r(st,a));

         tmp1 = cost(st,a) + gamma*(P(st,a,1)*(d_1(1)*Q_c(1,1)+d_1(2)*Q_c(1,2)) + P(st,a,2)*(d_2(1)*Q_c(2,1)+d_2(2)*Q_c(2,2)) );
         Q_c_new(st,a) = Q_c(st,a) + alpha*(tmp1 - Q_c(st,a));
     end
     
 
 end
 max_diff_r = max(max(abs(Q_r - Q_r_new) ));
 max_diff_c = max(max(abs(Q_c - Q_c_new) ));

 if mod(i,10)==1
     
     fprintf('\n iteration = %d, (d_1, d_2) = %f %f Q factors reward\n',i, d_1(1), d_2(1))
     disp(Q_r_new)
     fprintf('\n Q factors cost\n')
     disp(Q_c_new)
     
     
            new_output_prob1 = D_mat(1, :);
            new_output_prob2 = D_mat(2, :);
         
           
            tmp_c(1) = new_output_prob1(1) * Q_c(1,1) + new_output_prob1(2) * Q_c(1,2) ;
            tmp_c(2) = new_output_prob2(1) * Q_c(2,1) + new_output_prob2(2) * Q_c(2,2) ;
            
            tmp_r(1) = new_output_prob1(1) * Q_r(1,1) + new_output_prob1(2) * Q_r(1,2) ;
            tmp_r(2) = new_output_prob2(1) * Q_r(2,1) + new_output_prob2(2) * Q_r(2,2) ;
             
            
            fprintf('\n -------------------------------------------------')
             fprintf('\n Iteration=%d Objective values = (%f, %f), constraint = %f %f ',i,tmp_r(1), tmp_r(2), tmp_c(1), tmp_c(2) )
        
            fprintf('\n  policy for state  1  (action 1), %f (action 2) %f', new_output_prob1(1), new_output_prob1(2));
            fprintf('\n  policy for state  2  (action 1), %f (action 2) %f', new_output_prob2(1), new_output_prob2(2));
            
 
            fprintf('\n Q factors reward \n')
            disp(Q_r);
            fprintf('\n Q factors cost \n')
            disp(Q_c);
            fprintf('\n -------------------------------------------------')
 end
 Q_r = Q_r_new;
 Q_c = Q_c_new;
 if (max_diff_r < delta) && (max_diff_c < delta) 
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
temp_reward = sum((D_mat.*Q_r)')
temp_cost = sum((D_mat.*Q_c)')
disp(D_mat)




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




function [val] = solveLPP2(st,Q_r,Q_c,B,d_mat, P, cost, gamma)  

p1 = optimvar('p1');
p2 = optimvar('p2');

obj =p1*Q_r(st,1) + p2*Q_r(st,2);
prob = optimproblem('Objective',obj,'ObjectiveSense','max');

s0 = 2; % state with constraint


if st==1
    j=2;
else
    j=1;
end

tmp1 = d_mat(s0,1)*(cost(s0,1) + gamma*P(s0,1,j)*(d_mat(j,1)*Q_c(j,1) + d_mat(j,2)*Q_c(j,2)) + gamma* P(s0,1,st)*(p1*Q_c(st,1) + p2*Q_c(st,2)));
tmp2 = d_mat(s0,2)*(cost(s0,2) + gamma*P(s0,2,j)*(d_mat(j,1)*Q_c(j,1) + d_mat(j,2)*Q_c(j,2)) + gamma* P(s0,2,st)*(p1*Q_c(st,1) + p2*Q_c(st,2)));
constr = tmp1 +tmp2;
prob.Constraints.c1 = constr <= B;
prob.Constraints.c2 = p1 +p2 == 1;
prob.Constraints.c3 = p1  >= 0;
prob.Constraints.c4 = p2  >= 0;


problem = prob2struct(prob);
options = optimoptions('linprog','Display','none');
problem.options = options;
[sol,~] = linprog(problem);

val = sol';

end


