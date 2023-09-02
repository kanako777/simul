import numpy as np
import cvxpy as cvx
from cvxpy import *
import math

# Define the variable
x = cvx.Variable(pos=True)
y = cvx.Variable(pos=True)
z = cvx.Variable(pos=True)

cost = square(x + inv_pos(y))

# Define the objective function
objective = cvx.Minimize(z)

# Define the constraints
constraints = [z>=cost] + [0<=x, x<=1] + [0<=y, y<=1]

# Formulate the problem
problem = cvx.Problem(objective, constraints)

# Solve the problem
problem.solve(gp=False)

# Print the results
print("Optimal value of x:", x.value)
print("Optimal value of y:", y.value)
print("Optimal value of z:", z.value)


class Task:
    def __init__(self, id, frequency, size, cycle, delay):
        self.id = id
        self.size = size
        self.frequency = frequency
        self.cycle = cycle
        self.delay = delay

from cvxpy import *
M = 4
U = 1
B = 4
FU = 10
FB = 20
omega1 = 1
omega2 = 1
epsilon_u = 1

rho_um = cvx.Variable([M, U], pos=True)
rho_bm = cvx.Variable([M, B], pos=True)
fum = cvx.Variable([M, U], pos=True)
fbm = cvx.Variable([M, B], pos=True)
mum = cvx.Variable(M, pos=True)


P_ub = np.ones((U, B)) * 1
R_ub = np.ones((U, B)) * 1
tou_rho_um = np.ones((M, U)) * 1
tou_f_um = np.ones((M, U)) * 1
sm = np.ones(M) * 10 # size of task m
cm = np.ones(M) * 10 # cpu cycles required for task m
dm = np.ones(M) * 10 # delay constraints for task m

rho_um_k = np.ones((M, U)) * 0.2
rho_bm_k = np.ones((M, B)) * 0.2
f_u_k = np.ones((M, U)) * 2.5
f_b_k = np.ones((M, B)) * 5
mum_k = np.ones((M, U))
#mum = np.ones((M, U))
t_u_loc = np.zeros(M)

difference = 2
criteria = 1
result_initial = 0
loop = 1

while (loop) :

    loop = 1
    initial_result = 0
    stable = 2

    e_um_cost = 0
    e_tx_cost = 0
    e_bm_cost = 0

    t_um_cost = 0
    t_tx_cost = 0
    t_bm_cost = 0

    P2_energy_cost = 0
    P2_time_cost = 0

    for m in range(M):

        e_um_cost = 0
        t_um_cost = 0

        for u in range(U):

            t_tx_cost = 0
            e_tx_cost = 0
            t_bm_cost = 0

            for b in range(B):

                ####### START (for_b) #########
                # task m 처리를 위해 UAV -> bus b로 데이터를 보내는 데 걸리는 시간 (t_tx) 계산 : 식(9)
                t_tx_cost += rho_bm[m, b] * sm[m] / R_ub[u, b]

                # task m 처리를 위해 UAV -> bus b로 데이터를 보내는 데 필요한 에너지(e_tx) 계산 : 식(10)
                e_tx_cost += rho_bm[m,b] * P_ub[u,b] * sm[m] / R_ub[u,b]

                # bus b가 task m 처리를 위해 걸리는 시간 (t~_bm) 계산 : 식(21)
                #t_bm_cost += cm[m] * (0.5 * ( power(rho_bm[m, b],2) + power(inv_pos(fbm[m, b]),2) - power(rho_bm_k[m, b],2) - power(inv_pos(f_b_k[m, b]),2) - (rho_bm_k[m, b] * (rho_bm[m, b] - rho_bm_k[m, b])) + (power(inv_pos(f_b_k[m, b]),3) * (inv_pos(fbm[m, b]) - inv_pos(f_b_k[m, b])))))
                t_bm_cost += cm[m] * (0.5 * (power( rho_bm[m, b] + inv_pos(fbm[m, b]), 2) - power(rho_bm_k[m, b],2) - power(inv_pos(f_b_k[m, b]),2) - (rho_bm_k[m, b] * (rho_bm[m, b] - rho_bm_k[m, b])) + (power(inv_pos(f_b_k[m, b]),3) * (inv_pos(fbm[m, b]) - inv_pos(f_b_k[m, b])))))
                ####### END (for_b) ###########

            # UAV가 task m 처리를 위해 걸리는 시간 (t~_um) 계산 : 식(19)
            t_um_cost += cm[m] * ( 0.5 * (power(rho_um[m,u] + inv_pos(fum[m,u]),2) - power(rho_um_k[m,u],2) - power(inv_pos(f_u_k[m,u]),2) - ( rho_um_k[m,u] * (rho_um[m, u] - rho_um_k[m,u]) ) + power(inv_pos(f_u_k[m,u]),3) * (inv_pos(fum[m, u]) - inv_pos(f_u_k[m,u]))))

            # UAV가 task m 처리를 위해 필요한 에너지 (e~_um) 계산 : 식(16)
            e_um_cost += epsilon_u * cm[m] * (( rho_um[m,u] * f_u_k[m,u]**2 + rho_um_k[m,u] * fum[m,u]**2 )) + ( 0.5 * tou_rho_um[m,u] * (rho_um[m,u] - rho_um_k[m,u])**2 ) + ( 0.5 * tou_f_um[m,u] * (fum[m,u] - f_u_k[m,u])**2 )

        # mum[m] = t_bm_cost + t_tx_cost
        #P2_time_cost += omega1 * (t_tx_cost + t_bm_cost)

        P2_time_cost += omega1 * mum[m]
        P2_energy_cost += omega2 * (e_um_cost + e_tx_cost)

    sum_rho_bm = cvxpy.sum(rho_bm, axis=1, keepdims=True)

    P2 = P2_time_cost + P2_energy_cost
    obj = cvx.Minimize(P2)
    constraints = \
        [0 <= fum, cvx.sum(fum) <=FU] + \
        [0 <= fbm, cvx.sum(fbm) <=FB] + \
        [rho_um + sum_rho_bm == 1] + \
        [0 <= rho_um, rho_um <= 1] + \
        [0 <= rho_bm, rho_bm <= 1] + \
        [mum <= dm] + \
        [mum >= t_um_cost] + \
        [mum >= t_bm_cost+t_tx_cost]

    prob = cvx.Problem(obj, constraints)
    result = prob.solve()

    criteria = abs(result - initial_result)
    initial_result = result

    rho_um_k = rho_um.value
    rho_bm_k = rho_bm.value
    f_u_k = fum.value
    f_b_k = fbm.value
    mum_k = mum.value

    print(rho_um.value)
    print(rho_bm.value)
    print(fum.value)
    print(fbm.value)
    print(mum.value)
    print(result)


