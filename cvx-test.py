import numpy as np
import cvxpy as cvx

f = np.array([[3],[3/2]])
lb = np.array([[-1], [0]])
ub = np.array([[2], [3]])

x = cvx.Variable([2,1])

obj = cvx.Minimize(-f.T@x)
constraints = [lb <=x, x <= ub]

prob = cvx.Problem(obj, constraints)
result = prob.solve()

#print(x.value)
#print(result)

class Task:
    def __init__(self, id, frequency, size, cycle, delay):
        self.id = id
        self.size = size
        self.frequency = frequency
        self.cycle = cycle
        self.delay = delay


#e_u_loc = e_u_m * task_size * task_frequency * ( X_U * f_u_k^2 + x_u_k * F_U^2 ) + 1/2 * ( (X_U - x_u_k)^2 + (F_U - f_u_k)^2 )
#e_u_tx = P_u_b * X_B * task_size / R_u_b
#t_u_loc = task_size * task_frequency * ( 1/2 * ((X_U + 1/F_U)^2 - (x_u_k)^2 - (1/f_u_k)^2) - (x_u_k * (X_U - x_u_k)) + (1/f_u_k)^3 * (1/F_U - 1/f_u_k))
#t_b_off = task_size * task_frequency * ( 1/2 * ((X_B + 1/F_B)^2 - (x_b_k)^2 - (1/f_b_k)^2) - (x_b_k * (X_B - x_b_k)) + (1/f_b_k)^3 * (1/F_B - 1/f_b_k))
#t_tx = X_B * task_size / R_u_b


from cvxpy import *
M = 4
U = 1
B = 3
FU = 1
FB = 2
omega1 = 1
omega2 = 1
epsilon_u = 1

xum = cvx.Variable([M, U])
xbm = cvx.Variable([M, B])
fum = cvx.Variable([M, U])
fbm = cvx.Variable([M, B])
sum_xbm = cvx.Variable([M, B])
sum_fum = cvx.Variable([M, U])
sum_fbm = cvx.Variable([M, B])
mum = cvx.Variable(M)

P_ub = np.ones((U, B)) * 0.1
R_ub = np.ones((U, B)) * 50
sm = np.ones(M) * 10
fm = np.ones(M) * 10

x_u_k = np.ones((M, U))
x_b_k = np.ones((M, B))
f_u_k = np.ones((M, U))
f_b_k = np.ones((M, B))
t_u_loc = np.zeros(M)

cost1 = 0
cost2 = 0
#t_tx_cost = np.zeros((M, B))
#t_off_cost = np.zeros((M, B))
#t_tx = np.zeros(M)
#t_off = np.zeros(M)

t_tx_cost = [[0 for _ in range(B)] for _ in range(M)]
t_off_cost = [[0 for _ in range(B)] for _ in range(M)]
t_tx = [0 for _ in range(M)]
t_off = [0 for _ in range(M)]


for m in range(M):
    for u in range(U):
        for b in range(B):
            cost1 += P_ub[u,b] * xbm[m,b] * sm[m] / R_ub[u,b]
            t_tx_cost[m][b] = xbm[m,b] * sm[m] / R_ub[u,b]
            t_off_cost[m][b] = sm[m] * fm[m] * (
                        1 / 2 * ((xbm[m, b] + 1 / fbm[m, b]) ** 2 - (x_b_k[m, b]) ** 2 - (1 / f_b_k[m, b]) ** 2) - (
                            x_b_k[m, b] * (xbm[m, b] - x_b_k[m, b])) + (1 / f_b_k[m, b]) ** 3 * (
                                    1 / fbm[m, b] - 1 / f_b_k[m, b]))

        e_u_loc = epsilon_u * sm[m] * fm[m] * (( xum[m,u] * f_u_k[m,u]**2 + x_u_k[m,u] * fum[m,u]**2 ) + 1/2 * ( (xum[m,u] - x_u_k[m,u])**2 + (fum[m,u] - f_u_k[m,u])**2 ))
        cost1 += omega2 * (cost1 + e_u_loc)

        #t_u_loc = sm[m] * fm[m] * ( 1/2 * ( (xum[m,u] + inv_pos(fum[m,u]))**2 - (x_u_k[m,u])**2 - (1/f_u_k[m,u])**2) - (x_u_k[m,u] * (inv_pos(xum[m, u]) - x_u_k[m,u])) + (1/f_u_k[m,u])**3 * (inv_pos(fum[m, u]) - 1/f_u_k[m,u]))


    cost1 += omega1 * mum[m] + cost1
    #cost1 += omega1 + cost1

for m in range(M):
    for u in range(U):
        for b in range(B):
            sum_xbm += xbm[m,b]
            #sum_fum += fum[m,u]
            #sum_fbm += fbm[m,b]
            t_tx[m] += t_tx_cost[m][b]
            # print(t_tx[m])
            t_off[m] += t_off_cost[m][b]

# print(t_tx)
sum_xbm = cvxpy.sum(xbm, axis=1, keepdims=True)

obj = cvx.Minimize(cost1)
constraints = \
    [0 <= xum, xum <= 1] + \
    [0 <= xbm, xbm <= 1] + \
    [xum + sum_xbm == 1] + \
    [0 <= fum, cvx.sum(fum) <=FU] + \
    [0 <= fbm, cvx.sum(fbm) <=FB] + \
    [t_u_loc <= mum]  \
    # [t_tx <= mum] \

prob = cvx.Problem(obj, constraints)
result = prob.solve()

print(xum.value)
print(xbm.value)
print(fum.value)
print(fbm.value)
print(mum.value)
print(result)