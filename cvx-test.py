import numpy as np
import cvxpy as cvx
import time
from cvxpy import *
import math
import random
import time
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib.font_manager
from cvxpy import *

def dbm_to_watt(dbm):
    return 10 ** (dbm / 10) / 1000

def watt_to_dbm(watt):
    return 10 * math.log10(1000 * watt)

NUM_BUS = 4 # 운행하는 버스의 대수(버스별로 자기 노선(path)을 가짐
NUM_UAV = 1  # UAV의 개수
NUM_TASK = 10
FU = 10
FB = 30

NUM_PATH = 100  # 버스 운행경로별로 지나는 정류장(지점)의 개수
MAP_SIZE = 1000  # MAP크기
MIN_HEIGHT = 100 # UAV의 최저 고도
MAX_HEIGHT = 150 # UAV의 최고 고도
M = NUM_TASK
U = NUM_UAV
B = NUM_BUS
omega1 = 1
omega2 = 1
epsilon_u = 0.1
lamda = 0.5


TIME_INTERVAL = 1
SIMUL_TIME = 1    # 시뮬레이션을 반복하는 횟수(t)

X, Y, Z = 500, 500, 100
BUS_POS = [100, 200, 300, 400, 500, 600, 700, 800, 900]

count = 0
position = 0

class UAV:
    def __init__(self, id, x, y, z):
        self.id = id
        self.x = x
        self.y = y
        self.z = z
        self.cpu = FU

class BUS:
    def __init__(self, id, path):
        self.id = id
        self.path = path
        self.reset()

    def reset(self):
        self.path_index = 0
        self.location = self.path[self.path_index]
        self.x = self.location[0]
        self.y = self.location[1]
        #self.x = BUS_POS[count]
        #self.y = BUS_POS[count]

    def move(self):
        self.path_index += 1
        if self.path_index == len(self.path):
            self.path_index = 0
        self.location = self.path[self.path_index]
        self.x = self.location[0]
        self.y = self.location[1]

class TASK:
    def __init__(self, id, size, cycle, delay):
        self.id = id
        self.size = size
        self.cycle = cycle
        self.delay = delay


rho_um = cvx.Variable([M, U], pos=True)
rho_bm = cvx.Variable([M, B], pos=True)
fum = cvx.Variable([M, U])
fbm = cvx.Variable([M, B])
mum = cvx.Variable([M, U])

tou_rho_um = np.ones((M, U)) * 1
tou_f_um = np.ones((M, U)) * 1

print("### SIMULATION START ###")
paths = []
simul_time = 0
num_bus = NUM_BUS

for i in range(num_bus):
    path = [(random.randint(0, MAP_SIZE), random.randint(0, MAP_SIZE))]
    while len(path) < NUM_PATH:
        x, y = path[-1]
        next_x = random.randint(x - random.randint(1, 50), x + random.randint(1, 50))
        next_y = random.randint(y - random.randint(1, 50), y + random.randint(1, 50))
        if next_x>0 and next_x <MAP_SIZE and next_y>0 and next_y < MAP_SIZE:
            path.append((next_x, next_y))

    paths.append(path)

# UAV 생성
uavs_original = []
for i in range(NUM_UAV):
    uavs_original.append(UAV(i, X, Y, Z))

# BUS 생성
buses_original = []
for i in range(num_bus):
    buses_original.append(BUS(i, paths[i]))
    count +=1

# UAV와 BUS간의 전송률 계산
Distance = [[0 for j in range(NUM_BUS)] for i in range(NUM_UAV)]
P_ub = [[1 for j in range(NUM_BUS)] for i in range(NUM_UAV)] # 전송 파워 (W)
R_ub = [[1 for j in range(NUM_BUS)] for i in range(NUM_UAV)]
W_ub = [[10**7 for j in range(NUM_BUS)] for i in range(NUM_UAV)] # 대역폭 (Hz)

alpha_0 = 10 ** ((-50.0) / 10)  # 1m 참조 거리에서의 수신 파워 (-50dB를 와트로 변환)
Noise = 10 ** ((-100.0 - 30) / 10)  # 노이즈 파워 (-100dBm를 와트로 변환)

for i in range(NUM_UAV):
    for j in range(num_bus):
        Distance[i][j] = math.dist((uavs_original[0].x,uavs_original[0].y,uavs_original[0].z), (buses_original[j].x,buses_original[j].y,0))

        # 전송률 계산 (Shannon Capacity 공식 사용)
        SNR = (P_ub[i][j] * alpha_0) / (Noise * Distance[i][j]**2)
        R_ub[i][j] = W_ub[i][j] * math.log2(1 + SNR) / 1E9  # Gbps

        # 결과 출력
        print("거리 :", Distance[i][j], "m ", " 전송률 :", R_ub[i][j], "Gbps")

# TASK 생성
task_original = []
sm = np.ones((M, U))
cm = np.ones((M, U))
dm = np.ones((M, U))

for i in range(NUM_TASK):
    sm[i] = random.randint(1, 20) / 1E3 # 10~20Mbits (Gbits 단위로 변환)
    #sm[i] = 20 / 1E3 # 10~20Mbits (Gbits 단위로 변환)
    cm[i] = sm[i] * 200 # 200 cycles per bit (Gcycls 단위로 변환)
    dm[i] = 10  # 100 seconds

print(sm)

rho_um_k = np.ones((M, U)) * 1 / (U+B)
rho_bm_k = np.ones((M, B)) * 1 / (U+B)
f_u_k = np.ones((M, U)) * FU / M
f_b_k = np.ones((M, B)) * FB / M
mum_k = np.ones((M, U)) * 100
#mum_k = np.ones((M, U)) * cm / (rho_um_k * FU)

criteria = 0.001
pre_result = 0
loop = 1


while (loop<=50) :

    e_um_cost = 0
    e_tx_cost = 0
    e_bm_cost = 0

    t_um_cost = 0
    t_tx_cost = 0
    t_bm_cost = 0

    P2_energy_cost = 0
    P2_time_cost = 0

    for m in range(M):

        for u in range(U):

            for b in range(B):

                ####### START (for_b) #########
                # task m 처리를 위해 UAV -> bus b로 데이터를 보내는 데 걸리는 시간 (t_tx) 계산 : 식(9)
                t_tx_cost += rho_bm[m, b] * sm[m] / R_ub[u][b]

                # task m 처리를 위해 UAV -> bus b로 데이터를 보내는 데 필요한 에너지(e_tx) 계산 : 식(10)
                e_tx_cost += rho_bm[m,b] * P_ub[u][b] * sm[m] / R_ub[u][b]

                # bus b가 task m 처리를 위해 걸리는 시간 (t~_bm) 계산 : 식(21)
                #t_bm_cost += cm[m] * (0.5 * ( power(rho_bm[m, b],2) + power(inv_pos(fbm[m, b]),2) - power(rho_bm_k[m, b],2) - power(inv_pos(f_b_k[m, b]),2) - (rho_bm_k[m, b] * (rho_bm[m, b] - rho_bm_k[m, b])) + (power(inv_pos(f_b_k[m, b]),3) * (inv_pos(fbm[m, b]) - inv_pos(f_b_k[m, b])))))
                t_bm_cost += cm[m] * (0.5 * (power( rho_bm[m, b] + inv_pos(fbm[m, b]), 2) - power(rho_bm_k[m, b],2) - power(inv_pos(f_b_k[m, b]),2) - (rho_bm_k[m, b] * (rho_bm[m, b] - rho_bm_k[m, b])) + (power(inv_pos(f_b_k[m, b]),3) * (inv_pos(fbm[m, b]) - inv_pos(f_b_k[m, b])))))
                ####### END (for_b) ###########

            # UAV가 task m 처리를 위해 걸리는 시간 (t~_um) 계산 : 식(19)
            t_um_cost += cm[m] * ( 0.5 * (power(rho_um[m,u] + inv_pos(fum[m,u]),2) - power(rho_um_k[m,u],2) - power(inv_pos(f_u_k[m,u]),2) - ( rho_um_k[m,u] * (rho_um[m, u] - rho_um_k[m,u]) ) + power(inv_pos(f_u_k[m,u]),3) * (inv_pos(fum[m, u]) - inv_pos(f_u_k[m,u]))))

            # UAV가 task m 처리를 위해 필요한 에너지 (e~_um) 계산 : 식(16)
            e_um_cost += epsilon_u * cm[m] * (( rho_um[m,u] * f_u_k[m,u]**2 + rho_um_k[m,u] * fum[m,u]**2 )) + ( 0.5 * tou_rho_um[m,u] * (rho_um[m,u] - rho_um_k[m,u])**2 ) + ( 0.5 * tou_f_um[m,u] * (fum[m,u] - f_u_k[m,u])**2 )

        #mum[m] = max(t_um_cost, t_bm_cost + t_tx_cost)
        #P2_time_cost += omega1 * (t_tx_cost + t_bm_cost)

        P2_time_cost += omega1 * mum[m]
        P2_energy_cost += omega2 * (e_um_cost + e_tx_cost)

    P2 = P2_time_cost + P2_energy_cost
    obj = cvx.Minimize(P2)
    constraints = \
        [0 <= fum, cvxpy.sum(fum) <=FU] + \
        [0 <= fbm, cvxpy.sum(fbm, axis=0, keepdims=True) <=FB] + \
        [rho_um + cvxpy.sum(rho_bm, axis=1, keepdims=True) == 1] + \
        [0 <= rho_um, rho_um <= 1] + \
        [0 <= rho_bm, rho_bm <= 1] + \
        [mum <= dm] + \
        [mum >= t_um_cost] + \
        [mum >= t_bm_cost + t_tx_cost]

    prob = cvx.Problem(obj, constraints)
    result = prob.solve(solver=SCS)

    if loop % 10 == -1:
    #if 1:
        np.set_printoptions(precision=3)

        print("Iteration : ", loop)
        print("Status : ", prob.status)
        print(rho_um.value)
        print(rho_bm.value)
        print(fum.value)
        print(fbm.value)
        print(mum.value)
        print(result)
        print("")

    rho_um_k = lamda * rho_um.value + (1 - lamda) * rho_um_k
    rho_bm_k = lamda * rho_bm.value + (1 - lamda) * rho_bm_k
    f_u_k = lamda * fum.value + (1 - lamda) * f_u_k
    f_b_k = lamda * fbm.value + (1 - lamda) * f_b_k
    mum_k = mum.value

    loop += 1


# creating the dataset
xaxis = np.arange(1, M+1, 1)
x = np.zeros((M, U))
y = np.zeros((M, B))

Color = ['r', 'g', 'b', 'r', 'g', 'b']

for m in range(M):
    for b in range(B):
        y[m][b] = rho_bm[m][b].value

for m in range(M):
    for u in range(U):
        x[m][u] = rho_um[m][u].value

y1 = y.transpose()
x1 = x.transpose()
print(y1)
print(x1)

plt.bar(xaxis, x1[0], label="UAV")
bottom = x1[0]

for b in range(B):
    plt.bar(xaxis, y1[b], bottom=bottom, label="BUS" + str(b) + " : " + str(round(Distance[0][b],1)) +"m")
    bottom += y1[b]

plt.legend(loc='best')
plt.legend(frameon=True)


# plt.bar(x, values, color='dodgerblue')
# plt.bar(x, values, color='C2')
# plt.bar(x, values, color='#e35f62')

plt.xlabel("Task")
plt.ylabel("Offloading ratio.")
plt.xticks(xaxis)
plt.show()
