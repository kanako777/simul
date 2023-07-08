import matplotlib.pyplot as plt
import numpy as np

from config import *
from utils import *
from UAV import UAV
from Bus import Bus
import scienceplots

MAX_ITER_COUNT = 200
num_uav = 5
num_bus = 3
task_cpu_cycle = 1000000
task_data_size = 10
task_delay = 1
budget = 30
simul_time = 50
num_x_step = simul_time / 5
SIGMA_SPEED = 100

paths = []

BUS_CPU_CYCLE = 10	# 버스의 최대 CPU 사이클
UAV_CPU_CYCLE = 10 # UAV의 최대 CPU 사이클

for i in range(num_bus):
    path = [(random.randint(0, MAP_SIZE), random.randint(0, MAP_SIZE))]
    while len(path) < NUM_PATH:
        x, y = path[-1]
        next_x = random.randint(max(0, x - random.randint(1, 50)), min(MAP_SIZE, x + random.randint(1, 50)))
        next_y = random.randint(max(0, y - random.randint(1, 50)), min(MAP_SIZE, y + random.randint(1, 50)))
        if math.dist((x, y), (next_x, next_y)) >= 50:
            path.append((next_x, next_y))
    paths.append(path)


buses_original = []
for i in range(num_bus):
    buses_original.append(Bus(i, 0, paths[i]))

uavs_original = []
for i in range(num_uav):
    uavs_original.append(UAV(i, 0, 0, 0, 0))


result_price1 = []
result_price2 = []
result_price3 = []
result_cpu1 = []
result_cpu2 = []
result_cpu3 = []
result_cpu4 = []
result_cpu5 = []
uav_utility1 = []
uav_utility2 = []
uav_utility3 = []
uav_utility4 = []
uav_utility5 = []

iter_cpu1 = []
iter_cpu2 = []
iter_cpu3 = []
iter_cpu4 = []
iter_cpu5 = []
iter_price1 = []
iter_price2 = []
iter_price3 = []
iter_utility1 = []
iter_utility2 = []
iter_utility3 = []
iter_utility4 = []
iter_utility5 = []


for simul_i in range(simul_time):

    iteration = 1
    changed = 1
    buses = deepcopy(buses_original)
    uavs = deepcopy(uavs_original)

    # 버스와 UAV의 인접매트릭스 초기화
    uav_bus_near_matrix = [[0 for _ in range(num_bus)] for _ in range(num_uav)]

    # UAV가 2대 이상의 버스와 인접한지를 나타내는 인접버스 리스트 초기화
    uav_has_more_than_2bus = [0 for _ in range(num_bus)]

    #print("simul time = ", simul_i)

    for uav in uavs:
        uav.init(task_cpu_cycle, task_data_size, task_delay, budget=budget)

    for bus in buses:
        bus.init()

    uavs[0].budget = 5
    #uavs[0].budget = simul_i + 1
    uavs[1].budget = 10
    uavs[2].budget = 15
    uavs[3].budget = 20
    uavs[4].budget = 25
    buses[0].cpu = 10
    buses[0].MAX_CPU = 10
    buses[1].cpu = 20
    buses[1].MAX_CPU = 20
    buses[2].cpu = 30
    buses[2].MAX_CPU = 30

    # 버스와 UAV간 전송률 및 딜레이를 구하고, 250m 이내에 위치한 버스와 UAV인 경우 각각 서로의 리스트(bus_list, uav_list)에 추가
    # 버스와 UAV의 인접매트릭스에도 반영

    iter_count = 0
    transmission_rate_list = [[0 for _ in range(num_bus)] for _ in range(num_uav)]
    uav_bus_maxcpu_list = [[0 for _ in range(num_bus)] for _ in range(num_uav)]

    for uav, bus in itertools.product(uavs, buses):
        uav.add_bus_id(bus.id)
        bus.add_uav_id(uav.id)
        uav_bus_near_matrix[uav.id][bus.id] = 1
        transmission_rate_list[uav.id][bus.id] = 10000

    # 인접매트릭스를 이용하여 인접버스 리스트 생성
    for i in range(num_uav):
        temp2 = 0
        for j in range(num_bus):
            temp2 = temp2 + uav_bus_near_matrix[i][j]

        if temp2 > 1:
            for j in range(num_bus):
                if uav_bus_near_matrix[i][j] > 0:
                    uav_has_more_than_2bus[j] = 1

    # 가격 설정 단계
    while (iteration):
        #iteration = 0
        iter_count += 1
        for i in range(num_uav):
            for j in range(num_bus):
                uav_bus_maxcpu_list[i][j] = 0

        # 모든 UAV에 대하여 주변의 버스로부터 구입가능한 최대 CPU를 구하는 부분
        for uav in uavs:
            price_sum = 0
            price_num = 0
            if uav.bus_id_list:  # UAV의 주변에 버스가 존재하면
                # 게임이론을 적용하기 위한 값 계산
                for uav_id in uav.bus_id_list:
                    price_sum += buses[uav_id].price
                    if buses[uav_id].price > 0:
                        price_num += 1
                for bus_id in uav.bus_id_list:
                    # 게임이론에 따라 UAV가 버스로부터 구입할 수 있는 최대의 CPU사이클 계산
                    MAX_CPU = min(uav.task_original['cpu_cycle'],
                                  (uav.budget + 1 * price_sum) / (buses[bus_id].price * price_num) - 1)
                    if MAX_CPU > 0:
                        uav_bus_maxcpu_list[uav.id][bus_id] = MAX_CPU

        no_change_count = 0
        num_list = 0

        # 모든 버스에 대하여, UAV의 인접버스리스트에 해당하는 경우에 게임이론을 적용하여 price를 변경
        # 이 과정을 끝내고 나면, 모든 버스는 가장 최적의 price를 설정하게 됨(모든 버스의 초기 price는 1)

        for bus in buses:
            if uav_has_more_than_2bus[bus.id] == 1:
                num_list += 1
                demand_sum_from_uav = 0
                for i in range(num_uav):
                    demand_sum_from_uav += uav_bus_maxcpu_list[i][bus.id]

                # 변경된 price 계산
                temp_price = max(round(bus.price + (demand_sum_from_uav - bus.MAX_CPU) / SIGMA_SPEED, 4), EPSILON)
                # 변경된 price와 기존 price의 차이가 threshold 이상인지 검사
                if abs(temp_price - bus.price) >= (1 / SIGMA_SPEED):

                    # 변경된 price와 기존 price의 차이가 threshold 이상이지만, 2회전 price와 가격차이가 크지 않은지 검사
                    if abs(bus.old_price - temp_price) <= (1 / SIGMA_SPEED) * 2:
                        no_change_count += 1

                    # 현재 price를 변경
                    else:
                        bus.change_price(temp_price)
                        iteration = 1

                # price를 변경하지 않은 버스의 대수를 계산
                else:
                    no_change_count += 1
        # 모든 버스가 더이상 price를 변경하지 않았다면 가격변경 중단

        #if no_change_count == num_list:
            #iteration = 0

        # 버스가 price를 바꿔가다가 더이상 바꾸지 않으면 while문을 벗어나야 하는데,
        # price를 계속 바꾸어 무한루프에 빠지는 현상이 있어서, 그걸 벗어나기 위해 iter_count를 사용
        # iter_count가 MAX_ITER_COUNT이상이면 price를 더이상 바꾸지 않고 중단

        if iter_count > MAX_ITER_COUNT:
            iteration = 0

        if simul_i == 0:
            iter_cpu1.append(sum(uav_bus_maxcpu_list[0]))
            iter_cpu2.append(sum(uav_bus_maxcpu_list[1]))
            iter_cpu3.append(sum(uav_bus_maxcpu_list[2]))
            iter_cpu4.append(sum(uav_bus_maxcpu_list[3]))
            iter_cpu5.append(sum(uav_bus_maxcpu_list[4]))
            iter_price1.append(buses[0].price)
            iter_price2.append(buses[1].price)
            iter_price3.append(buses[2].price)

    n_th = 0
    while (changed):
        n_th += 1
        changed = 0
        for uav in uavs:
            price_sum = 0
            price_num = 0
            uav.bus_id_list.sort(key=lambda x: buses[x].price, reverse=True)
            if uav.bus_id_list:
                for uav_id in uav.bus_id_list:
                    price_sum += buses[uav_id].price
                    if buses[uav_id].price > 0:
                        price_num += 1
                tmp_list = deepcopy(uav.bus_id_list)
                for uav_id in tmp_list:
                    # UAV가 버스로부터 CPU를 구매
                    # CPU를 구매한 UAV가 존재한다면 반복
                    if buses[uav_id].cpu > 0 and uav.purchase_cpu2(buses[uav_id],
                                                                  transmission_rate_list[uav.id][uav_id],
                                                                  price_sum, price_num,True):
                        changed = 1

    #print(uav_bus_maxcpu_list)

    if simul_i % 5 ==0:
        result_price1.append(buses[0].price)
        result_price2.append(buses[1].price)
        result_price3.append(buses[2].price)
        result_cpu1.append(sum(uav_bus_maxcpu_list[0]))
        result_cpu2.append(sum(uav_bus_maxcpu_list[1]))
        result_cpu3.append(sum(uav_bus_maxcpu_list[2]))
        result_cpu4.append(sum(uav_bus_maxcpu_list[3]))
        result_cpu5.append(sum(uav_bus_maxcpu_list[4]))
        uav_utility1.append(uavs[0].utility)
        uav_utility2.append(uavs[1].utility)
        uav_utility3.append(uavs[2].utility)
        uav_utility4.append(uavs[3].utility)
        uav_utility5.append(uavs[4].utility)


#print(result_price1,result_price2,result_price3)
#print(result_cpu1, result_cpu2, result_cpu3, result_cpu4, result_cpu5)

print(len(iter_cpu1),len(iter_cpu2),len(iter_cpu3),len(iter_cpu4),len(iter_cpu5))
print(iter_cpu1)
print(iter_cpu2)
print(iter_cpu3)
print(iter_cpu4)
print(iter_cpu5)

data1 = [result_price1, result_price2, result_price3]
data2 = [result_cpu1,result_cpu2,result_cpu3,result_cpu4,result_cpu5]
data3 = [uav_utility1, uav_utility2, uav_utility3, uav_utility4, uav_utility5]
data_iter_cpu = [iter_cpu1, iter_cpu2, iter_cpu3, iter_cpu4, iter_cpu5]
data_iter_cpu100 = [iter_cpu1[0:100], iter_cpu2[0:100], iter_cpu3[0:100], iter_cpu4[0:100], iter_cpu5[0:100]]
data_iter_price =[iter_price1, iter_price2, iter_price3]
data_iter_price100 =[iter_price1[0:100], iter_price2[0:100], iter_price3[0:100]]


marker = itertools.cycle(('+', '2', '.', 'x', '*'))
plt.style.use(['science', 'ieee', 'no-latex'])

x_idx = np.arange(1, num_x_step+1)
x = x_idx * 5
print(x_idx)

plt.figure(figsize=(3,3), dpi=500)

for i in range(4):
    plt.plot(x, data2[i], marker = next(marker), label="UAV"+str(i+1))

plt.xlabel("Budget of UAV1")
plt.ylabel("Demands")
#plt.ylim((0, 20))
plt.legend(loc='upper left')
plt.legend(frameon=True)
plt.savefig("./test_graphs/" + "UAV demands")
plt.clf()

for i in range(4):
    plt.plot(x, data3[i], marker = next(marker), label="UAV"+str(i+1))

plt.xlabel("Budget of UAV1")
plt.ylabel("UAV Utility")
#plt.ylim((0, 20))
plt.legend(loc='upper left')
plt.legend(frameon=True)
plt.savefig("./test_graphs/" + "UAV utility")
plt.clf()

for i in range(3):
    plt.plot(x, data1[i], marker = next(marker), label="BUS"+str(i+1))

plt.xlabel("Budget of UAV1")
plt.ylabel("CPU Price")
plt.legend(loc='upper left')
plt.legend(frameon=True)
plt.savefig("./test_graphs/" + "CPU Prices")
plt.clf()


marker2 = itertools.cycle(('o', 'v', '.', 's'))
plt.figure(figsize=(5,5), dpi=500)


x = np.arange(0, len(iter_cpu1))
for i in range(4):
    plt.plot(x, data_iter_cpu[i], marker = next(marker2), markersize=3, label="UAV"+str(i+1))

plt.xlabel("Iterations")
plt.ylabel("Demands")
#plt.ylim((0, 20))
plt.legend(loc='upper left')
plt.legend(frameon=True)
plt.savefig("./test_graphs/" + "iter demand")
plt.clf()

x = np.arange(0, len(iter_price1))
for i in range(3):
    plt.plot(x, data_iter_price[i], marker = next(marker2), markersize=3, label="BUS"+str(i+1))

plt.xlabel("Iterations")
plt.ylabel("CPU Price")
#plt.ylim((0, 20))
plt.legend(loc='upper left')
plt.legend(frameon=True)
plt.savefig("./test_graphs/" + "iter price")
plt.clf()