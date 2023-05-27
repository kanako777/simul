import random
import time
from copy import deepcopy
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import numpy as np
from config import *
from utils import *
from UAV import UAV
from Bus import Bus
import scienceplots

import argparse

# POI 위치 설정
X, Y, Z = 500, 500, 0
RANDOM_TASK = [{'name': "low", 'min': 5, 'max': 10}, {'name': "medium", 'min': 10, 'max': 20},
               {'name': "large", 'min': 20, 'max': 50}]
SCHEME = ["Game", "Matching", "Offloading", "Local"]
SIMUL_NAME = ["Bus Num", "UAV Num", "Budget", "Scheme", "CPU required"]
SAVE_X_NAME = ["Bus", "UAV", "Budget", "Scheme", "CPU required"]
SAVE_Y_NAME = ["overhead", "UAV_utility", "bus_utility", "bus_num", "price"]

NUM_OBJECT = [NUM_BUS, NUM_UAV, BUDGET, len(SCHEME), len(RANDOM_TASK)]
NUM_STEP = [NUM_BUS_STEP, NUM_UAV_STEP, NUM_BUDGET_STEP, len(SCHEME), len(RANDOM_TASK)]
STEP = [BUS_STEP, UAV_STEP, BUDGET_STEP, 1, 1]
X_LABEL = ["Number of buses", "Number of UAVs", "Budget", "Scheme", "CPU required"]
Y_LABEL = ["UAV overhead", "UAV utility", "Bus utility", "Avg. # of offloaded buses per UAV", "CPU price"]
LEGEND_LABEL = ["Bus=", "UAV=", "Budget=", "", ""]


def simul_value(type, i):
    if type == 0:
        return NUM_BUS - i * BUS_STEP
    elif type == 1:
        return NUM_UAV - i * UAV_STEP
    elif type == 2:
        return BUDGET - i * BUDGET_STEP
    elif type == 3:
        return SCHEME[i]
    elif type == 4:
        return RANDOM_TASK[i]
    return -1


def mean_without_outliers(lst: list, decision):
    lst.sort()
    l = len(lst)
    l_min = int(l * 0.1)
    l_max = int(l * 0.9)
    # lst = lst[l_min:l_max]
    return round(Average(lst), decision)


if __name__ == "__main__":
    # parsing / default = UAV-Bus task
    parser = argparse.ArgumentParser(description="_")
    parser.add_argument("--x", type=int, default=1, help="x value in graph. range 0~3")
    parser.add_argument("--y", type=int, default=0, help="label in graph, range 0~3")
    parser.add_argument("--real", type=bool, default=0, help="flag for using real data")
    args = parser.parse_args()

    print("### SIMULATION START ###")
    paths = []
    simul_time = 0
    num_bus = NUM_BUS
    num_x_step = NUM_STEP[args.x]
    num_y_step = NUM_STEP[args.y]

    if args.real:
        if args.x == 0:
            num_x_step = 1
        if args.y == 0:
            num_y_step = 1
        simul_time = 3

        with open("./buspos.txt", "r") as fp:
            t = 0
            while t < simul_time:
                path = []
                line = fp.readline()
                poslst = line.split('/')[:-1]
                for pos in poslst:
                    x, y = np.array(pos.split(','), dtype=np.float32)
                    path.append((x, y))
                paths.append(path)
                t += 1

        paths = np.array(paths).transpose((1, 0, 2))
        num_bus = len(paths)
    else:
        # make environment
        simul_time = SIMUL_TIME
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

    # POI로부터 일정거리 이내에 위치하도록 UAV 생성
    uavs_original = []
    for i in range(NUM_UAV):
        uavs_original.append(UAV(i, X, Y, Z))

    # for graph
    uav_bus_avg_overhead = [[0 for _ in range(num_x_step)] for _ in range(num_y_step)]
    uav_avg_utility = [[0 for _ in range(num_x_step)] for _ in range(num_y_step)]
    bus_avg_utility = [[0 for _ in range(num_x_step)] for _ in range(num_y_step)]
    uav_avg_bus_num = [[0 for _ in range(num_x_step)] for _ in range(num_y_step)]
    bus_avg_price = [[0 for _ in range(num_x_step)] for _ in range(num_y_step)]

    buses = deepcopy(buses_original)
    uavs = deepcopy(uavs_original)
    scheme = SCHEME[0]
    budget = BUDGET
    task_range = {'min': TASK_CPU_CYCLE, 'max': TASK_CPU_CYCLE}
    # UAV 대수 - 버스의 대수를 점점 줄여나가면서 시뮬레이션
    for i in range(num_y_step):
        if args.y == 3:
            scheme = SCHEME[i]
        elif args.y == 4:
            task_range = RANDOM_TASK[i]

        if args.x == 0:
            buses = deepcopy(buses_original)
        elif args.x == 1:
            uavs = deepcopy(uavs_original)
        elif args.x == 2:
            budget = BUDGET
        elif args.x == 3:
            pass

        for j in range(num_x_step):
            print(
                f"### {SIMUL_NAME[args.y]}:{simul_value(args.y, i)} {SIMUL_NAME[args.x]}:{simul_value(args.x, j)} simulation start")

            # simulate
            simulation(simul_time, uavs, buses, scheme=scheme, budget=budget, random_task_range=task_range)
            print("### SIMULATION RESULT ###")

            tmp_overhead = []
            tmp_uav_utility = []
            tmp_bus_utility = []
            tmp_bus_num = []
            tmp_bus_avg_price = []

            under = 0
            upper = 0

            for uav in uavs:
                print(uav.overhead_list)
                if len(uav.overhead_list) > 0:
                    avg_overhead = mean_without_outliers(uav.overhead_list, 4)
                    tmp_overhead.append(avg_overhead)
                if len(uav.utility_list) > 0:
                    avg_utility = mean_without_outliers(uav.utility_list, 4)
                    tmp_uav_utility.append(avg_utility)
                if len(uav.bus_num_list) > 0:
                    avg_bus_num = mean_without_outliers(uav.bus_num_list, 4)
                    tmp_bus_num.append(avg_bus_num)
                if len(uav.price_list) > 0:
                    avg_price = round(Average(uav.price_list), 4)
                    tmp_bus_avg_price.append(avg_price)

                print(
                    f"UAV(ID={uav.id}) has overhead")  # : {avg_overhead}, utility : {avg_utility}, price : {avg_price}")

            for bus in buses:
                if len(bus.utility_list) > 0:
                    avg_utility = mean_without_outliers(bus.utility_list, 4)
                    # print(bus.price_list)
                    tmp_bus_utility.append(avg_utility)
                print(f"BUS(ID={bus.id}) has utility ")  #: {avg_utility}")

            uav_bus_avg_overhead[i][j] = round(Average(tmp_overhead), 4)  # mean_without_outliers(tmp_overhead,4)
            uav_avg_utility[i][j] = round(Average(tmp_uav_utility), 4)  # mean_without_outliers(tmp_uav_utility,4)
            uav_avg_bus_num[i][j] = round(Average(tmp_bus_num), 4)  # mean_without_outliers(tmp_bus_num,4)
            bus_avg_utility[i][j] = round(Average(tmp_bus_utility), 4)  # mean_without_outliers(tmp_bus_utility,4)
            bus_avg_price[i][j] = round(Average(tmp_bus_avg_price), 4)

            # print(f"UAV overhead : {AVE}, Under : {under}, Upper : {upper}")
            if args.x == 0:
                for k in range(BUS_STEP):
                    del buses[-1]
            elif args.x == 1:
                for k in range(UAV_STEP):
                    del uavs[-1]
            elif args.x == 2:
                budget -= BUDGET_STEP
            elif args.x == 3:
                scheme = SCHEME[j]
            elif args.x == 4:
                task_range = RANDOM_TASK[j]

        if args.y == 0:
            for k in range(BUS_STEP):
                del buses[-1]
        elif args.y == 1:
            for k in range(UAV_STEP):
                del uavs[-1]
        elif args.y == 2:
            budget -= BUDGET_STEP

    # print result

    x_idx = np.arange(0, num_x_step)
    x = NUM_OBJECT[args.x] - x_idx * STEP[args.x]
    legend_value_idx = np.arange(0, num_y_step)
    legend_value = []
    for i in legend_value_idx:
        v = NUM_OBJECT[args.y] - i * STEP[args.y]
        if args.y == 3:
            v = SCHEME[NUM_OBJECT[args.y] - v]
        elif args.y == 4:
            v = RANDOM_TASK[NUM_OBJECT[args.y] - v]['name']
        legend_value.append(v)
    data = [uav_bus_avg_overhead, uav_avg_utility, bus_avg_utility, uav_avg_bus_num, bus_avg_price]

    plt.style.use(['science', 'ieee', 'no-latex'])

    for i in range(5):
        for j in range(len(legend_value)):
            cubic_interpolation_model = interp1d(x, data[i][j], kind="cubic")
            X_ = np.linspace(x.min(), x.max(), 500)
            Y_ = cubic_interpolation_model(X_)
            plt.plot(X_, Y_, label=LEGEND_LABEL[args.y] + str(legend_value[j]))

            # plt.plot(x, data[i][j], label=LEGEND_LABEL[args.y] + str(legend_value[j]))

        plt.xlabel(X_LABEL[args.x])
        plt.ylabel(Y_LABEL[i])
        plt.legend(loc='upper right')
        plt.legend(frameon=True)
        plt.savefig("./test_graphs/" + SAVE_X_NAME[args.x] + "_" + SAVE_X_NAME[args.y] + "_" + SAVE_Y_NAME[i])
        plt.clf()

    cubic_interpolation_model0 = interp1d(x, data[0][0], kind="cubic")
    cubic_interpolation_model1 = interp1d(x, data[1][0], kind="cubic")
    cubic_interpolation_model2 = interp1d(x, data[2][0], kind="cubic")

    # Plotting the Graph
    X_ = np.linspace(x.min(), x.max(), 500)
    Y_0 = cubic_interpolation_model0(X_)
    Y_1 = cubic_interpolation_model1(X_)
    Y_2 = cubic_interpolation_model2(X_)

    plt.plot(X_, Y_0, label="UAV overhead")
    plt.plot(X_, Y_1, label="UAV utility")
    plt.plot(X_, Y_2, label="Bus utility")

    # plt.plot(x, data[0][0], label="UAV overhead")
    # plt.plot(x, data[1][0], label="UAV utility")
    # plt.plot(x, data[2][0], label="Bus utility")

    plt.xlabel(X_LABEL[args.x])
    plt.ylabel("overhead,utility")
    plt.legend(loc='upper right')
    plt.legend(frameon=True)
    plt.savefig("./test_graphs/" + SAVE_X_NAME[args.x] + "_" + SAVE_X_NAME[args.y] + "_" + "overutil")