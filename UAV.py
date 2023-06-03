from config import *
import random
import math
from Bus import Bus


class UAV:
    def __init__(self, id, x, y, z):
        self.id = id

        # init position
        angle = random.uniform(0, 2 * math.pi)
        distance = random.uniform(0, POI_RADIUS)
        self.x = int(x + distance * math.cos(angle))
        self.y = int(y + distance * math.sin(angle))
        self.z = int(z + random.uniform(MIN_HEIGHT, MAX_HEIGHT))

        # init task
        self.cpu = float(UAV_CPU_CYCLE)
        self.cpu_cycle = random.uniform(UAV_CPU_CYCLE * 0.5, UAV_CPU_CYCLE)  # not used
        self.init()

    def init(self, cpu_cycle=TASK_CPU_CYCLE, data_size=TASK_DATA_SIZE, delay=TASK_DELAY, budget=BUDGET):
        # 매 시간마다 수행
        # self.task = [round(task_cpu_cycle * random.random(), 2), round(task_data_size * random.random(), 2), round(task_delay * random.random(), 2)]
        # self.task_original = {'cpu_cycle': cpu_cycle, 'data_size': data_size, 'delay': delay}
        self.task_original = {'cpu_cycle': cpu_cycle, 'data_size': data_size, 'delay': random.uniform(delay, delay*2)}
        self.T_LOCAL = self.task_original['cpu_cycle'] / self.cpu
        self.E_LOCAL = self.T_LOCAL * UAV_COMPUTING_UNIT_ENERGY * self.cpu ** 3
        # self.task = {'cpu_cycle': cpu_cycle, 'data_size': data_size, 'delay': delay}
        self.task = {'cpu_cycle': cpu_cycle, 'data_size': data_size, 'delay': random.uniform(delay, delay*2)}
        self.budget = budget
        self.T_local = 0
        self.E_local = 0
        self.T_transmission = 0
        self.T_offload = 0
        self.E_transmission = 0
        self.E_offload = 0
        self.overhead = 0
        self.utility = 0
        self.bus_num = 0
        self.buy_cpu = 0  # 구매한 cpu 양 초기화
        self.bus_id_list = []  # 구매할 버스 리스트 초기화
        self.purchase_bus_id_list = []  # 구매할 버스 리스트 초기화
        self.time_consume = 0
        # self.buy_price = []

        # for scheme
        self.closest = []
        self.preference_list = []

    def reset(self):
        self.overhead_list = []
        self.utility_list = []
        self.bus_num_list = []
        self.price_list = []

    def result_update(self, sigma):
        self.T_local = self.task['cpu_cycle'] / self.cpu
        self.E_local = self.T_local * UAV_COMPUTING_UNIT_ENERGY * self.cpu ** 3

        #self.overhead = ALPHA * ((self.T_local + self.T_transmission + self.T_offload) / self.T_LOCAL) + (1 - ALPHA) * (self.E_transmission + self.E_local) / self.E_LOCAL
        self.overhead = sigma * ((self.T_local + self.T_transmission + self.T_offload) / self.T_LOCAL) + (1 - sigma) * (self.E_transmission + self.E_local) / self.E_LOCAL

        self.overhead_list.append(self.overhead)
        self.utility_list.append(self.utility)
        self.bus_num_list.append(self.bus_num)

    def price_result_update(self, buses: list[Bus]):
        s = 0
        for bid in self.bus_id_list:
            s += buses[bid].price
        if s > 0:
            self.price_list.append(s / len(self.bus_id_list))

    def add_bus_id(self, bus_id):
        self.bus_id_list.append(bus_id)

    def purchase_cpu(self, bus: Bus, transmission_rate, price_sum, price_num, is_compare=True):
        # buy cpu from bus
        max_cpu = float(min(((self.budget + price_sum) / (bus.price * price_num)) - 1, bus.cpu, self.budget / bus.price,
                            self.task['cpu_cycle']))  # Game theory cpu calculation

        ratio = max_cpu / self.task_original['cpu_cycle']
        T_trans = ratio * self.task_original['data_size'] / transmission_rate
        T_off = max_cpu / bus.cpu
        E_trans = T_trans * UAV_TRANSMISSION_UNIT_ENERGY
        E_off = T_off * BUS_COMPUTING_UNIT_ENERGY * BUS_CPU_CYCLE ** 3
        if is_compare and max_cpu / self.cpu < (T_trans + T_off):
            return False

        if max_cpu > 0:
            temp_t = self.time_consume + T_trans + T_off
            if temp_t <= self.task['delay']:
                cost = bus.sell_cpu(max_cpu, self.id)
                # self.buy_price.append(bus.price)
                self.bus_id_list.remove(bus.id)

                self.time_consume = self.time_consume + temp_t
                self.buy_cpu += max_cpu
                self.budget = round(self.budget - cost, 2)
                self.purchase_bus_id_list.append(bus.id)
                self.utility += math.log10(1 + max_cpu)  # alpha = 1, beta = 1
                self.bus_num += 1

                # task
                self.task['cpu_cycle'] = self.task['cpu_cycle'] - max_cpu
                self.task['data_size'] = self.task['data_size'] - self.task_original['data_size'] * ratio

                # time & energy
                self.T_transmission = self.T_transmission + T_trans
                self.T_offload = self.T_offload + T_off
                self.E_transmission = self.E_transmission + E_trans
                self.E_offload = self.E_offload + E_off
                return True
        return False

    def matching_bus(self, buses: list[Bus], transmission_rate_list):  # not used
        # minimum overhead matching
        self.overhead = 1e9
        buy_id = -1

        # find minimum overhead
        for bus in buses:
            if bus.cpu < self.task_original['cpu_cycle']:
                continue
            T_trans = self.task_original['data_size'] / transmission_rate_list[self.id][bus.id]
            T_off = self.task_original['cpu_cycle'] / bus.cpu
            E_trans = T_trans * UAV_TRANSMISSION_UNIT_ENERGY
            E_off = T_off * BUS_COMPUTING_UNIT_ENERGY * BUS_CPU_CYCLE ** 3
            tmp_overhead = ALPHA * ((T_trans + T_off) / self.T_LOCAL) + (1 - ALPHA) * (E_trans) / self.E_LOCAL
            if tmp_overhead < self.overhead:
                buy_id = bus.id

        if buy_id == -1:
            print("ERROR cannot match bus !!!")
            return

        self.budget = 1e9
        if not self.purchase_cpu(buses[buy_id], transmission_rate_list[self.id][buy_id], 1e9, 1):
            print("ERROR something is wrong in calculating max_cpu !!!")

    def __str__(self):
        return f"UAV at ({self.x}, {self.y}, {self.z})"