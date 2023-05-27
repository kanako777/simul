from config import *


class Bus:
    def __init__(self, id, type, path):
        self.id = id
        self.type = type  # 0이면 버스, 1이면 RSU
        self.path = path
        self.MAX_CPU = BUS_CPU_CYCLE if self.type == 0 else RSU_CPU_CYCLE  # 총 cpu
        self.reset()
        self.init()

    def init(self):
        # 매 시간 마다 수행
        self.uav_id_list = []
        self.sell_uav_id_list = []
        self.cpu = BUS_CPU_CYCLE if self.type == 0 else RSU_CPU_CYCLE  # 남은 cpu
        self.price = 1  # 자신의 여유분 cpu에 대한 cost 부여
        self.old_price = 1
        self.utility = 0

        # for scheme
        self.closest = []
        self.preference_list = []

    def reset(self):
        # 매 시뮬레이션 마다 수행
        self.path_index = 0
        self.location = self.path[self.path_index]
        self.x = self.location[0]
        self.y = self.location[1]

        self.price_history = []
        self.utility_list = []
        self.price_list = []

    def move(self):
        if self.type == 0:
            self.path_index += 1
            if self.path_index == len(self.path):
                self.path_index = 0
            self.location = self.path[self.path_index]
            self.x = self.location[0]
            self.y = self.location[1]
            # print(f"Bus {self.id} moved to {self.location}")

    def add_uav_id(self, uav_id):
        self.uav_id_list.append(uav_id)

    def change_price(self, new_price):
        self.old_price = self.price
        self.price = new_price
        self.price_history.append(self.price)

    def sell_cpu(self, cpu_amount, uav_id):
        self.uav_id_list.remove(uav_id)
        self.sell_uav_id_list.append(uav_id)
        self.cpu = round(self.cpu - cpu_amount, 0)
        return cpu_amount * self.price

    def result_update(self):
        # GU utility update
        self.utility_list.append(self.price * (self.MAX_CPU - self.cpu))
        self.price_list.append(self.price)