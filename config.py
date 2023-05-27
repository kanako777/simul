NUM_BUS = 50  # 운행하는 버스의 대수(버스별로 자기 노선(path)을 가짐
NUM_RSU = 10  # 설치된 RSU의 개수
NUM_UAV = 20  # UAV의 개수
NUM_PATH = 200  # 버스 운행경로별로 지나는 정류장(지점)의 개수
NUM_PASSENGER = 1 # 사용안함
BUDGET = 50 # UAV의 최대 BUDGET (이 BUDGET을 이용하여 버스나 RSU로부터 CPU를 구매)

MAP_SIZE = 1000  # MAP크기

MIN_HEIGHT = 100 # UAV의 최저 고도
MAX_HEIGHT = 150 # UAV의 최고 고도
POI_RADIUS = 200 # POI로부터 UAV가 위치하는 최대 반경
RSU_DISTANCE = 50 # RSU가 설치될 때 서로간의 최소 이격거리

TIME_INTERVAL = 1
SIMUL_TIME = 200 # 시뮬레이션을 반복하는 횟수(t)

BUS_STEP = 10        # 버스의 숫자 변화량
NUM_BUS_STEP = 4   # 버스의 숫자를 변화시키는 횟수(X축)
UAV_STEP = 4        # UAV 숫자 변화량
NUM_UAV_STEP = 4    # UAV의 숫자를 변화시키는 횟수
BUDGET_STEP = 10    # budget 변화량
NUM_BUDGET_STEP = 4 # budget 변화 횟수

TASK_CPU_CYCLE = 20 # 단위 TASK 수행에 요구되는 CPU사이클
TASK_DATA_SIZE = 10 # 단위 TASK의 파일용량(MB)
TASK_DELAY= 1      # 단위 TASK 수행의 최대허용 딜레이(초)

BUS_CPU_CYCLE = 100	# 버스의 최대 CPU 사이클
RSU_CPU_CYCLE = 100 # RSU의 최대 CPU 사이클
UAV_CPU_CYCLE = 10 # UAV의 최대 CPU 사이클
SIGMA_SPEED = 1000 # 게임이론을 적용해서 버스가 자신의 CPU가격을 변화시킬때 변화값을 판별하기 위한 값

ALPHA = 0.5 #(OVERHEAD를 결정할 때, 딜레이와 에너지의 비율)

BANDWIDTH = 10E6
#BUS_ENERGY = 1000 #(사용안함)
#UAV_ENERGY = 1000 #(사용안함)
UAV_COMPUTING_UNIT_ENERGY = 1 # UAV가 LOCAL로 TASK 수행시 사용하는 단위 에너지
UAV_TRANSMISSION_UNIT_ENERGY = 0.25 # UAV가 BUS나 RSU에 데이터전송시 사용하는 단위 에너지
BUS_COMPUTING_UNIT_ENERGY = 0.25 # 버스가 TASK 수행시 사용하는 단위 에너지