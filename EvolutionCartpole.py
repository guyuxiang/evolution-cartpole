import numpy as np
import random 
import gym
import matplotlib.pyplot as plt

class Agent:
    def __init__(self):
    	# 生成一个个体，包含由两个四维向量组成的随机矩阵，分别代表进化策略ES的两种DNA
    	# 一条是用实数表达的基因DNA，另一条与基因等长的、决定对应位置基因变异率的变异率DNA。
    	# 每个基因位点的变异率随着该基因对适应的贡献程度的增加而趋于0，相当于帮助种群在最优解处收敛
        self.gene = np.random.rand(2, 4) * 10	
        self.fitness = 0	# 初始化个体适应度为0

    def __mul__(self, other):		# 重载乘法运算*使其成为两个个体产生子代的运算
        kid = Agent()				# 初始化子代
        gene_T = kid.gene.T 		# 将子代基因转置便于下面的操作
        tup_DNA = (self.gene[0], other.gene[0])
        DNA_P = np.vstack(tup_DNA)	# 将self和other的基因DNA合并
        tup_MUT = (self.gene[1], other.gene[1])
        MUT_P = np.vstack(tup_MUT)	# 将self和other的变异率DNA合并
        for col in range(4):	
            rnd = random.randint(0, 1)
            gene_T[col] = np.array([DNA_P.T[col][rnd], MUT_P.T[col][rnd]])	# 随机遗传的过程
            # 随机变异的过程,对应基因以符合均值为0的正态分布为幅度变异时，该正态分布的标准差σ。
            gene_T[col][0] += np.random.randn() * gene_T[col][1]	# 对生成的新基因和变异率加以符合正态分布N(0，σ^2 )的变异，其中σ为对应基因的变异率
            gene_T[col][1] += np.random.randn() * gene_T[col][1]	# 对生成的新基因和变异率加以符合正态分布N(0，σ^2 )的变异，其中σ为对应基因的变异率
        kid.gene = gene_T.T
        return kid	#返回子代

    def action(self, observation):	# 根据观测值决定行动
    	# 因为是四个实数的DNA，根据这四个数与观测值的四个值对应相乘的和
        res = np.sum(self.gene[0] * observation)
        # 选择小车的两种操作，若大于0则执行操作1，否则执行操作0。	
        if res > 0:
            return 1
        if res < 0:
            return 0
        return 0

def get_fittness(agents):	    # 在Gym中训练并获得个体适应度
    fitness = []				# 初始化一个种群适应度数组
    for agent in agents:
        agent.fitness = 0		# 对每个个体重新初始化适应度

    for agent in agents:
        observation = env.reset()	# 初始化一个事件（Eposide）
        # 对于种群中的每一个个体,进行游戏的50000次测试训练
        for _ in range(1000):
        	# 执行行动获取结果
            observation, reward, done, info = env.step(agent.action(observation))
            if done:	# 如果事件结束则退出循环
                break
            agent.fitness += reward	    # 根据获得的奖励reward调整个体适应度
        fitness.append(agent.fitness)	# 将得到的个体的新的适应度添加到数组中
    return np.array(fitness)			# 返回种群适应度数组

def choose_agent(agents):	
    fitness = get_fittness(agents)			# 获取每个个体的适应度
    fitness_sort = np.argsort(-fitness) 	# 将是适应度从大到小排序
    new_agents = []							# 初始化新种群
    for i in range(keep):					# 进行自然选择,根据Keep保留Fitness排在前面的一部分个体
        new_agents.append(agents[fitness_sort[i]])			# 选取好的个体加入新种群，淘汰不好的个体
    return new_agents, fitness, agents[fitness_sort[0]]		# 返回了自然选择后的种群,适应度,表现最好的个体

def make_agent(agents):	 # 从选择好的个体中随机选出两个产生新的个体，直到种群再次达到规定数目
    for _ in range(population - len(agents)):
        agents.append(agents[random.randint(0, keep - 1)]
                      * agents[random.randint(0, keep - 1)])
    return agents 	# 返回自然选择后,适应高的个体进行繁衍,形成的新的种群

env = gym.make('CartPole-v0')	# 初始化训练环境

keep = 60	# 自然选择中保留个体的数量	 
population = 100	# 种群的总数
agents = [Agent() for _ in range(population)]	# 生成初始种群

# 创建用于保存绘图数据的列表
fitness_avg = []	 # 保持每一代种群平均适应度的列表
fitness_max = []	 # 保持每一代种群最高适应度的列表

# 训练主循环,进行50代进化
for i in range(51):
    agents = make_agent(agents)                  # 生成种群
    agents, fitness, best = choose_agent(agents) # 自然选择后的种群,种群适应度,适应度最高的个体
    fitness_avg.append(np.mean(fitness))         # 保存这一代平均适应度数据
    fitness_max.append(np.max(fitness))			 # 保存这一代最高适应度数据
    print("第%d次进化完成! Average fitness: %d Maximum fitness: %d" % (i, np.mean(fitness), np.max(fitness)))

observation = env.reset()
for _ in range(1000):
    env.render()    # 刷新当前环境，并显示
    # 适应度最好的个体对observation做出动作,获得的结果
    observation, reward, done, info = env.step(best.action(observation)) 
    if done:
       break

print("进化结束,展示适应度变化曲线")
print("-------------------------------------------------------")
plt.figure()
plt.plot(fitness_avg)	# 绘制各代种群平均适应度变化曲线
plt.plot(fitness_max)	# 绘制各代种群最高个体适应度变化曲线
plt.show()
