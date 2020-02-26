import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import random as rd
import math

class GeneticAlgorithm():
    def __init__(self, tmax = 200, popsize=100, cross_rate=0.3, mut_rate=0.02):
        self.tmax = tmax
        self.popsize = popsize
        self.cross_rate = cross_rate
        self.mut_rate = mut_rate
        self.fitness = np.zeros(popsize)
        self.pop = np.zeros((popsize,2))
        self.run = True
        self.gmin = []
        self.gmax = []
        self.gmed = []
        self.t = 0

        for i in range(len(self.pop)):
            self.pop[i][0] = rd.uniform(0,10)
            self.pop[i][1] = rd.uniform(0,10)

    def step(self):
        if(self.run):
            self.select()
            self.crossover()
            self.mutate()
            self.evaluate()
            self.gmin.append(min(self.fitness))
            self.gmed.append(sum(self.fitness)/len(self.fitness))
            self.gmax.append(max(self.fitness))
            self.t+=1
            if (self.t==self.tmax):
                self.run=False
                #self.report()

    def getpop(self):
        return self.pop

    def evaluate(self):
        # Evaluation function
        for i in range(len(self.pop)):
            self.fitness[i] = math.sqrt(self.pop[i][0])*math.sin(self.pop[i][0]) * math.sqrt(self.pop[i][1])*math.sin(self.pop[i][1])

    def crossover(self):
        # Crossover function
        newpop = np.zeros((self.popsize, 2))
        alpha = rd.uniform(0,1)

        for i in range(0,len(self.pop),2):
            if (rd.randint(0, 100) > self.cross_rate*100):
                newpop[i] = self.pop[i]
                newpop[i+1] = self.pop[i+1]

            else:
                newpop[i] = alpha*self.pop[i] + (1-alpha)*self.pop[i+1]
                newpop[i+1] = alpha*self.pop[i+1] + (1-alpha)*self.pop[i]

            newpop[i] = self.bound(newpop[i])
            newpop[i+1] = self.bound(newpop[i+1])
        
        self.pop = newpop

    def bound(self, person):
        # Bound function just to keep points between 0 and 10
        if person[0] < 0:
            person[0] = 0
        elif person[0] > 10:
            person[0] = 10

        if person[1] < 0:
            person[1] = 0
        elif person[1] > 10:
            person[1] = 10
        return person

    def mutate(self):
        # Mutation function
        for i in range(len(self.pop)):
            if rd.uniform(0,1) < self.mut_rate:
                self.pop[i][0] = self.pop[i][0] + rd.uniform(-3,3)
            
            if rd.uniform(0,1) < self.mut_rate:
                self.pop[i][1] = self.pop[i][1] + rd.uniform(-3,3)
            
            self.pop[i] = self.bound(self.pop[i])


    def select(self):
        # Fitness proportionate selection, also known as roulette wheel selection
        scale_roulette = []
        newpop = np.zeros((self.popsize, 2))
        min_apt = min(self.fitness)

        if(min_apt < 0):
            adjustment = min_apt*(-1)
        else:
            adjustment = min_apt

        scale_roulette.append(adjustment)

        for i in range(1, len(self.fitness)):
            scale_roulette.append(scale_roulette[i-1] + (self.fitness[i]+adjustment))

        for i in range(len(self.fitness)):
            j=0
            aux = rd.uniform(scale_roulette[0], scale_roulette[-1])
            while (aux > scale_roulette[j]) and (j<len(scale_roulette)-1):
                j+=1
            newpop[i] = self.pop[j]
        
        self.pop = newpop

    def report(self):
        # Final report of the Maximum, Average and Minimum fitness per generation
        x = [i for i in range(len(self.gmin))]
        y = [float(i) for i in self.gmin]
        
        fig, ax = plt.subplots()
        plt.rcParams.update({'figure.figsize':(10,7), 'figure.dpi':100})
        plt.plot(x, y)

        y = [float(i) for i in self.gmed]
        plt.plot(x, y)

        y = [float(i) for i in self.gmax]
        plt.plot(x, y)

        print("Generations: " + str(self.tmax))
        print("Population size: " + str(self.popsize))
        print("Crossover rate: " + str(self.cross_rate))
        print("Mutation Rate: " + str(self.mut_rate))
        plt.title("Maximum, Average, Minimum Fitness per Generation")
        plt.show()


if __name__ == '__main__':
    plt.ion()
    fig, ax = plt.subplots()
    plt.title('Elements in population') 
    x, y = [],[]
    sc = ax.scatter(x,y)
    plt.xlim(0,10)
    plt.ylim(0,10)
    plt.draw()

    plt.plot(7.97,7.97,'ro', label='Global Max')
    plt.annotate("Global Max", (7.97,7.97),textcoords="offset points", xytext=(0,10), ha='center')

    ga = GeneticAlgorithm(tmax = 200, popsize=200, cross_rate=0.3, mut_rate=0.05)
    while (ga.run):
        ga.step()
        x, y = [], []
        textvar = plt.text(1, 9, "Step: " + str(ga.t), bbox=dict(facecolor='red', alpha=0.5))
        for item in ga.getpop():
            x.append(item[0])
            y.append(item[1])

        sc.set_offsets(np.c_[x,y])
        fig.canvas.draw_idle()
        plt.pause(0.2)
        textvar.set_visible(False)


    if not ga.run:
        plt.ioff()
        ga.report()
    