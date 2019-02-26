from Coach import Coach
#from tictactoe.TicTacToeGame import TicTacToeGame as Game
#from tictactoe.keras.NNet import NNetWrapper as nn
from othello.OthelloGame import OthelloGame as Game
from othello.keras.NNet import NNetWrapper as nn
from utils import *
import numpy
import GA

args = dotdict({
    'numIters': 2,
    'numEps': 100,
    'tempThreshold': 15,
    'updateThreshold': 0.6,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 25,
    'arenaCompare': 40,
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
})

"""
Genetic algorithm parameters:
    Mating pool size
    Population size
"""
sol_per_pop = 10
num_parents = 5
#Creating the initial population.
new_population = []
game = Game()
for i in range(0,sol_per_pop):
    nnet = nn(game)
    new_population.append(nnet)
new_population = numpy.array(new_population)

num_generations = 100
for generation in range(num_generations):
    # Training each chromosome in the population.
    for iidx, nnet in enumerate(new_population):
        fab = Coach(Game(), nnet, args)
        fab.learn(iidx)

    # Measing the fitness of each chromosome in the population.
    fitness = GA.cal_pop_fitness(new_population, args)

    # Selecting the best parents in the population for mating.
    nnet_parents = GA.select_mating_pool(new_population, fitness, num_parents)
    parents = []
    for p in nnet_parents:
        parents.append(p.get_weights())

    # Generating next generation using crossover.
    num_weights = len(parents[0])
    offspring_crossover = GA.crossover(parents, offspring_size=(sol_per_pop-num_parents, num_weights))

    # Adding some variations to the offsrping using mutation.
    offspring_mutation = GA.mutation(offspring_crossover)
    nnet_offspring_mutation = []
    for m in offspring_mutation:
        nnet = nn(game)
        nnet.set_weights(m)
        nnet_offspring_mutation.append(nnet)

    # Creating the new population based on the parents and offspring.
    new_population[0:num_parents] = nnet_parents
    new_population[num_parents:] = nnet_offspring_mutation

    # The best result in the current iteration.
    lossness = []
    for p in nnet_parents:
        lossness.append(p.get_loss())
    print("Generation : ", generation, " Best result : ", min(lossness)[0])
    print(new_population)

# Getting the best solution after iterating finishing all generations.
#At first, the fitness is calculated for each solution in the final generation.
fitness = GA.cal_pop_fitness(new_population, args)
# Then return the index of that solution corresponding to the best fitness.
best_match_idx = numpy.where(fitness == numpy.max(fitness))

print("Best solution : ", new_population[best_match_idx])
print("Best solution fitness : ", fitness[best_match_idx])




# training new network, keeping a copy of the old one
# self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
# self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
# pmcts = MCTS(self.game, self.pnet, self.args)

#nmcts = MCTS(self.game, self.nnet, self.args)

#print('PITTING AGAINST PREVIOUS VERSION')
#arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
#              lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), self.game)
#pwins, nwins, draws = arena.playGames(self.args.arenaCompare)

#print('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
#if pwins+nwins == 0 or float(nwins)/(pwins+nwins) < self.args.updateThreshold:
#    print('REJECTING NEW MODEL')
#    self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
#else:
#    print('ACCEPTING NEW MODEL')
#    self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
#    self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')

