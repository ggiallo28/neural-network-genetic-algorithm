from Coach import Coach
from tictactoe.TicTacToeGame import TicTacToeGame as Game
from tictactoe.keras.NNet import NNetWrapper as nn
from othello.OthelloGame import OthelloGame as Game
from othello.keras.NNet import NNetWrapper as nn
#from othello.tensorflow.NNet import NNetWrapper as nn # ResNet
#from othello.pytorch.NNet import NNetWrapper as nn
#from othello.chainer.NNet import NNetWrapper as nn
from utils import *
import numpy
import GA

args = dotdict({
    'numIters': 5,
    'numEps': 10,
    'tempThreshold': 15,
    'updateThreshold': 0.6,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 15,
    'arenaCompare': 10,
    'cpuct': 1,

    'model_path': './pretrained_models/othello/',
    'load_model': False,
    'load_examples': False,
    'numItersForTrainExamplesHistory': 20,
})

"""
Genetic algorithm parameters:
    Mating pool size
    Population size
"""
sol_per_pop = 3
num_parents = 2

# Creating the environment.
game = Game()
master = Coach(game, args)

# Creating the initial population.
new_population = numpy.array([nn(game) for i in range(0,sol_per_pop)])

if args['load_model']:
    for i,p in enumerate(new_population):
        p.load_checkpoint(args['model_path'], 'padawan{}.network'.format(i))
if args['load_examples']:
    master.loadTrainExamples('{}/train.examples'.format(args['model_path']))

ancestors = list(range(0,sol_per_pop))
alpha_index = 0
num_generations = 100

for generation in range(num_generations):
    # Generate Train Examples by using alpha chromosome
    alpha_padawan = new_population[alpha_index] # Select Alpha Padawan, usually is the first in list.
    train_examples = master.generate(alpha_padawan)

    # Training each chromosome in the population osing Train Examples.
    master.train(new_population, train_examples)

    # Measing the fitness of each chromosome in the population.
    fitness = GA.cal_pop_fitness(new_population, game, args)

    # Selecting the best parents in the population for mating.
    parents, indices = GA.select_mating_pool(new_population, fitness, num_parents)

    # Generating next generation using crossover.
    offspring_crossover = GA.crossover(parents, sol_per_pop-num_parents, nn)

    # Adding some variations to the offsrping using mutation.
    offspring_mutation = GA.mutation(offspring_crossover)

    # Creating the new population based on the parents and offspring.
    new_population[0:num_parents] = parents
    new_population[num_parents:] = offspring_mutation

    # Save Current State
    print("Save Checkpoint")
    for i,p in enumerate(new_population):
        p.save_checkpoint(args['model_path'], 'padawan{}.network'.format(i))
    master.saveTrainExamples(args['model_path'], 'train.examples')

    # Print current results
    lossness = [p.get_loss() for p in parents]
    ancestors = [indices.index(i) for i in set(ancestors).intersection(indices)]
    print("Generation : {} | Best result : {} | Ancestors {}".format(generation, min(lossness)[0], ancestors) )
    print('>Alpha Padawan ', parents[0].name)
    for m in parents[1:]:
        print('Senior Padawan ', m.name)
    for m in offspring_mutation:
        print('Junior Padawan ', m.name)

    if (len(ancestors) == 0):
        print("All ancestors are dead, generating..")
        ancestors = list(range(0,num_parents))
    alpha_index = 0

# Getting the best solution after iterating finishing all generations.
# At first, the fitness is calculated for each solution in the final generation.
fitness = GA.cal_pop_fitness(new_population, args)
# Then return the index of that solution corresponding to the best fitness.
best_match_idx = numpy.where(fitness == numpy.max(fitness))

print("Best solution : ", new_population[best_match_idx])
print("Best solution fitness : ", fitness[best_match_idx])
