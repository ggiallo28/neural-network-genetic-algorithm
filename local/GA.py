from Arena import Arena
from MCTS import MCTS
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle
from random import randint
import numpy as np
import time
import os
import sys
import numpy
import random
import copy

def cal_pop_fitness(pop, game, args):
    # Calculating the fitness value of each solution in the current population.
    # The fitness function caulcuates the sum of products between each input and its corresponding weight.
    fitness = numpy.zeros(pop.shape)
    print('FIGHT!')
    for idx, pnnet in enumerate(pop):
        for jdx, nnnet in enumerate(pop):

            if idx == jdx:
                continue

            pmcts = MCTS(game, pnnet, args)
            nmcts = MCTS(game, nnnet, args)

            arena = Arena(lambda x: numpy.argmax(pmcts.getActionProb(x, temp=0)),
                          lambda x: numpy.argmax(nmcts.getActionProb(x, temp=0)), game)
            pwins, nwins, draws, total = arena.playGames(args.arenaCompare)

            print('P{}/P{} WINS : {} / {} ; DRAWS : {} | Time {}s'.format(idx, jdx, pwins, nwins, draws, total))

            fitness[idx] += ((pwins > nwins) + (pwins==nwins==0)*0.01 + (pwins==nwins!=0)*0.0001 + pwins*0.0000001)
            fitness[jdx] += ((nwins > pwins) + (pwins==nwins==0)*0.01 + (pwins==nwins!=0)*0.0001 + nwins*0.0000001)

    return fitness

def select_mating_pool(pop, fitness, num_parents):
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
    parents = [0]*num_parents
    indices = []
    for parent_num in range(num_parents):
        max_fitness_idx = numpy.where(fitness == numpy.max(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        print('Padawan {} fitness: {}'.format(max_fitness_idx, round(fitness[max_fitness_idx],7) ))
        parents[parent_num] = pop[max_fitness_idx]
        fitness[max_fitness_idx] = -99999999999
        indices.append(max_fitness_idx)

    nnet_parents = numpy.array(parents)
    return nnet_parents, indices

def crossover(parents, offspring_size, nn):
    babies = [0]*offspring_size
    game = parents[0].game

    mating_arena = list(range(offspring_size))
    shuffle(mating_arena)

    for k in mating_arena:
        # Index of the parents to mate.
        male_idx = k%len(parents)
        female_idx = (k+1)%len(parents)
        # Get Parents
        male = parents[male_idx].get_weights()
        female = parents[female_idx].get_weights()
        # Create crossover
        onesm = np.ones(male.shape)
        alpha = np.random.uniform(0,1,female.shape)
        # The new offspring will take genes from both parents
        babies[k] = nn(game).set_weights((onesm-alpha)*male + alpha*female)
        print('Mating Crossover. Baby:',babies[k].name,'= Male:',parents[male_idx].name,'& Female:',parents[female_idx].name)

    return babies

def mutation(offspring_crossover, mutation_propability=0.05, multiplier=0.10):
    # Mutation genes in each offspring randomly.
    for idx in range(len(offspring_crossover)):
        if(randint(0, 1) == 0):
            print('Mutate Baby:',offspring_crossover[idx].name)
            baby_data = offspring_crossover[idx].get_weights()
            neg = numpy.random.binomial(1, mutation_propability, baby_data.shape)
            pos = numpy.random.binomial(1, mutation_propability, baby_data.shape)
            baby_data += (pos-neg)*baby_data*multiplier
            offspring_crossover[idx].set_weights(baby_data)
    return offspring_crossover
