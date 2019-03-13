import numpy
from Arena import Arena
from MCTS import MCTS

from collections import deque
import time, os, sys
from pickle import Pickler, Unpickler
from random import shuffle, randint
import random
import copy

def cal_pop_fitness(pop, args):
    # Calculating the fitness value of each solution in the current population.
    # The fitness function caulcuates the sum of products between each input and its corresponding weight.

    fitness = numpy.zeros((len(pop),1))
    print('Prove your value Padawans!')
    for idx, pnnet in enumerate(pop):
        for jdx, nnnet in enumerate(pop):

            if idx == jdx:
                continue

            pmcts = MCTS(pnnet.game, pnnet, args)
            nmcts = MCTS(nnnet.game, nnnet, args)

            arena = Arena(lambda x: numpy.argmax(pmcts.getActionProb(x, temp=0)),
                          lambda x: numpy.argmax(nmcts.getActionProb(x, temp=0)), pnnet.game)
            pwins, nwins, draws, total = arena.playGames(args.arenaCompare)

            print('P{}/P{} WINS : {} / {} ; DRAWS : {} | Time {}s'.format(idx, jdx, pwins, nwins, draws, total))

            fitness[idx] += ((pwins > nwins) + (pwins==nwins==0)*0.01 + (pwins==nwins!=0)*0.0001 + pwins*0.0000001)
            fitness[jdx] += ((nwins > pwins) + (pwins==nwins==0)*0.01 + (pwins==nwins!=0)*0.0001 + nwins*0.0000001)

    return fitness

def select_mating_pool(pop, fitness, num_parents):
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
    # parents = numpy.empty((num_parents,))
    parents = [0]*num_parents
    indices = []
    for parent_num in range(num_parents):
        max_fitness_idx = numpy.where(fitness == numpy.max(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        print('Padawan {} fitness: {}'.format(max_fitness_idx, round(fitness[max_fitness_idx][0],7) ))
        parents[parent_num] = pop[max_fitness_idx]
        fitness[max_fitness_idx] = -99999999999
        indices.append(max_fitness_idx)

    nnet_parents = numpy.array(parents)
    parents_weights = [[nnet.get_weights(), nnet.name] for nnet in nnet_parents]
    return nnet_parents, parents_weights, indices

#def half_crossover(parents, offspring_size):
#    offspring = [[]]*offspring_size[0]
#    # The point at which crossover takes place between two parents. Usually it is at the center.
#    crossover_point = numpy.uint8(offspring_size[1]/2)
#
#    mating_arena = list(range(offspring_size[0]))
#    shuffle(mating_arena)
#    for k in mating_arena:
#        # Index of the first parent to mate.
#        male_idx = k%len(parents)
#        # Index of the second parent to mate.
#        female_idx = (k+1)%len(parents)
#        # The new offspring will have its first half of its genes taken from the first parent.
#        # The new offspring will have its second half of its genes taken from the second parent.
#        mf = randint(0, 1)
#        if mf == 0:
#            male = parents[male_idx][0:crossover_point]
#            female = parents[female_idx][crossover_point:]
#            offspring[k] = male + female
#        if mf == 1:
#            female = parents[female_idx][0:crossover_point]
#            male = parents[male_idx][crossover_point:]
#            offspring[k] = female + male
#    return offspring

def cross(male, female, baby, alpha):
    for idx, val in enumerate(male):
        if isinstance(val, list):
            cross(male[idx], female[idx], baby[idx], alpha)
        else:
            baby[idx] = male[idx]*(1-alpha) + female[idx]*(alpha)
    return baby

def crossover(parents, offspring_size):
    offspring = [[]]*offspring_size

    mating_arena = list(range(offspring_size))
    shuffle(mating_arena)
    for k in mating_arena:
        # Index of the first parent to mate.
        male_idx = k%len(parents)
        # Index of the second parent to mate.
        female_idx = (k+1)%len(parents)
        # Get Parents
        male = parents[male_idx][0]
        female = parents[female_idx][0]
        # The new offspring will have its first half of its genes taken from the first parent and its second half of its genes taken from the second parent.
        alpha = numpy.random.uniform(0,1)
        baby = copy.deepcopy(female)
        cross(male, female, baby, alpha)
        offspring[k] = baby
        print('Mating Crossover Baby = '+str(round(1-alpha,3))+' Male(',parents[male_idx][1],') + '+str(round(alpha,3))+' Female(',parents[female_idx][1],')')
    return offspring


def mutate(offspring, mutation_propability, multiplier=0.10):
    for idx, val in enumerate(offspring):
        if isinstance(val, list):
            mutate(val)
        else:
            sign = random.choice([-1, 1])
            prob = numpy.random.uniform(0,1)
            if prob <= mutation_propability:
                offspring[idx] = val + sign*val*multiplier

def mutation(offspring_crossover, mutation_propability=0.05, multiplier=0.10):
    # Mutation genes in each offspring randomly.
    for idx in range(len(offspring_crossover)):
        if(randint(0, 1) == 0):
            mutate(offspring_crossover[idx][0], mutation_propability, multiplier)
    return offspring_crossover
