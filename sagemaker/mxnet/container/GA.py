import numpy
from Arena import Arena
from MCTS import MCTS

from collections import deque
import time, os, sys
from pickle import Pickler, Unpickler
from random import shuffle, randint
import random
import copy
from mxnet import nd

def cal_pop_fitness(pop, args):
    # Calculating the fitness value of each solution in the current population.
    # The fitness function caulcuates the sum of products between each input and its corresponding weight.

    fitness = nd.zeros((len(pop),1))
    print('Prove your value Padawans!')
    for idx, pnnet in enumerate(pop):
        for jdx, nnnet in enumerate(pop):

            if idx == jdx:
                continue

            pmcts = MCTS(pnnet.game, pnnet, args)
            nmcts = MCTS(nnnet.game, nnnet, args)

            arena = Arena(lambda x: nd.argmax(pmcts.getActionProb(x, temp=0), axis=0),
                          lambda x: nd.argmax(nmcts.getActionProb(x, temp=0), axis=0), pnnet.game)
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
        parents[parent_num] = pop[max_fitness_idx]
        fitness[max_fitness_idx] = -99999999999
        indices.append(max_fitness_idx)

    return parents, indices

def cross(male, female, nn, args):
    pdict_male = male.nnet.model.collect_params()
    pdict_female = female.nnet.model.collect_params()

    baby = nn(female.game, args)
    pdict_baby = baby.nnet.model.collect_params().as_in_context(baby.ctx)

    for m,f,b in zip(pdict_male.keys(),pdict_female.keys(),pdict_baby.keys()):
        male_data = pdict_male[m].data().as_in_context(male.ctx)
        female_data = pdict_female[f].data().as_in_context(female.ctx)
        onesm = nd.ones(shape=male_data.shape, ctx=male.ctx)
        alpha = nd.random.uniform(shape=female_data.shape, ctx=female.ctx)
        pdict_baby[b].set_data((onesm-alpha)*male_data + alpha*female_data)

    print('Mating Crossover. Baby:',baby.name,'= Male:',male.name,'& Female:',female.name)
    return baby

def crossover(parents, offspring_size, nn, args):
    babies = [0]*offspring_size

    mating_arena = list(range(offspring_size))
    shuffle(mating_arena)

    for k in mating_arena:
        # Index of the parents to mate.
        male_idx = k%len(parents)
        female_idx = (k+1)%len(parents)
        # Get Parents
        male = parents[male_idx]
        female = parents[female_idx]

        # The new offspring will take genes from both parents
        babies[k] = cross(male, female, nn, args)

    return babies


def mutate(baby, mutation_propability, multiplier=0.10):
    pdict_baby = baby.nnet.model.collect_params()

    for k in pdict_baby.keys():
        baby_data = pdict_baby[k].data().as_in_context(baby.ctx)

        neg = nd.array(numpy.random.binomial(1, mutation_propability, baby_data.shape), baby.ctx)
        pos = nd.array(numpy.random.binomial(1, mutation_propability, baby_data.shape), baby.ctx)

        pdict_baby[k].set_data(baby_data + (pos-neg)*baby_data*multiplier)
    return baby

def mutation(offspring_crossover, mutation_propability=0.05, multiplier=0.10):
    # Mutation genes in each offspring randomly.
    for idx in range(len(offspring_crossover)):
        if(randint(0, 1) == 0):
            print('Mutate Baby:',offspring_crossover[idx].name)
            mutate(offspring_crossover[idx], mutation_propability, multiplier)
    return offspring_crossover

def save(new_population, model_path):
    new_population[0].save_checkpoint(model_path, 'alpha.network')
    for iidx, nnet in enumerate(new_population[1:]):
        nnet.save_checkpoint(model_path, 'beta{}.network'.format(iidx))
