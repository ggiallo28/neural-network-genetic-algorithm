from collections import deque
from Arena import Arena
from MCTS import MCTS
import numpy as np
from pytorch_classification.utils import Bar, AverageMeter
import time, os, sys
from pickle import Pickler, Unpickler
from random import shuffle
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures as futures
from utils import *
from othello.OthelloGame import OthelloGame as Game
from othello.keras.NNet import NNetWrapper as nn
#from tictactoe.TicTacToeGame import TicTacToeGame as Game
#from tictactoe.keras.NNet import NNetWrapper as nn
import multiprocessing
import copy

args = dotdict({
    'numIters': 1,
    'numEps': 100,
    'tempThreshold': 25,
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

def executeEpisode(game, mcts, args):
    trainExamples = [[]]*500
    board = game.getInitBoard()
    curPlayer = 1
    episodeStep = 0
    #tt = []

    while True:
        episodeStep += 1
        #end = time.time()
        canonicalBoard = game.getCanonicalForm(board, curPlayer)
        temp = int(episodeStep < args.tempThreshold)
        #print('1 ', time.time() - end)

        #end = time.time()
        pi = mcts.getActionProb(canonicalBoard, temp=temp)
        #print('a', time.time() - end)

        sym = game.getSymmetries(canonicalBoard, pi)

        #end = time.time()
        for b,p in sym:
            trainExamples[episodeStep-1] = [b, curPlayer, p, None]
            #trainExamples.append([b, curPlayer, p, None])
        #print('3 ', time.time() - end)

        #end = time.time()
        action = np.random.choice(len(pi), p=pi)
        board, curPlayer = game.getNextState(board, curPlayer, action)
        #print('4 ', time.time() - end)

        r = game.getGameEnded(board, curPlayer)

        if r!=0:
            #print('5 ', len(trainExamples))
            #print(sum(tt)/len(tt))
            trainExamples = trainExamples[:episodeStep]
            return [(x[0],x[2],r*((-1)**(x[1]!=curPlayer))) for x in trainExamples]


def procedure(game, mcts, args):
    return executeEpisode(game, mcts, args)

game = Game()
nnet = nn(game)
eps_time = AverageMeter()
bar = Bar('Self Play', max=args.numEps)
end = time.time()

futurelist = []
executor = ThreadPoolExecutor(max_workers=multiprocessing.cpu_count())
for eps in range(args.numEps):
    mcts = MCTS(game, nnet, args)
    futurelist.append(executor.submit(procedure, game, mcts, args))

iterationTrainExamples = deque([], maxlen=args.maxlenOfQueue)
for eps in range(args.numEps):
    iterationTrainExamples += futurelist[eps].result()
    eps_time.update(time.time() - end)
    end = time.time()
    bar.suffix  = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(eps=eps+1, maxeps=args.numEps, et=eps_time.avg,total=bar.elapsed_td, eta=bar.eta_td)
    bar.next()
bar.finish()
executor.shutdown()

eps_time = AverageMeter()
bar = Bar('Self Play', max=args.numEps)
end = time.time()
iterationTrainExamples = deque([], maxlen=args.maxlenOfQueue)

for eps in range(args.numEps):
    mcts = MCTS(game, nnet, args)
    iterationTrainExamples += executeEpisode(game, mcts, args)

    ## bookkeeping + plot progress
    eps_time.update(time.time() - end)
    end = time.time()
    bar.suffix  = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(eps=eps+1, maxeps=args.numEps, et=eps_time.avg, total=bar.elapsed_td, eta=bar.eta_td)
    bar.next()
bar.finish()

a = [0]*1000000
end = time.time()
for i in range(len(a)):
    a[i] = i
print(time.time()-end)

