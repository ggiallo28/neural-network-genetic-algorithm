from collections import deque
from Arena import Arena
from MCTS import MCTS
import numpy as np
import time, os, sys
from pickle import Pickler, Unpickler
from random import shuffle
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures as futures
import time


class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """
    def __init__(self, game, args):
        self.game = game
        self.args = args
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()

    def executeEpisode(self, mcts):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard,pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamples = []
        board = self.game.getInitBoard()
        self.curPlayer = 1
        episodeStep = 0

        while True:
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(board,self.curPlayer)
            temp = int(episodeStep < self.args.tempThreshold)

            pi = mcts.getActionProb(canonicalBoard, temp=temp)
            sym = self.game.getSymmetries(canonicalBoard, pi)

            for b,p in sym:
                trainExamples.append([b, self.curPlayer, p, None])

            action = np.random.choice(len(pi), p=pi)
            board, self.curPlayer = self.game.getNextState(board, self.curPlayer, action)

            r = self.game.getGameEnded(board, self.curPlayer)

            if r!=0:
                return [(x[0],x[2],r*((-1)**(x[1]!=self.curPlayer))) for x in trainExamples]

    def generate(self, nnet):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximium length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        end = time.time()
        for i in range(0, self.args.numIters+1):
            if not self.skipFirstSelfPlay or i>0:
                self.iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

                futurelist = []
                executor = ThreadPoolExecutor(max_workers=multiprocessing.cpu_count())
                for eps in range(self.args.numEps):
                    mcts = MCTS(self.game, nnet, self.args)
                    #self.iterationTrainExamples += self.executeEpisode(mcts)
                    futurelist.append(executor.submit(self.executeEpisode, mcts))

                for eps in range(self.args.numEps):
                    self.iterationTrainExamples += futurelist[eps].result()

        total = round(time.time()-end,2)
        print('Generated {} Train Examples in {} Eps Time: {}s | Total: {}s'.format(len(self.iterationTrainExamples), self.args.numEps, round(total/self.args.numEps,3), total))

        # save the iteration examples to the history
        self.trainExamplesHistory.append(self.iterationTrainExamples)

        if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
            print("len(trainExamplesHistory) =", len(self.trainExamplesHistory), " => remove the oldest trainExamples")
            self.trainExamplesHistory.pop(0)

        trainExamples = []
        for e in self.trainExamplesHistory:
            trainExamples.extend(e)

        return trainExamples

    def train(self, new_population, trainExamples):
        for iidx, nnet in enumerate(new_population):
            print('------ Padawan {}: {} ------'.format(iidx, nnet.name))
            shuffle(trainExamples)
            nnet.train(trainExamples)

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, folder, filename):
        if not os.path.exists(folder):
            os.makedirs(folder)
        with open(folder+filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self, examplesFile):
        if not os.path.isfile(examplesFile):
            print(examplesFile)
            r = input("File with trainExamples not found. Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            print("File with trainExamples found. Read it.")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            f.closed
            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True
