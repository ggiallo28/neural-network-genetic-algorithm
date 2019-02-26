from Coach import Coach
from tictactoe.TicTacToeGame import TicTacToeGame as Game
from tictactoe.keras.NNet import NNetWrapper as nn
from utils import *

args = dotdict({
    'numIters': 1,
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

if __name__=="__main__":
    g = Game()
    nnet = nn(g)

    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    w = nnet.get_weights()
    for idx, wi in enumerate(w):
        w[idx] = wi
    nnet.set_weights(w)

    c = Coach(g, nnet, args)
    #if args.load_model:
    #    print("Load trainExamples from file")
    #    c.loadTrainExamples()

    c.learn()