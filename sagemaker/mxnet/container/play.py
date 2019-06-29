import Arena
from MCTS import MCTS

GAME = 'othello'

if GAME == 'tictactoe':
    from games.tictactoe.TicTacToeGame import TicTacToeGame as Game
    from games.tictactoe.TicTacToeGame import display
    from games.tictactoe.TicTacToePlayers import RandomPlayer
    from games.tictactoe.TicTacToePlayers import HumanTicTacToePlayer as HumanPlayer
    from games.tictactoe.mxnet.NNet import NNetWrapper as NNet
    g = Game(3)

if GAME == 'othello':
    from games.othello.OthelloGame import OthelloGame as Game
    from games.othello.OthelloGame import display
    from games.othello.OthelloPlayers import RandomPlayer
    from games.othello.OthelloPlayers import HumanOthelloPlayer as HumanPlayer
    from games.othello.mxnet.NNet import NNetWrapper as NNet
    g = Game(8)

from mxnet import nd
from utils import *

MODEL_FOLDER = '/opt/ml/model'
MODEL_NAME = 'alpha.network'
"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

# all players
rp = RandomPlayer(g).play
#gp = GreedyPlayer(g).play
hp = HumanPlayer(g).play

# nnet players
n1 = NNet(g)
n1.load_checkpoint(MODEL_FOLDER,MODEL_NAME)
args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
mcts1 = MCTS(g, n1, args1)

n1p = lambda x: nd.argmax(mcts1.getActionProb(x, temp=0), axis=0)


#n2 = NNet(g)
#n2.load_checkpoint('/dev/8x50x25/','best.pth.tar')
#args2 = dotdict({'numMCTSSims': 25, 'cpuct':1.0})
#mcts2 = MCTS(g, n2, args2)
#n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))

arena = Arena.Arena(n1p, hp, g, display=display)
print(arena.playGames(2, verbose=True))
