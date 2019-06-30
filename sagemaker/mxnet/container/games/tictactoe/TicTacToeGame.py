from __future__ import print_function
import sys
sys.path.append('..')
from Game import Game
from .TicTacToeLogic import Board
from mxnet import nd
import mxnet as mx

"""
Game class implementation for the game of TicTacToe.
Based on the OthelloGame then getGameEnded() was adapted to new rules.

Author: Evgeny Tyurin, github.com/evg-tyurin
Date: Jan 5, 2018.

Based on the OthelloGame by Surag Nair.
"""
class TicTacToeGame(Game):
    def __init__(self, n=3, cuda=True):
        self.n = n
        self.cuda = cuda
        self.ctx = mx.gpu() if cuda else mx.cpu()

    def getInitBoard(self):
        # return initial board (numpy board)
        b = Board(self.n, self.cuda)
        return b.pieces

    def getBoardSize(self):
        # (a,b) tuple
        return (self.n, self.n)

    def getActionSize(self):
        # return number of actions
        return self.n*self.n + 1

    def getNextState(self, board, player, action):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        if action == self.n*self.n:
            return (board, -player)
        b = Board(self.n, self.cuda)
        b.pieces = board.copy()
        move = (int(action/self.n), action%self.n)
        b.execute_move(move, player)
        return (b.pieces, -player)

    def getValidMoves(self, board, player):
        # return a fixed size binary vector
        valids = nd.zeros((self.getActionSize(),), ctx=self.ctx)
        b = Board(self.n, self.cuda)
        b.pieces = board.copy()
        legalMoves =  b.get_legal_moves(player)
        if len(legalMoves)==0:
            valids[-1]=1
            return valids
        for x, y in legalMoves:
            valids[self.n*x+y]=1
        return valids

    def getGameEnded(self, board, player):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1
        b = Board(self.n, self.cuda)
        b.pieces = board.copy()

        if b.is_win(player):
            return 1
        if b.is_win(-player):
            return -1
        if b.has_legal_moves():
            return 0
        # draw has a very little value
        return 1e-4

    def getCanonicalForm(self, board, player):
        # return state if player==1, else return -state if player==-1
        return player*board

    def flipud(self, x):
        return nd.flip(data=x, axis=0)

    def fliplr(self, x):
        return nd.flip(data=x, axis=1)

    def rot90(self, x, k):
        k = k%4
        if k == 0:
            return x
        if k == 1:
            y = self.flipud(nd.transpose(x,axes=(0,1,2)))
        if k == 2:
            y = self.flipud(self.fliplr(x))
        if k == 3:
            y = nd.transpose(self.flipud(x),axes=(0,1,2))
        return y

    def getSymmetries(self, board, pi):
        # mirror, rotational
        assert(len(pi) == self.n**2+1)  # 1 for pass
        pi_board = nd.reshape(pi[:-1], (1, self.n, self.n))
        l = []

        for i in range(1, 5):
            for j in [True, False]:
                newB = self.rot90(board, i)
                newPi = self.rot90(pi_board, i)
                if j:
                    newB = self.fliplr(newB)
                    newPi = self.fliplr(newPi)
                l += [(newB, nd.concat(newPi.reshape((self.n*self.n)), pi[-1], dim=0))]
        return l


    def stringRepresentation(self, board):
        # 8x8 numpy array (canonical board)
        return board.reshape((self.n*self.n)).__repr__().split('\n')[1]

def display(board):
    n = board.shape[2]

    print("   ", end="")
    for y in range(n):
        print (y,"", end="")
    print("")
    print("  ", end="")
    for _ in range(n):
        print ("-", end="-")
    print("--")
    for y in range(n):
        print(y, "|",end="")    # print the row #
        for x in range(n):
            piece = board[0][0][y][x]    # get the piece to print
            if piece == -1: print("X ",end="")
            elif piece == 1: print("O ",end="")
            else:
                if x==n:
                    print("-",end="")
                else:
                    print("- ",end="")
        print("|")

    print("  ", end="")
    for _ in range(n):
        print ("-", end="-")
    print("--")
