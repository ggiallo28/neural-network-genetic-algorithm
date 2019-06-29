from __future__ import print_function
import sys
sys.setrecursionlimit(10000)
sys.path.append('..')
from Game import Game
from .OthelloLogic import Board
from mxnet import nd


class OthelloGame(Game):
    def __init__(self, n=8):
        self.n = n

    def getInitBoard(self):
        # return initial board (numpy board)
        b = Board(self.n)
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
        b = Board(self.n)
        b.pieces = board.copy()
        move = (int(action/self.n), action%self.n)
        b.execute_move(move, player)
        return (b.pieces, -player)

    def getValidMoves(self, board, player):
        # return a fixed size binary vector
        valids = nd.zeros((self.getActionSize(),))
        b = Board(self.n)
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
        b = Board(self.n)
        b.pieces = board.copy()
        if b.has_legal_moves(player):
            return 0
        if b.has_legal_moves(-player):
            return 0
        if b.countDiff(player) > 0:
            return 1
        return -1

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
        pi_board = nd.reshape(pi[:-1], (1,self.n, self.n))
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
        return board.reshape((self.n*self.n)).__repr__().split(']\n')[0][2:]

    def getScore(self, board, player):
        b = Board(self.n)
        b.pieces = board.copy()
        return b.countDiff(player)

def display(board):
    n = board.shape[0]

    for y in range(n):
        print (y,"|",end="")
    print("")
    print(" -----------------------")
    for y in range(n):
        print(y, "|",end="")    # print the row #
        for x in range(n):
            piece = board[y][x]    # get the piece to print
            if piece == -1: print("b ",end="")
            elif piece == 1: print("W ",end="")
            else:
                if x==n:
                    print("-",end="")
                else:
                    print("- ",end="")
        print("|")

    print("   -----------------------")
