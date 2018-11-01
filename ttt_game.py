import numpy as np

class GameEnvironment:
    def __init__(self, gameState = None, opponent = None):
        self.gameState = gameState if gameState is not None else GameEnvironment.randomInitialGame()
        self.opponent = opponent   if opponent is not None  else RandomPlayer()
    
    def step(self, row, col):
        # Illegal moves result in auto-loss
        if not self.gameState.isLegalMove(row, col):
            return (self.gameState, -10, True, None)
        
        # Player plays move
        self.gameState.playMove_unchecked(row, col)
        winner = self.gameState.getWinner_unchecked()
        if winner != 0:
            return (self.gameState, 1, True, None)

        # Check tie
        if self.gameState.isGameOver():
            return (self.gameState, 0, True, None)

        # Opponent plays move
        self.opponent.makeMove_unchecked(self.gameState)
        winner = self.gameState.getWinner_unchecked()
        if winner != 0:
            return (self.gameState, -1, True, None)
        
        # Check tie
        if self.gameState.isGameOver():
            return (self.gameState, 0, True, None)
        
        return (self.gameState, 0, False, None)
    
    def reset(self):
        self.gameState = GameEnvironment.randomInitialGame()
        return self.gameState

    def randomInitialGame():
        gameState = GameState()
        if np.random.rand() < 0.5:
            gameState.mostRecentPlayer = 1
            gameState.playRandomLegalMove_unchecked()
        return gameState

class RandomPlayer:
    def __init__(self):
        pass
    
    def makeMove_unchecked(self, gameState):
        gameState.playRandomLegalMove_unchecked()

class GameState:
    def __init__(self):
        self.mostRecentMove = None
        self.mostRecentPlayer = -1
        self.board = np.zeros([3,3], dtype = "int8")
        self.legalMoves = { (0,0), (0,1), (0,2), 
                            (1,0), (1,1), (1,2),
                            (2,0), (2,1), (2,2) }

    def playMove_unchecked(self, row, col):
        player = self.getNextPlayer()
        self.board[row][col] = player
        self.mostRecentPlayer = player
        self.mostRecentMove = (row, col)
        self.legalMoves.remove((row, col))

    def getWinner_unchecked(self):
        row = self.mostRecentMove[0]
        col = self.mostRecentMove[1]
        # Check main diagonal
        if (row == col):    
            if self.mostRecentPlayer == self.board[(row + 1) % 3][(col + 1) % 3] == self.board[(row + 2) % 3][(col + 2) % 3]:
                return self.mostRecentPlayer
        
        # Check anti-diagonal
        if (row + col == 2):    
            if self.mostRecentPlayer == self.board[(row - 1) % 3][(col + 1) % 3] == self.board[(row - 2) % 3][(col + 2) % 3]:
                return self.mostRecentPlayer

        # Check column
        if self.mostRecentPlayer == self.board[(row + 1) % 3][col] == self.board[(row + 2) % 3][col]:
            return self.mostRecentPlayer

        # Check row
        if self.mostRecentPlayer == self.board[row][(col + 1) % 3] == self.board[row][(col + 2) % 3]:
            return self.mostRecentPlayer

        return 0

    def getNextPlayer(self):
        # 1 -> -1
        # -1 -> 1
        return (self.mostRecentPlayer & 3) - 2
    
    def isLegalMove(self, row, col):
        return (row, col) in self.legalMoves

    def getLegalMoves(self):
        return self.legalMoves

    def isGameOver(self):
        return len(self.legalMoves) == 0

    def playRandomLegalMove_unchecked(self):
        row,col = self.getRandomLegalMove_unchecked()
        self.playMove_unchecked(row, col)

    def getRandomLegalMove_unchecked(self):
        legalMovesList = list(self.legalMoves)
        idx = np.random.choice(len(legalMovesList))
        return legalMovesList[idx]

    def __str__(self):
        ret = ""
        pieceToStringMap = {1: " X ", 0: "   ", -1: " O "}
        for r, row in enumerate(self.board):
            for c, col in enumerate(row):
                piece = col
                ret += pieceToStringMap[piece]
                if c < 2:
                    ret += "|"
            if r < 2:
                ret += "\n-----------\n"
        return ret
