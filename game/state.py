import numpy as np
from scipy.ndimage import convolve

class C4State(object):
    """
    Implementation inspired by James Stovold's lab material
    
    Connect 4 game state
    Board is represented by a 2D matrix
    Each entry of the matrix can be:
        - 0: no chip
        - 1: p1 chip (red)
        - 2: p2 chip (yellow)
    """

    def __init__(self, 
                 rows: int=7, 
                 cols: int=6,
                 connect: int=4
                 ):
        self.connect = connect
        self.rows = rows
        self.cols = cols

        self.last_turn = 2      # p1 will start
        self.winner = 0         # 0: no winner, 1: p1 wins, 2: p2 wins

        self.board = np.zeros((self.rows, self.cols), dtype=int)

        # kernels created once for optimization, winning patterns
        self.kernels = [
            np.ones((1, connect), dtype=int),       # horizontal
            np.ones((connect, 1), dtype=int),       # vertical
            np.eye(connect, dtype=int),             # top-left 2 bottom-right
            np.fliplr(np.eye(connect, dtype=int))   # top-right 2 bottom-left
        ]

    def make_move(self, movecol: int):
        """ 
        Changes state by "dropping" a chip in the specified column
        """
        assert movecol >= 0 and movecol <= self.cols and self.board[0][movecol] == 0
        row = 0
        while row < self.rows and self.board[row][movecol] == 0:
            row += 1
        row -= 1 

        self.last_turn = 3 - self.last_turn
        self.board[row][movecol] = self.last_turn
        self.update_winner(self.last_turn)
            
    def get_possible_moves(self):
        """
        Get list with all possible moves.
        Not full column indices.
        """
        if self.winner != 0:
            return []
        return [col for col in range(self.cols) if self.board[0][col] == 0]

    def update_winner(self, player=None):
        """ 
        Checks if last turn player just won the game. 
        Accepts last moves coordinates for a faster lookup.
        
        Parameters:
        player (int): Player to check for (optional)
        
        Returns:
        bool: True if last turn player won the game, False otherwise
        """
        players = [player] if player is not None else [1, 2]

        for kernel in self.kernels:
            for player in players:
                # slide kernel over the board to match with binary pattern
                res = convolve((self.board == player).astype(int), kernel, mode="constant", cval=0) 
                if np.any(res >= self.connect): 
                    self.winner = player
                    return 
                
    def copy(self):
        """ 
        Creates a deep copy of the game state.
        """
        copy = C4State(rows=self.rows, cols=self.cols, connect=self.connect)
        copy.last_turn = self.last_turn
        copy.winner = self.winner
        copy.board = self.board.copy()
        return copy

