import numpy as np
from scipy.signal import convolve2d

class C4State(object):
    """
    Implementation inspired by James Stovold's lab material.
    
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

        self.last_player = 2      # p1 will start
        self.last_move = None
        self.winner = 0         # 0: no winner, 1: p1 wins, 2: p2 wins

        self.board = np.zeros((self.rows, self.cols), dtype=int)

        self.directions = {"horizontal": (0, +1), "diag": (+1, +1), "vertical": (+1, 0), "antidiag": (+1, -1)}

    def make_move(self, movecol: int):
        """ 
        Changes state by "dropping" a chip in the specified column
        """
        assert movecol >= 0 and movecol <= self.cols and self.board[0][movecol] == 0
        row = 0
        while row < self.rows and self.board[row][movecol] == 0:
            row += 1
        row -= 1 

        self.last_move = movecol
        self.last_player = 3 - self.last_player
        self.board[row][movecol] = self.last_player
        self.update_winner(row, movecol)
    
    def undo_move(self, movecol: int):
        """
        Undo last move by removing topmost chip on column specified.
        """
        row = 0
        while row < self.rows and self.board[row][movecol] == 0:
            row += 1
        if row == self.rows:  # no chip
            raise ValueError("Cannot undo move in an empty column.")
        
        # remove chip
        self.board[row][movecol] = 0
        self.last_player = 3 - self.last_player  # revert to the previous player's turn
        self.winner = 0  # reset winner
    
    def get_possible_moves(self):
        """
        Get list with all possible moves.
        Not full column indices.
        """
        if self.winner != 0:
            return []
        return [col for col in range(self.cols) if self.board[0][col] == 0]

    def available_immediately(self, row, col):
        """
        Check if a chip can be placed in the provided x and y coordinates immediately.
        """
        if self.on_board(row, col) and self.board[row][col] == 0:
            if (row == 0) or self.board[row-1][col] != 0:
                return True
        return False

    def on_board(self, row, col):
        return row >= 0 and row < self.rows and col >= 0 and col < self.cols

    def update_winner(self, row, col):
        """ 
        Checks if last turn player just won the game. 
        Accepts last moves coordinates for a faster lookup.
        
        Parameters:
        player (int): Player to check for (optional)
        
        Returns:
        bool: True if last turn player won the game, False otherwise
        """
        player = self.last_player
        for (dx, dy) in self.directions.values():
            p = 1
            while self.on_board(row+p*dx, col+p*dy) and self.board[row+p*dx][col+p*dy] == player:
                p += 1
            n = 1
            while self.on_board(row-n*dx, col-n*dy) and self.board[row-n*dx][col-n*dy] == player:
                n += 1
            if p + n >= (self.connect + 1):
                self.winner = player
                return
        return 
    
    def find_sequence(self, length, player):
        """
        Finds sequences of given length on all directions.
        """
        board = (self.board == player)  # binary mask for the last player's chips

        kernel = np.ones((1, length), dtype=int)
        diag_kernel = np.eye(length, dtype=int)
        
        result_dict = {}
        result_dict["horizontal"] = self.find_direction(board, length, kernel, "horizontal")
        result_dict["vertical"] = self.find_direction(board, length, kernel.T, "vertical")
        result_dict["diag"] = self.find_direction(board, length, diag_kernel, "diag")
        result_dict["antidiag"] = self.find_direction(board, length, np.fliplr(diag_kernel), "antidiag")

        return result_dict

    def find_direction(self, board, length, kernel, direction):
        """
        Helper function for find_sequence.
        Finds sequences of specified length in a given direction.

        Parameters:
        board (np.array): Binary array.
        length (int): Sequence length.
        kernel (np.array): Direction kernel for convolution.

        Returns:
        sequences (list): List of coordinate (start, end) pairs for each match.
        """

        def within_bounds(row, col, rows, cols):
            return row >= 0 and row < rows and col >= 0 and col < cols

        conv = convolve2d(board, kernel, mode="valid")  # 2D convolution
        bool_conv = conv == length              
        match_indices = np.argwhere(bool_conv)  # matches for the sequence length
    
        sequences = []
        for idx in match_indices:
            
            start_row, start_col = idx

            # checking if sequence belongs to a bigger sequence
            is_pure = True
            dx, dy = self.directions[direction]
            if within_bounds(start_row+dx, start_col+dy, bool_conv.shape[0], bool_conv.shape[1]) and bool_conv[start_row+dx][start_col+dy] or \
                within_bounds(start_row-dx, start_col-dy, bool_conv.shape[0], bool_conv.shape[1]) and bool_conv[start_row-dx][start_col-dy]:
                is_pure = False
            
            if not is_pure:
                continue

            end_row = start_row + kernel.shape[0] - 1
            end_col = start_col + kernel.shape[1] - 1

            # due to convolution, antidiag cols are inversed
            if direction == "antidiag":
                start_col, end_col = end_col, start_col

            sequences.append(((start_row, start_col), (end_row, end_col)))
        
        return sequences

    def copy(self):
        """ 
        Creates a deep copy of the game state.
        """
        copy = C4State(rows=self.rows, cols=self.cols, connect=self.connect)
        copy.last_player = self.last_player
        copy.winner = self.winner
        copy.board = self.board.copy()
        return copy
