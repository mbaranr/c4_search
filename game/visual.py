import pygame
from game.state import C4State

class C4Visual:
    """
    Helper class to display a Connect 4 game state.
    """

    def __init__(self, 
                 state: C4State,
                 cell_rad: float=50,
                 margin: float=20
                 ):
        
        self.state = state
        self.cell_rad = cell_rad
        self.margin = margin

        # board dimensions
        self.cols = self.state.cols
        self.rows = self.state.rows
        
        # width and height of pygame window
        self.w = margin + (self.cols * (cell_rad*2 + margin))
        self.h = margin + (self.rows * (cell_rad*2 + margin))
        
        pygame.init()
        self.screen = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("C4")

    def render(self):
        """
        Renders game state.
        """

        self.screen.fill((0, 0, 255))  # blue background
        
        for row in range(self.rows):
            for col in range(self.cols):
                # calculate position relative to the window
                x = self.margin + col * (self.cell_rad*2 + self.margin) + self.cell_rad
                y = self.margin + row * (self.cell_rad * 2 + self.margin) + self.cell_rad
                
                if self.state.board[row][col] == 1:
                    pygame.draw.circle(self.screen, (255, 0, 0), (x, y), self.cell_rad)     # p1: red
                elif self.state.board[row][col] == 2:
                    pygame.draw.circle(self.screen, (255, 255, 0), (x, y), self.cell_rad)   # p2: yellow
                else:
                    pygame.draw.circle(self.screen, (0, 0, 0), (x, y), self.cell_rad)     # empty: white
    
        pygame.display.update()

    def wait_for_quit(self):
        """
        Method to wait waiting window closing.
        """
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
        pygame.quit()
