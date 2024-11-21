import threading
import time
from search.mcts import mcts_uct
from game.state import C4State
from game.visual import C4Visual
import random
import pygame

def play(initial_state: C4State, budget=10000, window: C4Visual=None, delay: float=.0):
    state = initial_state

    if window is not None:
        window.render()

    while state.winner == 0:  # If we're not in a terminal state
        
        # p2: yellow
        if state.last_turn == 1:
            move = mcts_uct(state, itermax=budget)
            # move = random.choice(state.get_possible_moves())
        # p1: red
        else:
            move = mcts_uct(state, itermax=budget)
            # move = random.choice(state.get_possible_moves())
        
        state.make_move(move)
        
        if window is not None:
            window.render()
        
        time.sleep(delay)
    
    print(f"Winner: Player {state.winner}")

if __name__ == "__main__":

    game_state = C4State(6, 7, 4)
    window = C4Visual(game_state, margin=30)

    play(game_state, 1000, window, 0.5)

    



    