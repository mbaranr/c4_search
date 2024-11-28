import time
from search.mcts import MCTS_UCT
from sim.collect_data import run_all_simulations 
from search.minimax import Minimax
from c4.state import C4State
from c4.visual import C4Visual

def play_demo(initial_state: C4State, budget, is_mm_p1: bool, window: C4Visual=None, delay: float=.5):
    state = initial_state
    minimax = Minimax(budget, 2)
    mcts = MCTS_UCT(budget, strategy="greedy", spaces=state.rows*state.cols)

    if window is not None:
        window.render()

    while state.winner == 0:  # If we're not in a terminal state
        
        # p2: yellow
        if state.last_player == 1:
            move = minimax.pick_move(state)
        # p1: red
        else:
            move = mcts.pick_move(state)
        
        state.make_move(move)
        
        if window is not None:
            window.render()
        
        time.sleep(delay)
    
    print(f"Winner: Player {state.winner}")

if __name__ == "__main__":

    run_all_simulations(100)

    # game_state = C4State(6, 7, 4)
    # window = C4Visual(game_state, margin=30)

    # play(game_state, 100, window, 1)

    



    