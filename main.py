import time
from search.mcts import MCTS_UCT
from sim.collect_data import *
from sim.figures import *
from search.minimax import Minimax
from c4.state import C4State
from c4.visual import C4Visual

def play_demo(state: C4State, budget, mm_depth: int, mcts_strat: str, is_mm_p1: bool, window: C4Visual=None, delay: float=.5):
    
    p1 = Minimax(budget=budget, depth=mm_depth, max_player=1) if is_mm_p1 else \
          MCTS_UCT(budget=budget, strategy=mcts_strat, spaces=state.rows*state.cols)
    p2 = MCTS_UCT(budget=budget, strategy=mcts_strat, spaces=state.rows*state.cols) if is_mm_p1 else \
          Minimax(budget=budget, depth=mm_depth, max_player=1)

    if window is not None:
        window.render()

    budget_exceeded = False

    # keep playing until winner
    while state.winner == 0 and state.get_possible_moves():
        if state.last_player == 2:
            curr_agent = p1
        else:
            curr_agent = p2

        try:
            move = curr_agent.pick_move(state)
        except BudgetExceededError:
            budget_exceeded = True

        if not budget_exceeded:
            state.make_move(move)   

        # stop simulation if budget exception occurred
        if budget_exceeded:
            break

        if window is not None:
            window.render()

        time.sleep(delay)
    
    if window is not None:
        window.render()

    if state.winner != 0:
        print(f"Winner: Player {state.winner}")
    elif budget_exceeded:
        print(f"Winner: Player {state.last_player}. Player {3 - state.last_player} forfeits! (hint: not enough budget)")
    else:
        print("Draw!")
    
    time.sleep(3)

def get_demo_params():
    print("Enter parameters for the demo:")
    
    connect = int(input("Enter connect sequence (default 4): ") or 4)
    bf = int(input("Enter board columns (default 7): ") or 7)
    gui = (input("Enable GUI? If yes, please don't attempt to close the window (yes/no, default yes): ").lower() or 'yes') == 'yes'
    is_mm_p1 = (input("Is Minimax Player 1? (yes/no, default yes): ").lower() or 'yes') == 'yes'
    mm_depth = int(input("Enter Minimax depth (default 2): ") or 2)
    budget = int(input("Enter budget (default 500): ") or 500)
    mcts_strat = input("Enter MCTS strategy (thrifty/optimistic/greedy, default thrifty): ") or "thrifty"
    
    return connect, bf, gui, is_mm_p1, mm_depth, budget, mcts_strat

if __name__ == "__main__":
    print("#####################")
    print("# CONNECT X SEARCH: #")
    print("#####################")
    choice = input("Do you want to run a demo or simulations? (demo/simulations, default demo): ").lower() or "demo"

    if choice == "demo":
        connect, bf, gui, is_mm_p1, mm_depth, budget, mcts_strat = get_demo_params()
        
        initial_state = C4State(bf - 1, bf, connect)
        window = None if not gui else C4Visual(initial_state, margin=30)

        play_demo(state=initial_state, budget=budget, mm_depth=mm_depth, mcts_strat=mcts_strat, is_mm_p1=is_mm_p1, window=window)
    
    elif choice == "simulations":
        print("This might take a while...")

        run_all_simulations(100)
        
        budgets = {100: 1, 500: 2, 1000: 2, 10000: 4}   # budget with respective best depth, gathered from prev simulations
        run_all_simulations_(200, budgets)

        fig_1("bin/simulations_1.parquet", "bin/fig_1.png")
        fig_2("bin/simulations_1.parquet", budgets, "bin/fig_2.png")
        fig_3("bin/simulations_1.parquet", budgets, "bin/fig_3.png")
        fig_4("bin/simulations_1.parquet", "bin/simulations_2.parquet", budgets, "bin/fig_4.png")
    
    else:
        print("Invalid option, please select either 'demo' or 'simulations'.")
    