import os
import shutil
import time
import pandas as pd
from tqdm import tqdm
from search.minimax import Minimax
from search.mcts import MCTS_UCT
from c4.state import C4State
from search.node import *
from itertools import product

def get_tree_metrics(in_place: dict, node: Node):
    if isinstance(node, NodeMinimax):
        in_place["n_pruned"] += 1 if node.pruned else 0
    in_place["n_nodes"] += 1
    for child in node.children:
        get_tree_metrics(in_place, child)
    return in_place

def run_simulation(id: int,
                   mm_depth: int,
                   budget: int,
                   is_mm_p1: bool,
                   mcts_strat: str,
                   state: C4State,
                   file_name: str="bin/simulations.parquet"
                   ):
    
    if os.path.exists(file_name):
        df = pd.read_parquet(file_name)
    else:
        df = pd.DataFrame(columns=["sim_id", "move_id", "ms", "agent_curr", "agent_start", "n_nodes", 
                                   "n_pruned", "is_win", "depth", "budget_total", "budget_consumed", "budget_left", "budget_exceeded"])

    p1 = Minimax(budget=budget, depth=mm_depth) if is_mm_p1 else \
          MCTS_UCT(budget=budget, strategy=mcts_strat, spaces=state.rows*state.cols)
    p2 = MCTS_UCT(budget=budget, strategy=mcts_strat, spaces=state.rows*state.cols) if is_mm_p1 else \
          Minimax(budget=budget, depth=mm_depth)
    
    start_agent = "minimax" if is_mm_p1 else "mcts"
    is_win = False
    move_count = 0

    prev_budget_p1 = p1.budget
    prev_budget_p2 = p2.budget

    # keep playing until winner
    while state.winner == 0:
        if state.last_player == 2:
            curr_agent = p1
            curr_agent_name = "minimax" if isinstance(p1, Minimax) else "mcts"
            prev_budget = prev_budget_p1
        else:
            curr_agent = p2
            curr_agent_name = "minimax" if isinstance(p2, Minimax) else "mcts"
            prev_budget = prev_budget_p2

        budget_exceeded = False
        try:
            start_time = time.time()
            move = curr_agent.pick_move(state)
            end_time = time.time()
        except Exception as e:
            print(e)
            end_time = time.time()
            budget_exceeded = True
            move = None

        if not budget_exceeded:
            if state.last_player == 2:
                prev_budget_p1 = curr_agent.budget
            else:
                prev_budget_p2 = curr_agent.budget

            budget_consumed = prev_budget - curr_agent.budget
            budget_left = curr_agent.budget

            # explore search tree for metrics
            in_place = get_tree_metrics({"n_nodes": 0, "n_pruned": 0}, curr_agent.rootnode)
        else:
            # default values when budget is exceeded
            budget_consumed = prev_budget
            budget_left = 0
            in_place = {"n_nodes": 0, "n_pruned": 0}

        if not budget_exceeded:
            state.make_move(move)
            is_win = state.winner != 0

        # record data for current move
        df = pd.concat([df, pd.DataFrame({
            "sim_id": [id],
            "move_id": [move_count],
            "ms": [(end_time - start_time) * 1000],  # s to ms
            "agent_curr": [curr_agent_name],
            "agent_start": [start_agent],
            "n_nodes": [in_place["n_nodes"]],
            "n_pruned": [in_place["n_pruned"]],
            "is_win": [is_win if not budget_exceeded else False],
            "depth": [mm_depth],
            "budget_total": [budget],
            "budget_consumed": [budget_consumed],
            "budget_left": [budget_left],
            "budget_exceeded": [budget_exceeded]
        })])

        move_count += 1

        # stop simulation if budget exception occurred
        if budget_exceeded:
            break

    df.to_parquet(file_name, index=False)

def run_all_simulations(repeats: int):

    # creating output directory
    if os.path.exists("bin"):
        shutil.rmtree("bin")  
    os.makedirs("bin", exist_ok=True)  

    budgets = [100, 500, 1000, 10000]
    strats = ["safe", "optimistic", "greedy"]
    depths = [1, 2, 3, 4, 5]
    is_mm_p1_options = [True, False]
    
    total_simulations = len(budgets) * len(strats) * len(depths) * len(is_mm_p1_options) * repeats
    print(f"Running {total_simulations} simulations...")

    sim_count = 0
    with tqdm(total=total_simulations, desc="Running simulations") as pbar:
        for budget, strat, depth, is_mm_p1, _ in product(budgets, strats, depths, is_mm_p1_options, range(repeats)):
            state = C4State(rows=6, cols=7, connect=4)
            run_simulation(
                id=sim_count, 
                mm_depth=depth, 
                budget=budget, 
                is_mm_p1=is_mm_p1, 
                mcts_strat=strat, 
                state=state
            )
            sim_count += 1
            pbar.update(1)
