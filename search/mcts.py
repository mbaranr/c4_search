import random
from math import sqrt, log
from search.node import NodeMCTS
from c4.state import C4State
from search.util import BudgetExceededError

# implementation taken from James Stovold's lab material

class MCTS_UCT:

    def __init__(self, 
                 budget: int, 
                 strategy: str,
                 spaces: int,
                 exploration_factor: float=sqrt(2)
                 ):
        self.budget = budget
        self.exploration_factor = exploration_factor
        self.rootnode = None
        self.turn_count = 0

        max_moves = spaces//2

        # budget allocations for each move
        if strategy == "greedy":
            self.budget_alloc = [budget // (max_moves // 2) if i < max_moves // 2 else 0 for i in range(max_moves)]
        elif strategy == "optimistic":
            scale = 0.9  # 90% of the previous allocation
            # decreasing itermax proportions
            budget_alloc = [scale**i for i in range(max_moves)]
            total_scale = sum(budget_alloc)  
            # scale proportions 
            self.budget_alloc = [int(budget * (x / total_scale)) for x in budget_alloc]
            self.budget_alloc[-1] += budget - sum(self.budget_alloc)  # correct rounding errors
        elif strategy == "safe":
            self.budget_alloc = [budget//max_moves]*max_moves

    def pick_move(self, rootstate: C4State):
        """
        Conducts a game tree search using the MCTS-UCT algorithm.
        Assumes that 2 players are alternating.

        Paramters:
        rootstate (C4State): The game state for which an action must be selected (where search begins).
        
        Returns:
        (int): Action that will be taken by an agent (column of C4 grid).
        """
        self.rootnode = NodeMCTS(state=rootstate)
        
        itermax = self.budget_alloc[self.turn_count]
        self.budget -= itermax
        self.turn_count += 1

        if itermax == 0:
            raise BudgetExceededError("MCTS ran out of computational budget!")
        
        for _ in range(itermax):
            
            state = rootstate.copy()
            
            node = self.selection(self.rootnode, state)
            child = self.expansion(node, state)
            
            self.rollout(state)
            self.backpropagation(child, state)

        return self.action_selection(self.rootnode)

    def ucb1(self, 
             node: NodeMCTS, 
             child: NodeMCTS, 
             ):
        return child.wins / child.visits + self.exploration_factor * sqrt(log(node.visits) / child.visits)
    
    def selection(self, 
                  node: NodeMCTS, 
                  state: C4State, 
                  ):
        if not node.is_fully_expanded() or node.children == []:
            return node
        selected_node = sorted(node.children, key=lambda child: self.ucb1(node, child))[-1]
        state.make_move(selected_node.move)
        return self.selection(selected_node, state)

    def expansion(self, node: NodeMCTS, state: C4State):
        child = node
        if node.untried_moves != []:  # if we can expand (i.e. state/node is non-terminal)
            move = random.choice(node.untried_moves)
            state.make_move(move)
            child = node.add_child(move, state)
        return child

    def rollout(self, state: C4State):
        while state.get_possible_moves() != []:
            state.make_move(random.choice(state.get_possible_moves()))

    def backpropagation(self, node: NodeMCTS, state: C4State):
        if node is not None:
            node.update(int(state.winner == node.last_player))
            self.backpropagation(node.parent, state)

    def action_selection(self, node: NodeMCTS):
        return sorted(node.children, key=lambda c: c.wins / c.visits)[-1].move
    




