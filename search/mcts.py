import random
from math import sqrt, log
from search.node import Node
from game.state import C4State

# implementation taken from James Stovold's lab material

def ucb1(node: Node, child: Node, exploration_factor=sqrt(2)):
    return child.wins / child.visits + exploration_factor * sqrt(log(node.visits) / child.visits)

def mcts_uct(rootstate: C4State, 
             itermax: int, 
             exploration_factor: float=sqrt(2)
             ):
    """
    Conducts a game tree search using the MCTS-UCT algorithm.
    Assumes that 2 players are alternating.

    Paramters:
    rootstate (C4State): The game state for which an action must be selected (where search begins).
    itermax (int): Number of MCTS iterations to be carried out. Also knwon as the computational budget.
    
    Returns:
    (int): Action that will be taken by an agent (column of C4 grid).
    """
    rootnode = Node(state=rootstate)
    
    for _ in range(itermax):
        node  = rootnode
        state = rootstate.copy()
        
        node = selection(node, state, selection_policy=ucb1, selection_policy_args=[exploration_factor])
        node = expansion(node, state)
        
        rollout(state)
        backpropagation(node, state)

    return action_selection(rootnode)

def selection(node: Node, state: C4State, selection_policy=ucb1, selection_policy_args=[]):
    if not node.is_fully_expanded() or node.children == []:
        return node
    selected_node = sorted(node.children, key=lambda child: selection_policy(node, child, *selection_policy_args))[-1]
    state.make_move(selected_node.move)
    return selection(selected_node, state)

def expansion(node: Node, state: C4State):
    if node.untried_moves != []:  # if we can expand (i.e. state/node is non-terminal)
        move = random.choice(node.untried_moves)
        state.make_move(move)
        node = node.add_child(move, state)
    return node

def rollout(state: C4State):
    while state.get_possible_moves() != []:
        state.make_move(random.choice(state.get_possible_moves()))

def backpropagation(node: Node, state: C4State):
    if node is not None:
        node.update(int(state.winner == node.last_turn))
        backpropagation(node.parent, state)

def action_selection(node: Node):
    return sorted(node.children, key=lambda c: c.wins / c.visits)[-1].move
