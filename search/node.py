from game.state import C4State

class Node:
    """ 
    Implementation taken from James Stovold's lab material.

    Node of a game tree. A tree is a connected acyclic graph.
    Note: self.wins is from the perspective of playerJustMoved.
    """

    def __init__(self, 
                 move: int=None, 
                 parent=None, 
                 state: C4State=None
                 ):
        self.move = move  # move taken to reach this game state 
        self.parent = parent  # None if root node
        self.children = []
        
        self.wins = 0
        self.visits = 0
        self.untried_moves = state.get_possible_moves()  # future children
        self.last_turn = state.last_turn  # 1 or 2 (to check which player won)
        
    def is_fully_expanded(self):
        return self.untried_moves == []

    def add_child(self, move, state):
        """
        Adds a new child node to this node. 

        Paremeters:
        move (int): action taken by the player.
        state (C4State): state corresponding to new child node.
        
        Returns:
        child (Node): Newly created child.
        """
        child = Node(move=move, parent=self, state=state)
        self.untried_moves.remove(move)
        self.children.append(child)
        return child

    def update(self, result):
        """
        Updates node statistics with the result from last rollout.
        
        Parameters:
        result (int): 1 for victory, 0 for draw / loss.
        """
        self.visits += 1
        self.wins += result