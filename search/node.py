from c4.state import C4State

class Node:
    """ 
    Implementation inspired from James Stovold's lab material.

    Generic game tree node implementation. A tree is a connected acyclic graph.
    """

    def __init__(self, 
                 move: int=None, 
                 parent=None, 
                 ):
        self.move = move  # move taken to reach this game state 
        self.parent = parent  # None if root node
        self.children = []

    def best_move(self):
        raise NotImplementedError("The method 'best_move' must be implemented in a subclass.")

    def add_child(self, move):
        raise NotImplementedError("The method 'add_child' must be implemented in a subclass.")

    def update(self, result):
        raise NotImplementedError("The method 'update' must be implemented in a subclass.")


class NodeMCTS(Node):
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
        super().__init__(move, parent)
        self.wins = 0
        self.visits = 0
        self.last_player = state.last_player  # 1 or 2 (to check which player won)
        self.untried_moves = state.get_possible_moves()  # future children

    def is_fully_expanded(self):
        return self.untried_moves == []
    
    def update(self, result):
        """
        Updates node statistics with the result from last rollout.
        
        Parameters:
        result (int): 1 for victory, 0 for draw / loss.
        """
        self.visits += 1
        self.wins += result

    def best_move(self):
        child = sorted(self.children, key=lambda c: c.wins / c.visits)[-1]
        return {"move": child.move, "node": child}

    def add_child(self, move, state):
        """
        Adds a new child node of type NodeMinimax to this node.
        """
        child = NodeMCTS(move=move, parent=self, state=state)
        self.untried_moves.remove(move)
        self.children.append(child)
        return child

class NodeMinimax(Node):
    """ 
    Minimax node.
    """

    def __init__(self, 
                 move: int=None, 
                 parent=None, 
                 util: float=None,
                 ):
        super().__init__(move, parent)
        self.pruned = False
        self.util = util    

    def update(self, result):
        self.util = result

    def best_move(self):
        # excluding pruned children
        valid_children = [child for child in self.children if not child.pruned]

        if not valid_children:
            # if every child was pruned, loss is inevitable
            arbitrary_child = self.children[-1]     # pick arbitrary move
            return {"move": arbitrary_child.move, "node": arbitrary_child}
        
        best_child = sorted(valid_children, key=lambda c: c.util)[-1]
        return {"move": best_child.move, "node": best_child}

    def add_child(self, move):
        """
        Adds a new child node of type NodeMinimax to this node.
        """
        child = NodeMinimax(move=move, parent=self)
        self.children.append(child)
        return child
    
    def print_tree(self, level=0):
        """
        Recursively prints the tree structure.
        """
        indent = "  " * level
        pruned_status = "(Pruned)" if self.pruned else ""
        print(f"{indent}- Move: {self.move}, Util: {self.util} {pruned_status}")
        for child in self.children:
            child.print_tree(level + 1)

