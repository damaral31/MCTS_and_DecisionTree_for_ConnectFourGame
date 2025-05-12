from typing import Optional
from Game.ConnectFour import ConnectFour

class Node:
    """
    A node for the MCTS tree.

    Methods
    -------
    add_child(child_state: ConnectFour, move: Optional[int] = None) -> None
        Add a child to the node.
    is_terminal() -> bool
        Check if the node is terminal.
    update(reward: float) -> None
        Update the reward and visit count of the node.
    fully_explored() -> bool
        Check if all the children of the node have been explored.
    """

    def __init__(self, state: ConnectFour, parent=None) -> None:
        """
        Create a new node.

        Parameters
        ----------
        state: the state of the node
        parent: the parent node of the node
        """
        self.visits = 1
        self.reward = 0.0
        self.state = state
        self.children = []
        self.children_move = []
        self.parent = parent

    def add_child(self, child_state: ConnectFour, move: Optional[int] = None) -> None:
        """
        Add a child to the node.

        Parameters
        ----------
        child_state: the state of the child node
        move: the move that led to the child node

        Returns
        -------
        none
        """
        child = Node(child_state, self)
        self.children.append(child)
        self.children_move.append(move)

    def is_terminal(self) -> bool:
        """
        Check if the node is terminal.

        Returns
        -------
        bool: True if the node is terminal, False otherwise
        """
        return self.state.is_over()

    def update(self, reward: float) -> None:
        """
        Update the reward and visit count of the node.

        Parameters
        ----------
        reward: the reward to add to the node

        Returns
        -------
        none
        """
        self.reward += reward
        self.visits += 1

    def fully_explored(self) -> bool:
        """
        Check if all the children of the node have been explored.

        Returns
        -------
        bool: True if all the children have been explored, False otherwise
        """
        if len(self.children) == len(self.state.legal_moves()):
            return True
        return False