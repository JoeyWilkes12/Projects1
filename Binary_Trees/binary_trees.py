# These imports are used in BST.draw().
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from matplotlib import pyplot as plt
import numpy as np
import random
import time


class SinglyLinkedListNode:
    """A node with a value and a reference to the next node."""
    def __init__(self, data):
        self.value, self.next = data, None

class SinglyLinkedList:
    """A singly linked list with a head and a tail."""
    def __init__(self):
        #self.head, self.tail = None, None
        self.head, self.tail,self.size = None, None, 0


    def append(self, data):
        """Add a node containing the data to the end of the list."""
        n = SinglyLinkedListNode(data)
        if self.head is None:
            self.head, self.tail = n, n
            self.size +=1
        else:
            self.tail.next = n
            self.tail = n
            self.size +=1

    def iterative_find(self, data):
        """Search iteratively for a node containing the data.
        If there is no such node in the list, including if the list is empty,
        raise a ValueError.

        Returns:
            (SinglyLinkedListNode): the node containing the data.
        """
        current = self.head
        while current is not None:
            if current.value == data:
                return current
            current = current.next
        raise ValueError(str(data) + " is not in the list")

    """
        def size_of_List(self):

        #the function returns the number of nodes in the linkedlist
        #which is used in probl4, but the idea failed

        return self.size
    """

    # Problem 1
    def recursive_find(self, data):
        """Search recursively for the node containing the data.
        If there is no such node in the list, including if the list is empty,
        raise a ValueError.

        Returns:
            (SinglyLinkedListNode): the node containing the data.
        """
        current = self.head

        def step(current):
            """ Step through the nodes in the linked list Recursively
            until the node containing the data is found, if the tree doesn't
            contain the data, or the tree is empty then ValueError will be raised
            """
            #there are three recursive cases
            #Either the current node is None meaning the end of the list has been reached
            #We found the correct node at some point in the recursion
            #or we did not find the correct node and continue recursion

            if current is None:
                raise ValueError(str(data) + " is not in the tree.")
            elif current.value == data:
                return current
            else:
                return step(current.next)

        return step(current)

class BSTNode:
    """A node class for binary search trees. Contains a value, a
    reference to the parent node, and references to two child nodes.
    """
    def __init__(self, data):
        """Construct a new node and set the value attribute. The other
        attributes will be set when the node is added to a tree.
        """
        self.value = data
        self.prev = None        # A reference to this node's parent node.
        self.left = None        # self.left.value < self.value
        self.right = None       # self.value < self.right.value

class BST:
    """Binary search tree data structure class.
    The root attribute references the first node in the tree.
    """
    def __init__(self):
        """Initialize the root attribute."""
        self.root = None

    def find(self, data):
        """Return the node containing the data. If there is no such node
        in the tree, including if the tree is empty, raise a ValueError.
        """

        # Define a recursive function to traverse the tree.
        def _step(current):
            """Recursively step through the tree until the node containing
            the data is found. If there is no such node, raise a Value Error.
            """
            if current is None:                     # Base case 1: dead end.
                raise ValueError(str(data) + " is not in the tree.")
            if data == current.value:               # Base case 2: data found!
                return current
            if data < current.value:                # Recursively search left.
                return _step(current.left)
            else:                                   # Recursively search right.
                return _step(current.right)

        # Start the recursion on the root of the tree.
        return _step(self.root)

    # Problem 2
    def insert(self, data):
        """Insert a new node containing the specified data.

        Raises:
            ValueError: if the data is already in the tree.

        Example:
            >>> tree = BST()                    |
            >>> for i in [4, 3, 6, 5, 7, 8, 1]: |            (4)
            ...     tree.insert(i)              |            / \
            ...                                 |          (3) (6)
            >>> print(tree)                     |          /   / \
            [4]                                 |        (1) (5) (7)
            [3, 6]                              |                  \
            [1, 5, 7]                           |                  (8)
            [8]                                 |
        """
        def step(current,data):
            """
            To recursively locate where the inserted node will be placed,
            this function checks whether the inserted value is
            greater than or less than currect node (starting with root)
            and then it verifies if node is a leaf, in
            which case the inserted value is appended to the parent with
            previous value assigned, but if the parent isn't a leaf then it recurses

            """
            newNode = BSTNode(data)
            if current.value == data:
                raise ValueError("Duplicate")
            else:
                # assuming current node is not the same
                #the compare current node value with data to be inserted
                #then verify if current node is a leaf

                if current.value < data:
                    #assuming data is greater than key of current node
                    if current.right is None:
                        current.right = newNode
                        newNode.prev = current
                    else:
                        current = current.right
                        step(current,data)
                else:
                    #thus data is less than key of current node
                    if current.left is None:
                        current.left = newNode
                        newNode.prev = current
                    else:
                        current = current.left
                        step(current,data)

        if self.root is None:
            self.root = BSTNode(data)
        else:
            #step function will verify that data is not duplicated
            current = self.root
            return step(current,data)

    # Problem 3
    def remove(self, data):
        """Remove the node containing the specified data.

        Raises:
            ValueError: if there is no node containing the data, including if
                the tree is empty.

        Examples:
            >>> print(12)                       | >>> print(t3)
            [6]                                 | [5]
            [4, 8]                              | [3, 6]
            [1, 5, 7, 10]                       | [1, 4, 7]
            [3, 9]                              | [8]
            >>> for x in [7, 10, 1, 4, 3]:      | >>> for x in [8, 6, 3, 5]:
            ...     t1.remove(x)                | ...     t3.remove(x)
            ...                                 | ...
            >>> print(t1)                       | >>> print(t3)
            [6]                                 | [4]
            [5, 8]                              | [1, 7]
            [9]                                 |
                                                | >>> print(t4)
            >>> print(t2)                       | [5]
            [2]                                 | >>> t4.remove(1)
            [1, 3]                              | ValueError: <message>
            >>> for x in [2, 1, 3]:             | >>> t4.remove(5)
            ...     t2.remove(x)                | >>> print(t4)
            ...                                 | []
            >>> print(t2)                       | >>> t4.remove(5)
            []                                  | ValueError: <message>
        """
        #this will flag an error if the targeted node isn't in the tree
        target = self.find(data)

        # if the node is a leaf
        if target.left is None and target.right is None:
            #3 cases - the target is the root, it's a left child, or right child
            if target is self.root:
                self.root = None
            elif target.prev.left is target:
                target.prev.left = None
            else:
                target.prev.right = None

        # if the node has two children
        elif target.left is not None and target.right is not None:
            #find and locate the predeccors value, and the
            #predeccesor will be deleted once values are swapped
            #there is the case when the predeccessor is the left node of the target
            current = target.left
            while current.right is not None:
                current = current.right

            #the predeccesor may have a left child
            if current.left is not None:
                target.value = current.value
                current.value = current.left.value
                current.left = None
            # the special case when the current node is left of the target node
            elif target.left is current:
                target.value = current.value
                target.left = None
            #case when predeccesor is the most right node and has a value smaller than target node value
            else:
                target.value = current.value
                current.prev.right = None

        #if the node has one child
        else:
            #3 cases - the target is the root, it's a left child, or right child
            #also with a child node either to the left or right

            def removeChild():
                """ this helper function will verify if the child node is
                to the left or right of the target, and delete node accordingly
                it will return the next node in the one sided chain
                """
                if target.left is not None:
                    target.value = target.left.value
                    return target.left
                    #target.left = None
                    #this totally killed off the rest of the nodes and was a big bug
                else:
                    target.value = target.right.value
                    return target.right
                    #target.right = None

            if target is self.root:
                nextNode = removeChild()
                self.root = nextNode
            elif target.prev.left is target:
                nextNode = removeChild()
                target.prev.left = nextNode
                nextNode.prev = target.prev
            else:
                nextNode = removeChild()
                target.prev.right = nextNode
                nextNode.prev = target.prev

    def __str__(self):
        """String representation: a hierarchical view of the BST.

        Example:  (3)
                  / \     '[3]          The nodes of the BST are printed
                (2) (5)    [2, 5]       by depth levels. Edges and empty
                /   / \    [1, 4, 6]'   nodes are not printed.
              (1) (4) (6)
        """
        if self.root is None:                       # Empty tree
            return "[]"
        out, current_level = [], [self.root]        # Nonempty tree
        while current_level:
            next_level, values = [], []
            for node in current_level:
                values.append(node.value)
                for child in [node.left, node.right]:
                    if child is not None:
                        next_level.append(child)
            out.append(values)
            current_level = next_level
        return "\n".join([str(x) for x in out])

    def draw(self):
        """Use NetworkX and Matplotlib to visualize the tree."""
        if self.root is None:
            return

        # Build the directed graph.
        G = nx.DiGraph()
        G.add_node(self.root.value)
        nodes = [self.root]
        while nodes:
            current = nodes.pop(0)
            for child in [current.left, current.right]:
                if child is not None:
                    G.add_edge(current.value, child.value)
                    nodes.append(child)

        # Plot the graph. This requires graphviz_layout (pygraphviz).
        nx.draw(G, pos=graphviz_layout(G, prog="dot"), arrows=True,
                with_labels=True, node_color="C1", font_size=8)
        plt.show()

class AVL(BST):
    """Adelson-Velsky Landis binary search tree data structure class.
    Rebalances after insertion when needed.
    """
    def insert(self, data):
        """Insert a node containing the data into the tree, then rebalance."""
        BST.insert(self, data)      # Insert the data like usual.
        n = self.find(data)
        while n:                    # Rebalance from the bottom up.
            n = self._rebalance(n).prev

    def remove(*args, **kwargs):
        """Disable remove() to keep the tree in balance."""
        raise NotImplementedError("remove() is disabled for this class")

    def _rebalance(self,n):
        """Rebalance the subtree starting at the specified node."""
        balance = AVL._balance_factor(n)
        if balance == -2:                                   # Left heavy
            if AVL._height(n.left.left) > AVL._height(n.left.right):
                n = self._rotate_left_left(n)                   # Left Left
            else:
                n = self._rotate_left_right(n)                  # Left Right
        elif balance == 2:                                  # Right heavy
            if AVL._height(n.right.right) > AVL._height(n.right.left):
                n = self._rotate_right_right(n)                 # Right Right
            else:
                n = self._rotate_right_left(n)                  # Right Left
        return n

    @staticmethod
    def _height(current):
        """Calculate the height of a given node by descending recursively until
        there are no further child nodes. Return the number of children in the
        longest chain down.
                                    node | height
        Example:  (c)                  a | 0
                  / \                  b | 1
                (b) (f)                c | 3
                /   / \                d | 1
              (a) (d) (g)              e | 0
                    \                  f | 2
                    (e)                g | 0
        """
        if current is None:     # Base case: the end of a branch.
            return -1           # Otherwise, descend down both branches.
        return 1 + max(AVL._height(current.right), AVL._height(current.left))

    @staticmethod
    def _balance_factor(n):
        return AVL._height(n.right) - AVL._height(n.left)

    def _rotate_left_left(self, n):
        temp = n.left
        n.left = temp.right
        if temp.right:
            temp.right.prev = n
        temp.right = n
        temp.prev = n.prev
        n.prev = temp
        if temp.prev:
            if temp.prev.value > temp.value:
                temp.prev.left = temp
            else:
                temp.prev.right = temp
        if n is self.root:
            self.root = temp
        return temp

    def _rotate_right_right(self, n):
        temp = n.right
        n.right = temp.left
        if temp.left:
            temp.left.prev = n
        temp.left = n
        temp.prev = n.prev
        n.prev = temp
        if temp.prev:
            if temp.prev.value > temp.value:
                temp.prev.left = temp
            else:
                temp.prev.right = temp
        if n is self.root:
            self.root = temp
        return temp

    def _rotate_left_right(self, n):
        temp1 = n.left
        temp2 = temp1.right
        temp1.right = temp2.left
        if temp2.left:
            temp2.left.prev = temp1
        temp2.prev = n
        temp2.left = temp1
        temp1.prev = temp2
        n.left = temp2
        return self._rotate_left_left(n)

    def _rotate_right_left(self, n):
        temp1 = n.right
        temp2 = temp1.left
        temp1.left = temp2.right
        if temp2.right:
            temp2.right.prev = temp1
        temp2.prev = n
        temp2.right = temp1
        temp1.prev = temp2
        n.right = temp2
        return self._rotate_right_right(n)

# Problem 4
def prob4():
    """Compare the build and search times of the SinglyLinkedList, BST, and
    AVL classes. For search times, use SinglyLinkedList.iterative_find(),
    BST.find(), and AVL.find() to search for 5 random elements in each
    structure. Plot the number of elements in the structure versus the build
    and search times. Use log scales where appropriate.
    """
    n = 2**np.arange(3,11)
    #n = np.arange(3,11)
    contents = []
    #randomList = SinglyLinkedList() #this is the subset of contents
    randomList = [] #this is the subset of contents
    findItems = [] #find 5 items out of the randomList

    #lists containing times
    linkedListCreateTimes = []
    BSTCreateTimes = []
    AVLCreateTimes = []

    locate_In_linked_List = []
    search_in_BST = []
    look_in_AVL = []


    with open("english.txt","r") as file:
        contents = file.readlines()
        """ this allows me get a few elements in the list if thats my preference
        for _ in range(5):
            contents.append(file.readline())
        """
    #this for loop is to repeat the experiment 8 times
    for j in range(len(n)):
        """
        this was a really cool idea that didn't fully work out

        #this loop will make a SinglyLinkedList called randomList of n[j] values
        #it will flag an error if it's not in the list
        #which is good because we don' want repeats

        while (randomList.size_of_List()) <= n[j]:

            try:
                input = random.choice(contents)
                n = randomList.iterative_find(input)
            except ValueError:
                randomList.append(input)
        """
        #this will make a randomList of n[j] values
        randomList = random.sample(contents, n[j])

        #make a linked list of randomList items and time it
        start = time.time()
        myList = SinglyLinkedList()
        for k in range(n[j]):
            myList.append(randomList[k])
        linkedListCreateTimes.append(time.time() - start)

        #make a new BST and time the creation times
        start = time.time()
        btree = BST()
        for k in range(n[j]):
            btree.insert(randomList[k])
        BSTCreateTimes.append(time.time() - start)

        #make a new AVL and time the creation times
        start = time.time()
        avy = AVL()
        for k in range(n[j]):
            avy.insert(randomList[k])
        AVLCreateTimes.append(time.time() - start)

        ###################### Search for values ###############################
        #make a subset of randomList of 5 values
        findItems = random.sample(randomList,5)

        #time the search in the singlylist for the values in findItems using iterative_find
        start = time.time()
        for k in range(5):
            myList.iterative_find(findItems[k])
        locate_In_linked_List.append(time.time() - start)

        #time the search in the BST
        start = time.time()
        for k in range(5):
            btree.find(findItems[k])
        search_in_BST.append(time.time() - start)

        #time the search in the AVL
        start = time.time()
        for k in range(5):
            avy.find(findItems[k])
        look_in_AVL.append(time.time() - start)

        #print('\n',"finished with experiment:", j, '\n')

    #DISPLAY THE VARIOUS TIMES OF THE FIND, AND SEARCH IN THE SinglyLinkedList,BST, AVL
    ax1 = plt.subplot(121)
    ax1.loglog(n,linkedListCreateTimes, "g--", label = "Linked List")
    ax1.loglog(n,BSTCreateTimes, "r:", label = "BST")
    ax1.loglog(n,AVLCreateTimes,"b-", label = "AVL")
    ax1.set_xscale("log", base=2)
    plt.xlabel("Size of Inputs")
    plt.ylabel("Time")
    plt.title("Creation times", fontsize = 18)
    plt.legend(loc = "upper left")

    #pg 74 of python essentials formatting for matplotlib
    ax2 = plt.subplot(122)
    ax2.loglog(n,locate_In_linked_List, "g--", label = "Linked List")
    ax2.loglog(n,search_in_BST, "r:", label = "BST")
    ax2.loglog(n,look_in_AVL,"b-", label = "AVL")
    ax2.set_xscale("log", base=2)
    plt.xlabel("Size of Inputs")
    plt.ylabel("Time")
    plt.title("Search times", fontsize = 18)
    plt.legend(loc = "upper left")

    #print("if my graphs look bad then use time.clock() instead of time.time()")
    #print(randomList)
    plt.show()

if __name__ == "__main__":
    pass

    #prob4()
    """
    btree = BST()
    #x = [5,7,6,2,1,8]
    x = [5,7,8,2,3,4,1] #edge case
    x = [4,2,10,1,3,5,11,6,15,9,14,16,7,12]
    x = [1,2,3,4,5,6]
    x = ["aa","ab","ac","ca"]
    #x = [5]
    #x = [6,5]
    #x = [4,5]
    #x = [5,3,7]
    #x = [7,5,6] #case 3b
    #x = [4,5,6] #case 3c

    for data in x:
        btree.insert(data)
    #print(btree.__str__())
    btree.draw()
    #btree.remove(1)
    #btree.draw()
    """


    """            TESTING LINKED LIST recursion
    mylist= SinglyLinkedList()
    x = ["people", "jerry", 9.0,17]
    #x = [7, 7, "jerry", 9.0] #remove the last element
    #x = [7] #remove one element
    for data in x:
        mylist.append(data)
    print(mylist.recursive_find(9).value)


    """
