import numpy as np
from scipy import linalg as la
#math function is used in test_search function
import math
from scipy.spatial import KDTree as skt #s for scipy, kt for ktree
from scipy import stats

def exhaustive_search(X, z):
    """Solve the nearest neighbor search problem with an exhaustive search.

    Parameters:
        X ((m,k) ndarray): a training set of m k-dimensional points.
        z ((k, ) ndarray): a k-dimensional target point.

    Returns:
        ((k,) ndarray) the element (row) of X that is nearest to z.
        (float) The Euclidean distance from the nearest neighbor to z.
    """
    #broadcast each row of X subtract element wise of z
    Norms = la.norm(X - z, axis = 1)
    #argmin will return the index
    return X[Norms.argmin()], Norms.min()

def test_search():
    """ Test exhaustive search """
    #each list is the coordinants of the point
    A = np.array([[5+2,4+3,math.sqrt(8)+7],[10+2,4+3,math.sqrt(5)+7]])
    #this is the point of the target
    B = np.array([2,3,7])
    # NORM is [  7.  11.]
    print(A, A-B, sep="\n\n")
    row, value = exhaustive_search(A,B)
    print(row, value, sep = "\n\n")

class KDTNode:
    """ Parent Class of KDT that will construct the nodes in the KDT

    Attributes:
        constructor - initializes the children, value, and pivot value of Node
    """

    def __init__(self,x):
        """Initialize the node with array or raise TypeError."""
        if isinstance(x, np.ndarray):
            self.value = x
            self.left = None
            self.right = None
            self.pivot = None
        else:
            raise TypeError("invalid type of input")

class KDT:
    """A k-dimensional binary tree for solving the nearest neighbor problem.

    Attributes:
        root (KDTNode): the root node of the tree. Like all other nodes in
            the tree, the root has a NumPy array of shape (k,) as its value.
        k (int): the dimension of the data in the tree.
    """

    def __init__(self):
        """Initialize the root and k attributes."""
        self.root = None
        self.k = None

    def find(self, data):
        """Return the node containing the data. If there is no such node in
        the tree, or if the tree is empty, raise a ValueError.
        """
        def _step(current):
            """Recursively step through the tree until finding the node
            containing the data. If there is no such node, raise a ValueError.
            """
            if current is None:                     # Base case 1: dead end.
                raise ValueError(str(data) + " is not in the tree")
            elif np.allclose(data, current.value):
                return current                      # Base case 2: data found!
            elif data[current.pivot] < current.value[current.pivot]:
                return _step(current.left)          # Recursively search left.
            else:
                return _step(current.right)         # Recursively search right.

        # Start the recursive search at the root of the tree.
        return _step(self.root)

    # Problem 3
    def insert(self, data):
        """Insert a new node containing the specified data.

        Parameters:
            data ((k,) ndarray): a k-dimensional point to insert into the tree.

        Raises:
            ValueError: if data does not have the same dimensions as other
                values in the tree.
        """
        #Initialize the root
        if self.root is None:
            newNode = KDTNode(data)
            self.root = newNode
            newNode.pivot = 0
            self.k = len(data)
        else:
            #throw an error if the array is incorrect size
            if len(data) != self.k: #self.root.k doesn't work
                raise ValueError("array is incorrect size")
            else:
                #try statement will verify duplicate possibility
                try:
                    self.find(data)
                    raise ValueError("Duplicate found")
                #Except means there is not a duplicate
                except ValueError:
                    newNode = KDTNode(data)
                    current = self.root

                    def assign_pivot(current,newNode):
                        """ Check the pivot value of parent of newNode which is current node,
                        assign pivot value appropriately

                        """
                        if current.pivot < self.k - 1:
                            newNode.pivot = current.pivot + 1
                        else:
                            newNode.pivot = 0

                    def insert_step(current,newNode):
                        """ recursively step through the tree and locate
                        correct position for new node
                        """
                        pivot = current.pivot
                        #compare data to be inserted to current node at pivot value
                        if  newNode.value[pivot] < current.value[pivot]:
                            if current.left is None:
                                current.left = newNode
                                assign_pivot(current,newNode)
                            else:
                                current = current.left
                                insert_step(current,newNode)
                        else:
                            if current.right is None:
                                current.right = newNode
                                assign_pivot(current,newNode)
                            else:
                                current = current.right
                                insert_step(current,newNode)

                    insert_step(current,newNode)

    def query(self, target):
        """Find the value in the tree that is nearest to z.

        Parameters:
            z ((k,) ndarray): a k-dimensional target point.

        Returns:
            ((k,) ndarray) the value in the tree that is nearest to z.
            (float) The Euclidean distance from the nearest neighbor to z.
        """
        minimal_Distance  = -1 #this should alert an error

        def KDSEARCH(current,nearest,distance):
            """ recursively test possible nodes and the distance
            from the target

            Returns:
                ((k,) ndarray) the value in the tree that is nearest to z.
                (float) The Euclidean distance from the nearest neighbor to z.

            """
            #the nodes came to a dead end
            if current is None:
                return nearest, distance
            compare_Array = current.value
            i = current.pivot
            normCompare = la.norm(compare_Array-target)
            # check if current is closer to target than nearest
            if normCompare < distance:
                nearest = current
                distance = normCompare
            #search to the left
            if target[i] < compare_Array[i]:
                nearest, distance = KDSEARCH(current.left,nearest,distance)
                #search to the right if needed
                if target[i] + distance >= compare_Array[i]:
                    nearest, distance = KDSEARCH(current.right,nearest, distance)
            else:
                #this else statement is searching to the right
                nearest, distance = KDSEARCH(current.right,nearest, distance)
                #search to the left if needed
                if target[i] - distance <= compare_Array[i]:
                    nearest, distance = KDSEARCH(current.left,nearest,distance)
            return nearest,distance

        initial_Distance = la.norm(self.root.value - target)
        node, minimal_Distance = KDSEARCH(self.root,self.root,initial_Distance)
        return node.value, minimal_Distance

    def __str__(self):
        """String representation: a hierarchical list of nodes and their axes.

        Example:                           'KDT(k=2)
                    [5,5]                   [5 5]   pivot = 0
                    /   \                   [3 2]   pivot = 1
                [3,2]   [8,4]               [8 4]   pivot = 1
                    \       \               [2 6]   pivot = 0
                    [2,6]   [7,5]           [7 5]   pivot = 0'
        """
        if self.root is None:
            return "Empty KDT"
        nodes, strs = [self.root], []
        while nodes:
            current = nodes.pop(0)
            strs.append("{}\tpivot = {}".format(current.value, current.pivot))
            for child in [current.left, current.right]:
                if child:
                    nodes.append(child)
        return "KDT(k={})\n".format(self.k) + "\n".join(strs)

def test_KDT():
    """ test KDT class """


    #it's better to write the test cases before writing the function
    A = np.array([[50,2,3,4],[25,50,2,3],[75,50,2,3],
        [19,45,17,3],[41,17,20,3],[60,60,25,3]])
    A = np.array([[50,2,3,4],[75,50,2,3],[60,60,25,3],
        [25,50,2,3],[19,45,17,3],[41,17,20,3],[17,18,19,20]])
    # ragged on purpose; NumPy >= 1.24 requires dtype=object for ragged arrays
    B = np.array([[5,5],[3,2],[8,4],[2,6,5]], dtype=object) #trivial case compared to C
    #C = np.array([[3,1,4],[1,2,7],[4,3,5],[2,0,3],[2,4,5],[6,1,4],[1,4,3],[0,5,7],[5,2,5]])
    C = np.array([[3,1,4],[4,3,5],[6,1,4],[5,2,5],[1,2,7],[2,4,5],[0,5,7],[1,4,3],[2,0,3]])

    myTree = KDT()
    #Test case with C: PASSED in both cases
    #Test case with A: SUCCESS
    #Test case with B: CORRECT ERROR

    LIST_ARRAY = [A,C,B,]
    for j in A:
        myTree = KDT()
        for i in range(len(j)):
            myTree.insert(j[i])
            if j == B:
                pass
                #print(myTree)
        #print(myTree)

def query_Test():
    """ Test the query method of KDT """
    A = np.array([[5+2,4+3,math.sqrt(8)+7],[10+2,4+3,math.sqrt(5)+7]])
    #this is the point of the target
    B = np.array([2,3,7])
    # NORM is [  7.  11.]
    testCase = np.array([0,0,0])

    C = np.array([[3,1,4],[4,3,5],[6,1,4],[5,2,5],[1,2,7],[2,4,5],[0,5,7],[1,4,3],[2,0,3],[1,1,1]])


    myTree = KDT()
    for i in range(len(C)):
        myTree.insert(C[i])

    print(myTree.query(testCase))

# Problem 5: Write a KNeighborsClassifier class.
class KNeighborsClassifier:
    """
    Evaluate the distance of K elements, and the majority
    classification will dictate the prediction of the label for the target
    """

    def __init__(self,n_neighbors):
        """ Constructor receives the number of neighbors whose labels
        will be included in the consideration for determining the label
        of the target - this integer is stored as an integer
        """
        self.tree = None
        self.labels = None
        self.n = n_neighbors

    def fit(self,training_Array,labels):
        """ Accept a (m,k) array that will fit data into KDtree
        and accept a 1-d array of training labels
        """
        self.tree = skt(training_Array)
        self.labels = labels

    def predict(self,vector):
        """ Accept a 1-dimensional array (the target) and
        based upon the KDtree - based upon the desired number of nearest
        neighbors, predict the label of the target

        return
        """
        distances, indices = self.tree.query(vector,self.n)
        #mode, frequency = stats.mode(self.labels[indices])[0][0]

        # atleast_1d handles n_neighbors=1, where query() returns a scalar index;
        # keepdims=False makes mode() return a scalar (SciPy >= 1.11 behavior)
        return stats.mode(self.labels[np.atleast_1d(indices)], keepdims=False)[0]

def test_5():
    """ test KNeighborsClassifier """
    #if it really doesn't work then I'll go back and try to debug it.

    data = np.random.random((50,2))
    target = np.random.random(5)
    A = np.array(["a","b"])
    labels = np.random.choice(A,10)
    print(labels)

# Problem 6
def prob6(n_neighbors, filename="mnist_subset.npz"):
    """Extract the data from the given file. Load a KNeighborsClassifier with
    the training data and the corresponding labels. Use the classifier to
    predict labels for the test data. Return the classification accuracy, the
    percentage of predictions that match the test labels.

    Parameters:
        n_neighbors (int): the number of neighbors to use for classification.
        filename (str): the name of the data file. Should be an npz file with
            keys 'X_train', 'y_train', 'X_test', and 'y_test'.

    Returns:
        (float): the classification accuracy.
    """
    data = np.load("mnist_subset.npz")
    # Training data
    X_train = data["X_train"].astype(float)
    # Training labels
    y_train = data["y_train"]
    # Test data
    X_test = data["X_test"].astype(float)
    # Test labels
    y_test = data["y_test"]

    # create a KDTree
    classifier = KNeighborsClassifier(n_neighbors)
    #train the data
    classifier.fit(X_train,y_train)
    sum = 0
    #test each image and verify if it's correct
    for i in range(500):
        outcome = classifier.predict(X_test[i])
        answer = y_test[i]
        #print(answer,outcome, sep='\n')
        if  answer == outcome:
            sum += 1
    return sum / 500

def Sixth_problem_Test():
    """ test the sixth problem """
    accuracy = []
    for i in range(1,10):
        accuracy.append(prob6(i))
        print(accuracy)
    print(accuracy)


if __name__ == "__main__":
    pass
    #Sixth_problem_Test()
    #test_5()
    #test_KDT()
    #query_Test()

    #print("I could test it for 2 dimensions and assume it works for greater")

    #print("does insert_step have access to newNode w/out passing as parameter")
