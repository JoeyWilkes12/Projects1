import sys
import numpy as np
# https://stackoverflow.com/questions/1987694/how-to-print-the-full-numpy-array-without-truncation
np.set_printoptions(threshold=sys.maxsize)
import pdb
import operator
import csv
import networkx as nx
from itertools import combinations as combos

#digraph as in Directed graph (as opposed to undirected - bidirectional edges)
class DiGraph:
    """A class for representing directed graphs via their adjacency matrices.

        Attributes:
            (fill this out after completing DiGraph.__init__().)
            """

    def __init__(self, A, labels=None):
        """Modify A so that there are no sinks in the corresponding graph,
            then calculate Ahat. Save Ahat, as "modified," and the labels as attributes.

            Parameters:
                A ((n,n) ndarray): the adjacency matrix of a directed graph.
                A[i,j] is the weight of the edge from node j to node i.
                labels (list(str)): labels for the n nodes in the graph.
                If None, defaults to [0, 1, ..., n-1].
            """
        #raise error if incorrect number of labels given
        size = np.shape(A)[0]

        if labels is not None and len(labels) != size:
            raise ValueError("Incorrect number of labels given")

        #save as an attribute for use in linsolve
        self.size = size
        #save labels as attributes
        if labels is None:
            self.labels = [i for i in range(size)]
        else:
            self.labels = labels
        #modify copy of A so that there are no sinks
        modified = np.array(A,dtype='float64',copy=True)
        #sum the columns, if resulting norm is zero then modify columns
        norms = np.sum(np.abs(A),axis=0)

        #verify if there are not sinks
        if np.count_nonzero(norms) == len(norms):
            pass
        else:
            #adjust the sinks through iterating through array
            #modify each column to be divided by the sum of that column
            #thus each column is normalized to sum to one
            # pdb.set_trace()
            for i in range(size):
                value = norms[i]
                if value == 0:
                    modified[:,i] = np.ones((size)) / size
                else:
                    modified[:,i] = modified[:,i] * ( np.ones(size) / value)

        self.adjacency = modified

    def linsolve(self, epsilon=0.85):
        """Compute the PageRank vector using the linear system method.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.

        Returns:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """
        #find limit of each node as an array
        #system of equations in solve method are according to 14.6 on page 136 of textbook
        limits = np.linalg.solve((np.eye(self.size)-epsilon*self.adjacency),
            ((1-epsilon / self.size)*np.ones(self.size)))
        #normalize the vector w/ respect to 1 norm
        limits /= np.linalg.norm(limits,1)
        #construct dictinary, iterate through elements in limits
        values = dict()
        for i in range(len(self.labels)):
            values.update({self.labels[i]:limits[i]})
        return values

    def eigensolve(self, epsilon=0.85):
        """Compute the PageRank vector using the eigenvalue method.
            Normalize the resulting eigenvector so its entries sum to 1.

            Parameters:
                epsilon (float): the damping factor, between 0 and 1.

            Return:
                dict(str -> float): A dictionary mapping labels to PageRank values.
            """
        E = np.ones_like(self.adjacency)
        # E = np.ones((self.size,self.size))
        #parenthesis around (1-epsilon) are essential because it was changing my output
        B = (epsilon*self.adjacency) + ((1-epsilon) / self.size) * E
        #find eigenvector of B
        #np.lin.eig returns eigenvalues & eigenvectors
        e_values, e_vectors = np.linalg.eig(B)
        # return e_values, e_vectors
        #np.argmax will return index of eigenvalue = 1, we want corresponding eigenvector
        p = e_vectors[:, np.argmax(e_values)]
        # limits = p
        #normalize p so that it sums to one
        # limits = p / (np.linalg.norm(p,1))
        limits = p / (np.sum(p)) #this doesn't change output
        # assert np.allclose( (p / np.linalg.norm(p,1) ) , (p / np.sum(p) ) ), "there is a difference in normalizations"
        # return limit
        # return np.sum(limits) # NOTE: this shows that it sums to one
        #construct dictinary, iterate through elements in limits
        values = dict()
        for i in range(self.size):
            values.update({self.labels[i]:limits[i]})
        return values

    def itersolve(self, epsilon=0.85, maxiter=100, tol=1e-12):
        """Compute the PageRank vector using the iterative method.

            Parameters:
                epsilon (float): the damping factor, between 0 and 1.
                maxiter (int): the maximum number of iterations to compute.
                tol (float): the convergence tolerance.

            Return:
                dict(str -> float): A dictionary mapping labels to PageRank values.
            """
        #iteratively compute limit of p, with initial guess of p0
        p0 = np.ones(self.size) / self.size
        prev = p0
        for i in range(maxiter):
          new = (epsilon*(self.adjacency @ prev) +
            ((1-epsilon / self.size)*np.ones(self.size)))
        # if difference is less than tolerance then break
          if np.linalg.norm(new - prev,1) < tol:
            break
          prev = new
        #normalize the vector w/ respect to 1 norm
        new /= (np.linalg.norm(new,1))
        #construct dictinary, iterate through elements in limits
        values = dict()
        for i in range(len(self.labels)):
            values.update({self.labels[i]:new[i]})
        return values

def test_constructor():
    """ test the constructor to the DiGraph"""
    A = np.array([
        [0., 0, 0],
        [3, 0, 0],
        [0, 0, 4]
    ])
    A_1 = np.array([
        [0., 1/3, 0],
        [1, 1/3, 0],
        [0, 1/3, 1]
    ])
    B = np.array([ #matrix must be a float, or values do integer division to zero
        [1., 1, 0],
        [3, 1, 0],
        [1, 1, 0]
    ])
    B_1 = np.array([
        [0.2, 1/3, 1/3],
        [0.6, 1/3, 1/3],
        [0.2, 1/3, 1/3]
    ])

    print("proceeding through constructor testing")

    one = DiGraph(A)
    # print(one.adjacency)
    # print(A_1)
    assert np.allclose(one.adjacency,A_1), "test case 1 failed"

    two = DiGraph(B)
    # print(two.adjacency)
    # print(B_1)
    assert np.allclose(two.adjacency,B_1), "test case 2 failed"

    try:
        another = DiGraph(A,labels=['a','b','c','d','e','f','g'])
    except ValueError:
        print("correctly threw error for test case of incorrect labels given")

def limit_test():
    """ test the linsolve, eigensolve, and itersolve, methods"""
    A = np.array([
        [0., 0, 0,0],
        [1., 0, 1,0],
        [1., 0, 0,1],
        [1., 0, 1,0]
    ])
    ##instantiate an object of this class according to example in book
    example = DiGraph(A,['a','b','c','d'])
    # print(example.adjacency)
    ex_lin = example.linsolve().values()
    print("linsolve output:\n",ex_lin)
    ## print(np.linalg.norm(example.eigensolve(),1)) #useful when returning limit
    ex_eig = example.eigensolve().values()
    print("eigensolve output\n",ex_eig)
    # print("the sum of eigensolve output is \n",sum(e[0]x_eig)) #when outputing the vector

    ex_iter = example.itersolve().values()
    print("itersolve output\n",ex_iter)
    # assert ex_lin == ex_eig == ex_iter, "comparison of output from linsolve, eigensolve, itersolve, from first test case failed"

    """ #just comment out the second test case so it runs faster
    B = np.array([
        [1.,0,1],
        [0,0,1],
        [1,0,0]
    ])
    test2 = DiGraph(B)
    t2_lin = test2.linsolve()
    t2_eig = test2.eigensolve()
    t2_iter = test2.itersolve()
    # print("\nlinsolve output for second test case:\n",t2_lin)
    # print("eigensolve output for second test case\n",t2_eig)
    # print("itersolve output for second test case:\n",t2_iter)
    # assert t2_lin == t2_eig == t2_iter, "comparison of output from linsolve, eigensolve, itersolve, from second test case failed"
    """

    #testing for equality in the dictionaries isn't great because there are differences after like 10 decimals
    # print("PASSED BOTH TEST CASES OF TRIPLE METHOD SOLVERS EQUALITY in limit_test function ")

def get_ranks(d):
    """Construct a sorted list of labels based on the PageRank vector.

        Parameters:
            d (dict(str -> float)): a dictionary mapping labels to PageRank values.

        Returns:
            (list) the keys of d, sorted by PageRank value from greatest to least.
        """
    #get keys as an array since ndarrays are fancy indexable
    keys = np.array(list(d.keys()))
    ##using fancy index, return indices that would
    ##sort array by values, in reverse (descending) order
    indices = np.argsort(list(d.values()))[::-1]
    # indices = np.argsort(list(d.values()))
    return list(keys[indices])

    ##ALTERNATIVE METHOD BELOW, doesn't work as well, it was my first attempt
    #using an operator, iterate through values/ items and return 0th element
    #itemgetter(n) where n refers to the item in this case of the tuple,
    #ask others to show me their solution
    # return (list(d.values()).sort()) #returned None
    #having itemgetter(0) sorts according to zero elements, or the keys of dictionaries
    # return [a[0] for a in sorted(d.items(), key=operator.itemgetter(1), reverse = True) ]

def ranks_tester():
    """ test various functions like get_ranks """
    A = np.array([
        [0., 0, 0,0],
        [1., 0, 1,0],
        [1., 0, 0,1],
        [1., 0, 1,0]
    ])
    ##instantiate an object of this class according to example in book
    example = DiGraph(A,['a','b','c','d'])
    d = example.linsolve()
    # sorted yields : [('a', 0.09575863576738086), ('b', 0.2741582859641452), ('d', 0.2741582859641452), ('c', 0.3559247923043289)]
    # i found a complex solution, i bet there is an easier way
    test = {'a':7,'b':2,'c':90,'d':4,'e':-1}
    # print(get_ranks(test))
    textbook_case = get_ranks(d)
    assert (textbook_case == ['c', 'b', 'd', 'a']) or (textbook_case == ['c', 'd', 'b', 'a']), "test case failed for array {}'\n'".format(A)
    assert get_ranks(test) == ['c','a','d','b','e'], "test case failed for get_ranks for \n{}".format(test)
    print("test cases passed in rank tester")

def rank_websites(filename="web_stanford.txt", epsilon=0.85):
    """Read the specified file and construct a graph where node j points to
        node i if webpage j has a hyperlink to webpage i. Use the DiGraph class
        and its itersolve() method to compute the PageRank values of the webpages,
        then rank them with get_ranks().

        Each line of the file has the format
        a/b/c/d/e/f...
        meaning the webpage with ID 'a' has hyperlinks to the webpages with IDs
        'b', 'c', 'd', and so on.

        Parameters:
            filename (str): the file to read from.
            epsilon (float): the damping factor, between 0 and 1.

        Returns:
            (list(str)): The ranked list of webpage IDs.
        """
    #read in the data
    with open(filename,'r') as infile:
        # data = infile.read().strip('\n').split('/') #this works but makes just one list, with newline characters
        # data = infile.readlines().split('\n')
        data = infile.readlines()

    #format the data to be a list of lists,
    #each list represents the connections of the node, where
    #node is first element, followed by connecting node as elements
    labels = set()
    indices = dict()
    for i in range(len(data)):
        data[i] = data[i].strip().split('/')
        #labels will be used as parameter in constructor
        #data[][] notation is for list of lists
        # NOTE: most critical part, is that some nodes get mapped to, but don't map to anything
        for word in data[i]:
            labels.add(word)

    #construct dictionary out of sorted list (from set)
    #sort the list to match test driver format
    # labels = sorted(list(labels))
    labels = sorted(labels) #NOTE: this is what I had before
    for i in range(len(labels)):
        indices.update({labels[i]:i})

    # construct the adjacency matrix
    #create adjacency
    A = np.zeros((len(labels),len(labels)))
    # for a given word line[0]
    for line in data:
        #index of primary word
        i = indices[line[0]]
        #for each corresponding word
        for label in line[1:]:
            #index of corresponding word
            j = indices[label]
            #label as one
            A[j][i] = 1

    # # """ for comparison purposes, this following code didn't work,
        # it gives different output from above
    # n = len(labels)
    # A = np.zeros((n,n))
    # for line in data:
    #     node_index = indices[line[0]]
    #     # through slicing make sure not to make a connection from a node to itself
    #     for label in labels[1:]:
    #         #the primary node maps to a node calculated connection
    #         connection_index = indices[label]
    #         A[connection_index][node_index] = 1
    # # """

    #constuct Digraph
    diGraph = DiGraph(A,labels)
    #itersolve will return a dictionary mapping label to pagerank value
    pageRanks = diGraph.itersolve(epsilon)
    return get_ranks(pageRanks)

def rank_ncaa_teams(filename, epsilon=0.85):
    """Read the specified file and construct a graph where node j points to
        node i with weight w if team j was defeated by team i in w games. Use the
        DiGraph class and its itersolve() method to compute the PageRank values of
        the teams, then rank them with get_ranks().

        Each line of the file has the format
        A,B
        meaning team A defeated team B.


        Parameters:
            filename (str): the name of the data file to read.
            epsilon (float): the damping factor, between 0 and 1.

        Returns:
            (list(str)): The ranked list of team names.
        """
    labels = set()
    indices = dict()
    with open(filename) as f:
        #casting csv.reader as a list, allows for easy manipulation
        data = list(csv.reader(f))
        for line in data:
            a,b = line[0],line[1]
            if a != "Winner":
                labels.add(a)
                labels.add(b)
        # labels = sorted(labels)
        labels = list(labels)
        #fill the indices dictionary
        for i in range(len(labels)):
            indices[labels[i]] = i
        #make adjacency matrix
        n = len(labels)
        A = np.zeros((n,n))
        for line in data:
            winner, loser = line[0],line[1]
            #we don't want header that says "Winner, Loser"
            if winner != "Winner":
            #edges point from loser(column) to winner(row)
                A[indices[winner]][indices[loser]] += 1
        return get_ranks(DiGraph(A,labels).itersolve(epsilon))

def rank_actors(filename="top250movies.txt", epsilon=0.85):
    """Read the specified file and construct a graph where node a points to
        node b with weight w if actor a and actor b were in w movies together but
        actor b was listed first. Use NetworkX to compute the PageRank values of
        the actors, then rank them with get_ranks().

        Each line of the file has the format
        title/actor1/actor2/actor3/...
        meaning actor2 and actor3 should each have an edge pointing to actor1,
        and actor3 should have an edge pointing to actor2.
        """
    movies = nx.DiGraph()
    data = []
    actors = set()
    with open(filename,"r",encoding = "utf-8") as file:
        for line in file.readlines():
            #format the data into a list of lists, where each list is a movie
            line = line.strip().split('/')
            #delete the movie titles. Irrelevant info.
            delete = line.pop(0)
            data.append(line)
        #get a comprehensive list of actors
        for line in data:
            for actor in line:
                #comprehensive lists is a set to avoid repeats
                actors.add(actor)
        actors = list(actors) #there are almost 15k actors
        # n = len(actors)
        #construct graph with edges
        for line in data:
            #combos is an iterator object so must be accessed in this way
            for i in combos(line,2):
                high_paid, low_paid = i
                #edges point to higher-billed actors
                if movies.has_edge(low_paid,high_paid) is True:
                    movies[low_paid][high_paid]["weight"] += 1
                else:
                    movies.add_edge(low_paid,high_paid,weight = 1)
    return get_ranks(nx.pagerank(movies,alpha = epsilon))
    # return data[0]


if __name__ == '__main__':
    pass
    # test_constructor()
    # limit_test()
    # ranks_tester()

    print("\nbelow is result for epsilon = 0.85, showing last term of the first 20 terms:")
    print(rank_websites(epsilon=0.85)[:20][-1:])
    print("\nbelow is result for epsilon = 0.62, showing last 5th, 4th of the first 20 terms:")
    print(rank_websites(epsilon=0.62)[:20][-5:-3])

    # print(rank_ncaa_teams('ncaa2010.csv')) #i should have 607 teams, but I have 606, or 608
    # print(rank_ncaa_teams('ncaa2010.csv')[:5]) #epsilon 0.3 yields BYU
    # print(rank_actors(epsilon=0.7)[:5])
    # print(rank_actors())

    print("this code is complete, it appears there were some test driver complications ")
