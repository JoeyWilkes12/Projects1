import sympy as sy
from matplotlib import pyplot as plt
import numpy as np
import time
from autograd import numpy as anp
from autograd import grad
from autograd import elementwise_grad
import pdb
import random

def prob1():
    """Return the derivative of (sin(x) + 1)^sin(cos(x)) using SymPy."""
    x = sy.symbols('x')
    #construct the function
    f = (sy.sin(x)+1)**(sy.sin(sy.cos(x)))
    #sy.diff will perform the differentiation, as opposed to other methods
    #lambdify f with "numpy" attribute
    prime = sy.lambdify(x,sy.diff(f,x),"numpy")
    #return derivative function handle
    return prime

def fdq1(f, x, h=1e-5):
    """Calculate the first order forward difference quotient of f at x."""
    #see textbook for formula
    return (f(x+h)-f(x))/ h

def fdq2(f, x, h=1e-5):
    """Calculate the second order forward difference quotient of f at x."""
    #see textbook for formula
    #the parenthesis in the denominator are crucial
    return (-3*f(x)+4*f(x+h)-f(x+2*h)) / (2*h)

def bdq1(f, x, h=1e-5):
    """Calculate the first order backward difference quotient of f at x."""
    #see textbook for formula
    return (f(x) - f(x-h)) / (h)

def bdq2(f, x, h=1e-5):
    """Calculate the second order backward difference quotient of f at x."""
    #see textbook for formula
    return (3*f(x) - 4*f(x-h) + f(x-2*h)) / (2*h)

def cdq2(f, x, h=1e-5):
    """Calculate the second order centered difference quotient of f at x."""
    #see textbook for formula
    return (f(x+h) - f(x-h)) / (2*h)

def cdq4(f, x, h=1e-5):
    """Calculate the fourth order centered difference quotient of f at x."""
    #see textbook for formula
    return (f(x- 2*h) - 8*f(x-h) + 8*f(x+h) - f(x+2*h) )  / (12*h)

def test_quotients():
    """ test the prob 1 and graph it as well as outputs from
        fdq1,fdq2, bdq1, bdq2, cdq2, cdq4 functions"""
    domain = np.linspace(-1*np.pi,np.pi,int(1e3))
    #lambdify it to evaluate on domain
    x = sy.symbols('x')
    f = sy.lambdify(x,(sy.sin(x)+1)**(sy.sin(sy.cos(x))),"numpy")

    #define outputs in a list so it's easy to make subplots
    approximations = [
        f(domain),prob1()(domain),
        fdq1(f,domain),fdq2(f,domain),
        bdq1(f,domain),bdq2(f,domain),
        cdq2(f,domain),cdq4(f,domain)]

    titles = [
        r"$(sin(x) + 1)^{sin(cos(x))}$","derivative",
        "1st Ord. Forward","2nd Ord. Forward",
        "1st Ord. Backward","2nd Ord. Backward",
        "2nd Ord. Centered","4th Ord. Centered"]
    #the titles dictionary could be used to map the axes indices to either the approximations or the title
    # titles = {(0,1): , (): , (): , (): ,
    #         (): , (): , (): , (): }

    #plot the subplots for visual comparison

    fig, axes = plt.subplots(4,2)
    fig.suptitle("Numerical Differentiation with Quotients")
    axes[0,0].plot(domain,approximations[0],'-b',label=titles[0])
    axes[0,0].set_title(titles[0])
    axes[0,1].plot(domain,approximations[1],'-b',label=titles[1])
    axes[0,1].set_title(titles[1])
    axes[1,0].plot(domain,approximations[2],label=titles[2])
    axes[1,0].set_title(titles[2])
    axes[1,1].plot(domain,approximations[3],label=titles[3])
    axes[1,1].set_title(titles[3])
    #the titles dictionary could be used to map the axes indices to either the approximations or the title
    # titles = {(0,1): , (): , (): , (): ,
    #         (): , (): , (): , (): }

    #plot the subplots for visual comparison

    # for i in range(4):
    #     for j in range(2):
    #         axes[i,j].plot(domain,approximations[i])
    #         axes[i,j].set_title(titles[i],'-b')le(titles[3])
    axes[2,0].plot(domain,approximations[4],label=titles[4])
    axes[2,0].set_title(titles[4])
    axes[2,1].plot(domain,approximations[5],label=titles[5])
    axes[2,1].set_title(titles[5])
    axes[3,0].plot(domain,approximations[6],label=titles[6])
    axes[3,0].set_title(titles[6])
    axes[3,1].plot(domain,approximations[7],label=titles[7])
    axes[3,1].set_title(titles[7])
    plt.tight_layout()

    #put the labelled axis on the y=0 axis.
    # ax = plt.gca()
    # ax.spines["bottom"].set_position("zero")
    plt.show()

def prob3(x0):
    """Let f(x) = (sin(x) + 1)^(sin(cos(x))). Use prob1() to calculate the
        exact value of f'(x0). Then use fdq1(), fdq2(), bdq1(), bdq2(), cdq1(),
        and cdq2() to approximate f'(x0) for h=10^-8, 10^-7, ..., 10^-1, 1.
        Track the absolute error for each trial, then plot the absolute error
        against h on a log-log scale.

        Parameters:
            x0 (float): The point where the derivative is being approximated.
            """
    #lambdify f to evaluate at point
    x = sy.symbols('x')
    f = sy.lambdify(x,(sy.sin(x)+1)**(sy.sin(sy.cos(x))),"numpy")
    exact = prob1()(x0)
    #domain
    h_values = np.logspace(-8,0,9)
    #record the values of the absolute error for each trial, then cast as arrays to plot
    #iterate through all h_values
    #my first attempt was to do 2 for loops, this is much better
    fdq1_list = np.array([np.abs(exact - fdq1(f,x0,h)) for h in h_values])
    fdq2_list = np.array([np.abs(exact - fdq2(f,x0,h)) for h in h_values])
    bdq1_list = np.array([np.abs(exact - bdq1(f,x0,h)) for h in h_values])
    bdq2_list = np.array([np.abs(exact - bdq2(f,x0,h)) for h in h_values])
    cdq2_list = np.array([np.abs(exact - cdq2(f,x0,h)) for h in h_values])
    cdq4_list = np.array([np.abs(exact - cdq4(f,x0,h)) for h in h_values])
    #plot the values
    plt.loglog(h_values,fdq1_list,marker=".",color="blue",label="Order 1 Forward")
    plt.loglog(h_values,fdq2_list,marker=".",color="orange",label='Order 2 Forward')
    plt.loglog(h_values,bdq1_list,marker=".",color="green",label='Order 1 Backward')
    plt.loglog(h_values,bdq2_list,marker=".",color="red",label='Order 2 Backward')
    plt.loglog(h_values,cdq2_list,marker=".",color="purple",label='Order 2 Centered')
    plt.loglog(h_values,cdq4_list,marker=".",color="brown",label='Order 4 Centered')
    plt.legend(loc="upper left")
    plt.xlabel("h")
    plt.ylabel("Absolut Error")
    plt.show()

def prob4():
    """The radar stations A and B, separated by the distance 500m, track a
        plane C by recording the angles alpha and beta at one-second intervals.
        Your goal, back at air traffic control, is to determine the speed of the
        plane.

        Successive readings for alpha and beta at integer times t=7,8,...,14
        are stored in the file plane.npy. Each row in the array represents a
        different reading; the columns are the observation time t, the angle
        alpha (in degrees), and the angle beta (also in degrees), in that order.
        The Cartesian coordinates of the plane can be calculated from the angles
        alpha and beta as follows.

        x(alpha, beta) = a tan(beta) / (tan(beta) - tan(alpha))
        y(alpha, beta) = (a tan(beta) tan(alpha)) / (tan(beta) - tan(alpha))

        Load the data, convert alpha and beta to radians, then compute the
        coordinates x(t) and y(t) at each given t. Approximate x'(t) and y'(t)
        using a forward difference quotient for t=7, a backward difference
        quotient for t=14, and a centered difference quotient for t=8,9,...,13.
        Return the values of the speed at each t.
        """
    data = np.load("plane.npy",'r')

    """ this is how the data looks in degrees

        [[ 7.   56.25 67.54]      index = 0
        [ 8.   55.53 66.57]       index = 1
        [ 9.   54.8  65.59]
        [10.   54.06 64.59]
        [11.   53.34 63.62]
        [12.   52.69 62.74]
        [13.   51.94 61.72]
        [14.   51.28 60.82]]      index = 7 for t=14

        the second two columns converted to radians

        [[0.9817477 1.17879538]
        [0.96918133 1.16186568]
        [0.95644043 1.14476146]
        [0.94352499 1.12730816]
        [0.93095862 1.11037847]
        [0.91961398 1.09501957]
        [0.90652401 1.07721721]
        [0.89500484 1.06150925]]

        """
    #convert degrees to radians, which leaves only 2 columns of radians
    rad_data = np.deg2rad(data[:,1:])
    #note that rad_data only has 2 columns thats why slicing index is strange
    alpha = rad_data[:,:1]
    beta = rad_data[:,1:]
    time = data[:,:1]

    # I want the derivatives in an  array so I can find norm along rows
    # we need to vertically stack the arrays

    #equations for converting are found in the text
    x_pos = (500 * np.tan(beta)) / (np.tan(beta) - np.tan(alpha))
    y_pos = (500 * np.tan(beta) * np.tan(alpha)) / (np.tan(beta) - np.tan(alpha))
    velocities = np.array([(x_pos[1] - x_pos[0])[0],(y_pos[1] - y_pos[0])[0]])

    #calculate times 8-13 using 2nd order centered difference quotient
    #where x is the index in x_pos, add 7 to get the time
    #set h =1 since time intervals is one
    h = 1
    #the [0] is an extractor because x_pos returns an array
    #it could be more efficient if I calculated everything in the for loop
        #and then stacked it once but I'm pressed for time - computational time isn't a constraint here
    for x in range(1,7):
        a = [(x_pos[x+h] - x_pos[x-h])[0] / (2), (y_pos[x+h] - y_pos[x-h])[0] / (2) ]
        velocities = np.vstack((velocities, a))
    #calculate final velocity using backward difference quotient
    velocities = np.vstack((velocities,
        [ (x_pos[7] - x_pos[6])[0],(y_pos[7] - y_pos[6])[0]] ))

    # print(velocities)
    speeds = np.linalg.norm(velocities,2,axis=1)

    return speeds

def jacobian_cdq2(f, x, h=1e-5):
    """Approximate the Jacobian matrix of f:R^n->R^m at x using the second
        order centered difference quotient.

        Parameters:
            f (function): the multidimensional function to differentiate.
                Accepts a NumPy (n,) ndarray and returns an (m,) ndarray.
                For example, f(x,y) = [x+y, xy**2] could be implemented as follows.
                >>> f = lambda x: np.array([x[0] + x[1], x[0] * x[1]**2])
                x ((n,) ndarray): the point in R^n at which to compute the Jacobian.
                h (float): the step size in the finite difference quotient.

                Returns:
                ((m,n) ndarray) the Jacobian matrix of f at x.
    """
    #find out the dimension of output of f
    m = len(f(np.ones_like(x)))
    n = np.shape(x)[0]
    #construct the matrix that will become the jacobian
    jacobian = np.zeros((m,n))
    #constuct identity which columns are standard vectors used in quotient calculation
    identity = np.eye(n)
    for i in range(n):
        jacobian[:,i] = np.array([(f(x + h*(identity[:,i])) - f(x - h*(identity[:,i]))) / (2*h)])

    return jacobian

def test_jacobian():
    """ test the jacobian_cdq2 function """
    f = lambda x: np.array([x[0] + x[1], x[0] * x[1]**2])
    # f = lambda x: np.array([x[0]**2,x[0]**3 - x[1]])
    x = np.array([2,3])
    print(jacobian_cdq2(f,x,h=10e-3))


    x,y = sy.symbols('x y')
    # f = sy.Matrix([[x**2],[x**3 - y]])
    f = sy.Matrix([[x+y],[x*y**2]])
    print(f.jacobian([x,y]))

def cheb_poly(x, n):
    """Compute the nth Chebyshev polynomial at x.

    Parameters:
        y (autograd.ndarray): the points to evaluate T_n(x) at.
        n (int): The degree of the polynomial.
    """
    #t0 should be a vector
    t0, t1 = anp.ones_like(x), x
    if n == 0:
        return t0
    elif n == 1:
        return t1
    else:
        #tn_1, and tn_2 stands for tn-1 and tn-2 respectively
        tn_1 = t1
        tn_2 = t0
        for i in range(n-1):
            tn = 2*x*(tn_1) - tn_2
            tn_1, tn_2 = tn, tn_1
        return tn

def prob6():
    """Use Autograd and cheb_poly() to create a function for the derivative
    of the Chebyshev polynomials, and use that function to plot the derivatives
    over the domain [-1,1] for n=0,1,2,3,4.
    """
    domain = anp.linspace(-1,1,int(1e3))
    #use elementwise_grad for array support of derivative of cheb_poly function (polynomial)
    prime = elementwise_grad(cheb_poly)
    colors = ["blue","green","red","black","purple","orange"]
    # plot the derivatives
    for i in range(5):
        plt.plot(domain,prime(domain,i),color=colors[i],label="T'{}".format(i))
    plt.axis((-1,1,-15,15))
    plt.title("Derivate of each Chebyshev Polynomial up to 4th degree")
    plt.legend(loc="lower left")
    plt.show()

def test_cheb_poly():
    """ visualize output from cheb_poly function """
    x = anp.linspace(-1,1,int(1e3))
    plt.plot(x,cheb_poly(x,0),'-b',label="t0")
    plt.plot(x,cheb_poly(x,1),color="orange",label="t1")
    plt.plot(x,cheb_poly(x,2),color="green",label="t2")
    plt.plot(x,cheb_poly(x,3),color="red",label="t3")
    plt.plot(x,cheb_poly(x,4),color="purple",label="t4")
    plt.axis((-1,1,-1,1))
    plt.legend(loc="lower left")
    plt.show()

def cheby_test():
    """ test previous two functions """
    x = sy.symbols('x')
    t0 = 1
    t1 = x
    t2 = 2*x**2 - 1
    t3 = 4*x**3 - 3*x
    t4 = 8*x**4 - 8*x**2 + 1
    t5 = 16*x**5 - 20*x**3 + 5*x
    print("note t6 shifted up for viewing purposes")
    # t6 = 32*x**6 - 48*x**4 + 18*x**2 - 1 + 10 #this one has been shifted up
    t6 = 32*x**6 - 48*x**4 + 18*x**2 - 1

    # y = 2 #call it y as to no confuse parameter of cheb_poly(x,n)
    y = anp.linspace(-25,25,int(1e4))
    # print(y)

    # print(sy.lambdify(x,t6,"numpy")(y))

    """
    ##another way of testing all the outputs at once
    chebys = [t1,t2,t3,t4,t5,t6]
    l = [t.subs({x:y}) for t in chebys]
    l.insert(0,1)
    print("Evaluate each T_n at x, which are:\n{}\nCompared to Cheby_poly output".format(l))
    """

    print("starting test cases")
    a = t6.subs({x:y}) #answer is 1351 for y = 2
    b = cheb_poly(y,6)
    string = "cheby calculation failed for n=6 and x={} \
        closed form calulation results {}, whereas my function results {}".format(y,a,b)
    # assert  a == b, string

    print(a,b,sep="\n")
    assert  a == b, "failed where function output is {}".format(b)


    # plt.plot(y,sy.lambdify(x,t6,"numpy")(y),label="correct values")
    # plt.plot(y,b,label="function output")
    # plt.show()

def prob7(N=200):
    """Let f(x) = (sin(x) + 1)^sin(cos(x)). Perform the following experiment N
        times:

        1. Choose a random value x0.
        2. Use prob1() to calculate the â€œexactâ€ value of fâ€²(x0). Time how long
        the entire process takes, including calling prob1() (each
        iteration).
        3. Time how long it takes to get an approximation of f'(x0) using
        cdq4(). Record the absolute error of the approximation.
        4. Time how long it takes to get an approximation of f'(x0) using
        Autograd (calling grad() every time). Record the absolute error of
        the approximation.

        Plot the computation times versus the absolute errors on a log-log plot
        with different colors for SymPy, the difference quotient, and Autograd.
        For SymPy, assume an absolute error of 1e-18.
        """

    #construct the function
    f = lambda x: (anp.sin(x)+1)**(anp.sin(anp.cos(x)))

    sympy_time = []
    autograd_time = []
    diff_quotient_time = []

    sympy_error = []
    autograd_error = []
    difference_q_error = []

    for i in range(N):
        #get random integer
        x0 = random.random()
        #record sympy time calculation
        start = time.time()
        prime = prob1()
        exact = prime(x0)
        sympy_time.append(time.time() - start)
        sympy_error.append(1e-18)

        #record autograd time calculation
        start = time.time()
        approximate_grad = grad(f)(x0)
        autograd_time.append(time.time() - start)
        autograd_error.append(np.abs(approximate_grad - exact))

        #record difference quotient time calculation
        start = time.time()
        #my error was coming from cdq4(prime,x0)
        aproximate_quotient = cdq4(f,x0)
        diff_quotient_time.append(time.time() - start)
        difference_q_error.append(np.abs(aproximate_quotient - exact))

    #plot the times on log scale, cast them as arrays out of preference
    #a is the alpha value, the color corresponding to overlap or density of markers
    a = 0.5
    plt.scatter(np.array(sympy_time),np.array(sympy_error),marker="o",color="blue",label="Sympy",alpha=a)
    plt.scatter(np.array(autograd_time),np.array(autograd_error),marker="o",color="green",label="Autograd", alpha = a)
    plt.scatter(np.array(diff_quotient_time),np.array(difference_q_error),marker="o",color="orange",label="Difference Quotients", alpha = a)

    plt.legend(loc="upper right")
    plt.xlabel("Computation Time (seconds)")
    plt.ylabel('Absolute Error')
    plt.xscale('log')
    plt.yscale('log')
    plt.axis((10e-6,5e-2,10e-20,10e-10))
    plt.show()


if __name__ == '__main__':
    pass
    # test_quotients()
    x0 = 1
    # prob3(x0)
    # print(prob4())
    # test_jacobian() #my test cases were weird,
    # test_cheb_poly()
    # cheby_test()
    # prob6()
    # prob7()
