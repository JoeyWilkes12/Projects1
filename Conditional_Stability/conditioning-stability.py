import numpy as np
from scipy import linalg as la
from matplotlib import pyplot as plt
import sympy as sy

def matrix_cond(A):
    """Calculate the relative condition number of A with respect to the 2-norm."""
    vals = la.svdvals(A)
    max_val = np.max(vals)
    min_val = np.min(vals)
    #if smallest singular value is 0 then it's a singular matrix
    if min_val == 0:
        return np.inf
    else:
        return max_val / min_val

def test_matrix_cond():
    """ Test the matrix_cond function with orthonormal matrices
        which have condition number of 1, and singular (non invertible)
        matrices which have condition number of infinity
        orthonormal matrices came from: http://mathworld.wolfram.com/OrthogonalMatrix.html
        simple matrix w/ simple eigenvalues example: http://www2.math.uconn.edu/~troby/Math2210F09/LT/sec5_3.pdf
        The singular Value of a matrix is the sqrt of eigenvalues
        """
    #orthonormal test case 1
    A = (1 / np.sqrt(2)) * np.array([[1, 1],[1 , -1]])
    assert matrix_cond(A) == np.linalg.cond(A), "failed orthonormal test case 1 np.linalg.cond"
    # assert matrix_cond(A) == 1, "failed orthonormal test case 1"
    #orthonormal test case 2
    B = (1/3) * np.array([[2,-2,1],[1,2,2],[2,1,-2]])
    assert matrix_cond(B) == np.linalg.cond(B), "failed orthonormal test case 2 compared to np.linalg.cond"
    # assert matrix_cond(B) == 1, "failed orthonormal test case 2"
    #singular (non invertible) matrix
    C = np.array([[1,2,3],[4,5,6],[7,8,9]])
    assert matrix_cond(C) == np.linalg.cond(C), "failed singular matrix example np.linalg.cond"
    # assert matrix_cond(C) == np.inf, "failed singular matrix example"
    #average matrix
    D = np.array([[6,-1],[2,3]])
    #has eigenvalues 5,4
    assert matrix_cond(D) == (np.linalg.cond(D)), "average matrix example failed compared to np.linalg.cond"
    # assert matrix_cond(D) == (np.sqrt(5/4)), "average matrix example failed"

    # print(np.linalg.cond(A))
    # print(np.linalg.cond(B))
    # print(np.linalg.cond(C))
    # print(np.linalg.cond(D))

    print("matrix_cond() function passed all test cases with np.linalg.cond ")

def prob2():
    """Randomly perturb the coefficients of the Wilkinson polynomial by
        replacing each coefficient c_i with c_i*r_i, where r_i is drawn from a
        normal distribution centered at 1 with standard deviation 1e-10.
        Plot the roots of 100 such experiments in a single figure, along with the
        roots of the unperturbed polynomial w(x) corresponding to imaginary part
        of the coefficient to see the deviance.

        Returns:
            (float) The average absolute condition number.
            (float) The average relative condition number.
            """
    # Get the exact Wilkinson polynomial coefficients using SymPy.
    x, i = sy.symbols('x i')
    w = sy.poly_from_expr(sy.product(x-i, (i, 1, 20)))[0]
    # The roots of w are 1, 2, ..., 20.
    w_roots = np.arange(1, 21)
    w_coeffs = np.array(w.all_coeffs(), dtype=float)
    # perturbed_coeffs = np.copy(w_coeffs)
    roots = np.roots(w_coeffs)
    #k_vals is a list containing absolute condition numbers based upon r_i
    k_vals = []
    #list of the relative conditional numbers based upon each experiment
    relatives = []
    #perform the experiment 100x
    for i in range(100):
        #the input of random.normal is location of median, standard deviation, and size
        #without the one in the third slot parameter, I can't use la.norm(r_i,np.inf)
        r_i = np.random.normal(1,1e-10,21)
        #perturb the coefficients
        pert_samples = w_coeffs * r_i
        # perturbed_coeffs = np.vstack((perturbed_coeffs,pert_samples))
        new_root = np.roots(np.poly1d(pert_samples))
        roots = np.vstack((roots,new_root))
        #sort the roots so that they are in the same order as (w_roots) or real roots
        k_individual = la.norm(np.sort(new_root) - w_roots,np.inf) / la.norm(r_i,np.inf)
        k_vals.append(k_individual)
        relatives.append(k_individual * la.norm(w_coeffs, np.inf) / la.norm(w_roots, np.inf))

    #find mean of the absolute conditional number
    #don't use np.average
    k = np.mean(np.array(k_vals))
    #compute the relative conditional number
    rel_cond = np.mean(np.array(relatives))

    #plot the imaginary part of the roots of the perturbed coefficients, the correct roots aren't imaginary
    plt.plot(np.real(roots[0]),np.imag(roots[0]), 'bo',label="Original")
    #plot one of the perturbed experiements with the label "Perturbed"
    plt.plot(np.real(roots[1]),np.imag(roots[1]),'k,',label="Perturbed")
    for i in range(2,101):
        plt.plot(np.real(roots[i]),np.imag(roots[i]),'k,')
    plt.xlabel("Real Axis")
    plt.ylabel("Imaginary Axis")
    plt.legend(loc = "upper left")
    plt.show()
    #return the absolute and relative conditional numbers
    return k, rel_cond

def eig_cond(A):
    """Approximate the condition numbers of the eigenvalue problem at A.

        Parameters:
            A ((n,n) ndarray): A square matrix.

        Returns:
            (float) The absolute condition number of the eigenvalue problem at A.
            (float) The relative condition number of the eigenvalue problem at A.
        """
    #H is matrix that will perturb A
    reals = np.random.normal(0,1e-10,A.shape)
    imags = np.random.normal(0,1e-10,A.shape)
    H = reals + 1j*imags
    eigs, eig_vecs = la.eig(A)
    #perturb matrix A
    A_pert = A + H
    #find eigenvalues of perturbed A
    p_eigs, p_e_vecs = la.eig(A_pert)
    #find the absolute conditional number in euclidean space, 2 norm
    k_hat = la.norm(eigs - p_eigs,2) / la.norm(H,2)
    #compute relative conditional number
    k = k_hat * la.norm(A,2) / la.norm(eigs,2)
    # print("the rel conditional number is {} whereas bellow is output of function\n".format(np.linalg.cond(A_pert)))
    #return the absolute & relative conditional numbers respectively
    return k_hat, k

def prob4(domain=[-100, 100, -100, 100], res=50):
    """Create a grid [x_min, x_max] x [y_min, y_max] with the given resolution. For each
        entry (x,y) in the grid, find the relative condition number of the
        eigenvalue problem, using the matrix   [[1, x], [y, 1]]  as the input.
        I'm using plt.pcolormesh() to plot the condition number over the entire grid.

        Parameters:
            domain ([x_min, x_max, y_min, y_max]): domain of image
            res (int): number of points along each edge of the grid.
        """
    #create domain
    X = np.linspace(domain[0],domain[1],res)
    Y = np.linspace(domain[2],domain[3],res)
    X_s, Y_s = np.meshgrid(X,Y)
    #construct matrix
    M = lambda x,y: np.array([[1, x], [y, 1]])
    relative_numbers = []
    #define matrix
    colors = np.zeros((res,res))
    #use enumerate to correspond with the matrix indices
    for i, x in enumerate(X):
        for j,y in enumerate(Y):
            abs_cond, rel_cond = eig_cond(M(x,y))
            colors[i][j] = rel_cond
    #plot the output
    plt.pcolormesh(X_s,Y_s,colors,cmap='gray_r')
    plt.title("Conditioning of finding eigenvalues for Nonsymmetric & Symmetric Matrices ")
    plt.colorbar()
    plt.show()

def prob5(n):
    """Approximate the data from "stability_data.npy" on the interval [0,1]
        with a least squares polynomial of degree n. Solve the least squares
        problem using the normal equation and the QR decomposition, then compare
        the two solutions by plotting them together with the data. Return
        the mean squared error of both solutions, ||Ax-b||_2.

        Parameters:
            n (int): The degree of the polynomial to be used in the approximation.

        Returns:
            (float): The forward error using the normal equations.
            (float): The forward error using the QR decomposition.
        """
    xk, yk = np.load("stability_data.npy").T
    #n+1 is the number of columns in the resulting vander matrix
    A = np.vander(xk, n+1)
    #first method, use la.inverse which isn't stable
    x_inv = la.inv(A.T @ A)  @  A.T @ yk
    error_inv = la.norm(A @ x_inv - yk,2)
    #second mehtod, use la.qr()
    Q,R = la.qr(A,mode="economic")
    x_qr = la.solve_triangular(R,Q.T @ yk)
    error_qr = la.norm(A @ x_qr - yk,2)
    #forward error of the inverse algorithm, and QR decomposition algorithm
    #use np.polyval which takes in coefficients for polynomial, and points to evaluate at
    plt.plot(xk,np.polyval(x_inv,xk),'-b',label="Normal Equations ")
    plt.plot(xk,np.polyval(x_qr,xk),color="orange",label= "QR Solver")
    plt.plot(xk,yk,"*",color="black",markersize="3")
    plt.legend(loc="upper left")
    plt.axis((0,1,0,25))
    plt.title("Least Squares with Polynomial of Degree {}".format(n))
    plt.show()
    return error_inv, error_qr

def prob6():
    """For n = 5, 10, ..., 50, compute the integral I(n) using SymPy (the
        true values) and the subfactorial formula (may or may not be correct).
        Plot the relative forward error of the subfactorial formula for each
        value of n. Use a log scale for the y-axis.Recall the relative forward
        error is abs(true value - approximation) / abs(true value )
        """
    x = sy.symbols('x')
    rel_forward_error = []
    n = np.arange(5,52,5)
    #for each n, perform the following experiment
    for j in n:
        #ensure that only integers are used
        j = int(j)
        true_val = float(sy.integrate(x**(j) * sy.exp(x-1),(x,0,1) ) )
        #np.exp makes a big difference as opposed to sy.exp in the line below
        approximation = (-1) ** (j) * (sy.subfactorial(j) - (sy.factorial(j) / np.exp(1)))
        rel_forward_error.append(abs(true_val - approximation) / abs(true_val) )
    # return rel_forward_error
    #plot the relative forward error to measure stability
    plt.plot(n,np.array(rel_forward_error),'-b')
    #use log scale in y-axis
    plt.xlabel("Values of n")
    plt.ylabel("Rel. Error")
    plt.yscale("log")
    plt.title(r"$I(n) = \int_{0}^{1} x^n e^{x-1} dx$")
    plt.show()
    # print("And thus we see that this isn't a stable way to compute I(n)")

def tester():
    """ test several functions like eig_cond, and prob5 for several values of n"""
    #constuct A such that the determinant equals zero, has eigs 1,4 where 1 has algebraic multiplicity 2
    A = np.array([[1,2,3],[0,4,0],[0,0,1]])
    eigs, evecs = la.eig(A)
    #output the relative conditional number of A, before perturbation
    # print(matrix_cond(A))
    # print(eig_cond(A))
    # return "delete this return in the tester function "
    print("initiating test cases for prob5")
    # plot prob5 for different values of n
    n = np.arange(7,17)
    for i in n:
        prob5(i)

    return "finished testing"

if __name__ == "__main__":
    pass
    # test_matrix_cond()
    # print(prob2())
    # prob4([-100, 100, -100, 100], res=200)
    # print(prob5(8)) #the n=13 case is comparable to textbook #best fit for n=8
    # print(prob5(14))
    # tester()
    # print(prob6())
