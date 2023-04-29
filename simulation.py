import numpy as np


VALID_DISTRIBUTIONS = ['normal', 'beta']

# generate X as standard distribution
def generate_X(p, d):
    return np.random.normal(size = (p, d))

# VAR model for F
def generate_F(T, K):
    A = np.random.uniform(-0.5, 0.5, size = (K,K))

    def rec(f):
        return A.dot(f.transpose()) + np.random.normal(size=(K,1))
    
    F = np.random.normal(size = (1,K))
    for _ in range(T-1):
        F = np.concatenate((F, rec(F[-1, :].reshape(1, K)).transpose()), axis=0)

    return F

# semiparametric model for Gx
def generate_Gx(X, functions):
    K = len(functions)
    d = len(functions[0])
    p = len(X)

    Gx = np.zeros((p, K))
    for dd, kk, pp in np.ndindex(d, K, p):
        Gx[pp][kk] += functions[kk][dd](X[pp][dd])
    return Gx

# utility function to get a positive semidefinite matrix
def get_near_psd(A):
    C = (A + A.T)/2
    eigval, eigvec = np.linalg.eig(C)
    eigval[eigval < 0] = 0

    return eigvec @ np.diag(eigval) @ eigvec.T

# columns of U from beta distribution
def u_col(T):
    D = np.diag(np.random.beta(7,500, size = T))
    Sigma = np.random.normal(scale = 0.15, size = (T,T))
    np.fill_diagonal(Sigma, 0)

    return np.random.multivariate_normal(mean = np.zeros(T), cov = get_near_psd(D @ Sigma @ D))


def generate_U(p, T, std, distribution = 'normal'):
    if distribution == 'normal':
        return np.random.normal(scale = std, size = (p, T))
    elif distribution == 'beta':
        return np.concatenate([[u_col(T)] for _ in range(p)], axis = 0)
    else:
        raise ValueError(f"\'{distribution}\' is not a valid distribution")

def gen_simulated_data(p, T, functions, error = 'none', identify = False):
    try:
        d = len(functions[0])
        K = len(functions)
    except:
        raise Exception("Incorrect format for functions")
    
    X = generate_X(p, d)
    Gx = generate_Gx(X, functions)

    # Identifiability conditions (Gx)
    values, vectors = np.linalg.eigh(Gx.T @ Gx)
    ind = np.argsort(values)[::-1] # sorting
    values, vectors = values[ind], vectors.T[ind]
    vectors = vectors.T

    H1 = vectors
    Gx = Gx @ np.linalg.inv(H1.T)
    

    # Identifiability conditions (F)
    while True: # assert positives eigenvalues
        F = generate_F(T, K)
        values, vectors = np.linalg.eigh(F.T @ F)
        if values.min() > 0: break

    O = vectors.T  
    D = np.zeros((K, K)) 
    np.fill_diagonal(D, values) 
    H2 = np.linalg.inv(np.sqrt(D) @ O) * np.sqrt(T)
    F = F @ H2 @ H1

    # first element from each column is positive
    for i in range(F.shape[1]):
        if F[0][i] < 0:
            F[:, i] = -F[:, i]
    
    if identify:
        print("Gx_identify:\n", Gx.T @ Gx) # DIAGONAL
        print("F_identify:\n", F.T @ F/T) # IDENTITY (K x K)


    Gamma = np.zeros(Gx.shape)
    # Gamma = np.random.normal(loc = 0, scale = Gx.std()/10, size = Gx.shape)
    Lambda = Gx + Gamma

    Y = Lambda @ F.T 

    if error == 'none':
        U = 0
    else:
        U = generate_U(p, T, Y.std()/10)

    Y += U 

    return X, Y, F, Lambda, Gamma, H1