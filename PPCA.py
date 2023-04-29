
import numpy as np
from sklearn.preprocessing import SplineTransformer, PolynomialFeatures
import matplotlib.pyplot as plt
from generators import Transformer

from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

VALID_SIEVE_CLASS = ['pol', 'spl', 'fourier', 'trad']
VALID_K = ['default', 'max']
VALID_BUILD = ['sklearn', 'constant']

## cheat sheet of formulas: https://miro.com/app/board/uXjVPH7DJK0=/

class ProjectedPCA:

    # 'trad' only estimates F and Lambda 
    def __init__(self, sieve_class = 'spl', K = 'default', J = 'default', build = 'sklearn') -> None:

        # validation
        if not isinstance(K, int) and K not in VALID_K:
            raise ValueError(f"\'{K}'\ is not a valid option for K")
        if sieve_class not in VALID_SIEVE_CLASS:
            raise ValueError(f"\'{sieve_class}\' is not a valid options for sieve_class")
        if build not in VALID_BUILD:
            raise ValueError(f"\'{build}'\ is not a valid option for build")
        if isinstance(J, int) and J <= 3:
            raise ValueError("J must be greater than 3")
        if not isinstance(J, int) and J != 'default':
            raise ValueError("Invalid value for J")
        
        self.sieve_class = sieve_class
        self.K = K
        self.J = J
        self.build = build
 
    def fit(self, X, Y, H1 = None):
        # alias
        p, d = X.shape
        T = Y.shape[1]

        if self.J == 'default':
            self.J  = np.floor(3*((p*min(T,p))**(1/4))).astype(int)
        
        J = self.J
        K = self.K
        sieve_class = self.sieve_class
        
        # capture
        self.X = X
        self.Y = Y

        ## CLASSIC PCA
        if sieve_class == 'trad':
            self.F_hat = np.sqrt(T)*self._PCA(1/T * Y.T @ Y, K, plot = False)
            self.Lambda_hat = 1/T * Y @ self.F_hat
            self.Y_hat = self.Lambda_hat @ self.F_hat.T
            return self
        
        ## PROJECTED PCA
        # Constant's functions, not adapted for pipeline (28/04)
        if self.build == 'constant':
            self.model = Transformer(sieve_class, J)   
        ## Sklearn (Scipy) functions
        else :
            self.model = self._gen_sieve_basis(sieve_class, J)

        # PHI matrix
        self.PHI = PHI = np.concatenate([self.phi(X[:, dd]) for dd in range(d)], axis=1)

        # Projection matrix
        eps = 1e-12
        P = PHI @ np.linalg.inv(PHI.T @ PHI + eps*np.identity(len(PHI.T))) @ PHI.T

        # Factors estimation
        self.F_hat = F_hat = np.sqrt(T)*self._PCA(1/T * Y.T @ P @ Y, K) 

        # remaining estimations
        Lambda_hat = Lambda_hat = 1/T * Y @ F_hat
        Gx_hat = Gx_hat = 1/T * P @ Y @ F_hat
        if isinstance(H1, np.ndarray): Gx_hat = Gx_hat @ H1.T
        Gamma_hat = Lambda_hat - Gx_hat

        # returns estimation
        Y_hat = Lambda_hat @ F_hat.T

        # capture
        self.F_hat = F_hat
        self.Gx_hat = Gx_hat 
        self.Lambda_hat = Lambda_hat
        self.Gamma_hat = Gamma_hat
        self.Y_hat = Y_hat

        # sieve coef.
        if self.build == 'constant':
            self.B_hat = 1/T*np.linalg.multi_dot([np.linalg.inv((PHI.T @ PHI) + eps*np.identity(len(PHI.T))), PHI.T, Y, F_hat])
        else: 
            self.B_hat = self._gen_sieve_coef() # pipeline 
        
        return self 


    # UTILITY PLOT FUNCTIONS
    def plot_K(self, K = 'default'):
        if K == 'default': K_out = self.K_hat
        else: K_out = K
        values = self.values
        plt.title(f"estimated K = {K_out}")
        plt.scatter(np.arange(1, 1+len(values)), values)
        plt.plot(np.arange(1, 1+len(values)), values)
        plt.scatter(K_out, values[K_out-1], facecolors = 'none', s = 100, color = 'red')
        plt.show()

    def plot_Y(self, stock):
        plt.title(f"revenues for stock \'{stock}\'")
        plt.plot(self.Y_hat[stock], label = 'estimation')
        # plt.show()
        plt.plot(self.Y[stock], label = 'true')
        plt.legend()
        plt.show()

    def plot_factor(self, k, l, f_compare = False, score = False):
        assert(k >= 0 and k < self.K)
        assert(l >=0 and l < self.X.shape[1])

        plt.title(f"factor function g_{k},{l}")
    
        if self.build == 'constant':
            J = self.J
            curve_k, curve_l = k, l
            x_plot = np.linspace(0, 1, 100)
            plt.plot(x_plot, self.phi(x_plot) @ self.B_hat[J*curve_l:J*curve_l + J,curve_k], color = 'orange', label = "prediction")

            if f_compare:
                plt.plot(x_plot, f_compare(x_plot), color = 'blue', label = "true")
        
        else: # pipeline
            curve_k, curve_l = k, l
            X_train = self.X[:, curve_l][:, np.newaxis]
            y_train = self.Gx_hat[:, curve_k][:, np.newaxis]
            model_pipeline = make_pipeline(self.model, Ridge(alpha=1e-3))
            model_pipeline.fit(X_train, y_train)
            x_plot = np.linspace(0,1,100)[:, np.newaxis]
            plt.plot(x_plot, model_pipeline.predict(x_plot), color = 'orange', label = "prediction")

            if f_compare:
                plt.plot(x_plot, f_compare(x_plot), color = 'blue', label = "true")
                if score:
                    pipe_score = model_pipeline.score(x_plot, f_compare(x_plot))
                    plt.text(0.6, 1.0, size = 15, color = 'purple', s = f'score = {pipe_score : .2f}')
        
        plt.legend()
        plt.show()

    def predict(self, X):
        # param.
        p, d = X.shape
        K = self.K

        if self.build == 'constant':
            PHI = np.concatenate([self.phi(X[:, dd]) for dd in range(d)], axis=1)
            Gx = PHI @ self.B_hat
        else: # pipeline
            Gx = np.zeros((p,K)) 
            for kk, dd in np.ndindex(K,d):
                Gx[:,kk] += self.curve_g(kk, dd, X[:,dd])

        return Gx @ self.F_hat.T

    def phi(self, x):
        x = x[:, np.newaxis]
        return self.model.transform(x) #[:,1:]

    def curve_g(self, k, l, x): 
        curve_k, curve_l = k, l
        X_train = self.X[:, curve_l][:, np.newaxis]
        y_train = self.Gx_hat[:, curve_k][:, np.newaxis]
        model_pipeline = make_pipeline(self.model, Ridge(alpha=1e-3))
        model_pipeline.fit(X_train, y_train)
        return model_pipeline.predict(x[:, np.newaxis])[:,0]

    def _gen_sieve_coef(self): # bugs or approximation problems...(28/04)
        # param.
        J = self.J 
        d = self.X.shape[1]

        B_hat = np.empty(shape = (J*d,0))
        for k in range(self.K):
            col = np.empty(shape = (1,0))
            for l in range(d):
                curve_k, curve_l = k, l
                X_train = self.X[:, curve_l][:, np.newaxis]
                y_train = self.Gx_hat[:, curve_k][:, np.newaxis]
                model_pipeline = make_pipeline(self.model, Ridge(alpha=1e-3))
                model_pipeline.fit(X_train, y_train)
                col = np.concatenate((col, model_pipeline[1].coef_), axis = 1)
            B_hat = np.concatenate((B_hat, col.T), axis = 1)

        return B_hat
    
    def _gen_sieve_basis(self, sieve_class, J):
        x_train = self.X.flatten() 
        x_train.sort()
        x_train = x_train[:, np.newaxis] 

        if sieve_class == "pol":
            model = PolynomialFeatures(degree=J-1)

        elif sieve_class == "spl": # n_splines = n_knots + degree - 1
            model = SplineTransformer(n_knots=J-2, degree=3, include_bias=True, knots = 'uniform', extrapolation='linear')
        
        model.fit(x_train)

        return model

    def _PCA(self, M : np.ndarray, K:int or str = 'default', plot:bool = False) -> np.ndarray :
        
        # get eigenvalues and eigenvectors
        values, vectors = np.linalg.eigh(M)

        # sort by eigenvalues (decreasing)
        ind = np.argsort(values)[::-1]
        values, vectors = values[ind], vectors.T[ind]
        for i in range(len(vectors)):
            if vectors[i][0] < 0:
                vectors[i] = -vectors[i]
        vectors = vectors.T
        
        # plot eigenvalues
        if plot: 
            plt.scatter(np.arange(1, 1+len(values)), values)
            plt.show()
        
        # estimate K
        K_hat = 1 + np.argmax([values[i]/values[i+1] for i in range(len(values)-1)])
        if K == 'default': K = K_hat 
        elif K == 'max': K = vectors.shape[1]

        # capture
        self.K_hat = K_hat
        self.values = values
        self.K = K

        # return K first eigenvectors
        return np.asarray(vectors[:,:K])