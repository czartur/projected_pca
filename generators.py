import numpy as np
import math
import matplotlib.pyplot as plt

class Transformer:
    
    # dans le cas des B-splines, on veut des courbes de même degré décalées pour remplir l'intervalle pris par les X

    def __init__(self, b_class, J):
        self.J = J
        self.b_class = b_class
        self.function = {'pol' : self.polynomial, 'spl' : self.b_spline, 'fourier' : self.fourier}

    def transform(self, X):
        # alias
        p = X.shape[0]
        J = self.J
        function = self.function[self.b_class]

        phi = np.empty(shape=(p, J))
        for pp, jj in np.ndindex(p,J):
            phi[pp][jj] = function(jj, X[pp])

        return phi
    
    def partie_positive(self, n,x):
        if x < 0:
            return 0
        elif x == 0 and n == 0:
            return 0.5
        elif x > 0 and n == 0:
            return 1
        elif x >= 0 and n>= 1:
            return x**n

    def b_spline(self, n,x):
        somme = 0
        for k in range(n+2):
            somme += (-1)**k * (n+1)/(math.factorial(n+1-k)*math.factorial(k)) * self.partie_positive(n,x-k+(n+1)/2)

        return somme

    def testCourbes(self, nbPoints,a,b,nbCourbes):
        abscisses = [a+k*(b-a)/nbPoints for k in range(nbPoints)]
        for k in range(nbCourbes):
            ordonnees = []
            for x in abscisses:
                ordonnees.append(self.b_spline(k,x))
            plt.plot(abscisses,ordonnees)
        plt.show()

    def fourier(self, n,x):
        if n%2 == 0:
            return np.cos((n//2)*x)/np.sqrt(2*np.pi)
        else:
            return np.sin((n-1)//2*x)/np.sqrt(2*np.pi)

    def polynomial(self, n,x):
        return x**n