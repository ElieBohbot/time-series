"""
Created on Tue Mar 28 20:57:14 2017

@author: gregory
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import pylab
import matplotlib.pyplot as plt

x = pd.read_csv('/media/gregory/DATA/x/TSA/extreme/rainfall-monthly-total.csv')

donnees = x["total_rainfall"].as_matrix()

#donnees = np.random.pareto(2, 100000)

stats.probplot(donnees, dist="gumbel_r", plot = plt)
plt.show()
x = np.arange(-10, 10, 0.1)
plt.plot(x, stats.distributions.gumbel_r.pdf(x))
plt.show()

plt.hist(donnees, bins = 20)
"""
On observe que la loi de Gumbel semble avoir les mêmes quantiles que notre série de valeur. 
Par ailleurs, les histogrammes et la distribution de Gumble semblent avoir la même forme. 

Allons plus loins dans notre résonnement. Estimons la fonction de répartition de nos données. 
"""

#%%
""" 
Estimateur de Hill pour des alpha > 0
"""

tri = np.array(donnees)
tri.sort()
N = donnees.size

def hill(donnees, tri, k, n):
    return 1 / ( 1/k * np.log(tri[n-k:n]).sum() - np.log(tri[-k]))

for k in np.arange(1, np.log(N)):
    print(hill(donnees, tri, k, N))



"""
Epsilon donne : 1/6.98 environ 0.14 ce qui est assez petit. 
On a évidemment un problème puisque Hill ne fonctionne pas pour un alpha négatif. 
Il faut utiliser une autre technique en supposant qu'aux extrèmes, nos variables suivent un Weibull
"""

#%%
U = np.arange(00, 500, 1)
Epsi_hat = np.zeros(U.size)

for i in range(U.size):
    Epsi_hat[i] = 1 / (donnees>=U[i]).sum() * ((donnees>=U[i])*(donnees-U[i])).sum()
    
plt.plot(U, Epsi_hat, "o")
plt.show()
#%%

U = np.arange(100, 400, 1)
Epsi_hat = np.zeros(U.size)

for i in range(U.size):
    Epsi_hat[i] = 1 / (donnees>=U[i]).sum() * ((donnees>=U[i])*(donnees-U[i])).sum()
    
plt.plot(U, Epsi_hat, "o")
plt.show()

stats.linregress(U, Epsi_hat)
epsilon = stats.linregress(U, Epsi_hat).slope/ (1 + stats.linregress(U, Epsi_hat).slope)
print(epsilon)
""" Fréchet 0.007 ou alors Gumbel """
    