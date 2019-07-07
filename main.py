import numpy as np
from My_GenSVM import My_GenSVM

# loading data
data_name = "breasttissue.csv"
# data_name = "alcohol_consumption.csv"
data = np.loadtxt(data_name, delimiter=' ', skiprows=2)
# data = np.loadtxt(data_name, delimiter=';', skiprows=1)

starting_V = np.loadtxt("startV.csv", delimiter='\t')
# starting_V = np.loadtxt("startV_alc.csv", delimiter=';')

x, y = np.hsplit(data, [-1])

rho = "unweighted"
lamb = 2**-12
kappa = -0.95
p = 2
epsilun = 10**-6
max_iter = 10**10
extension = False


SVM = My_GenSVM(x, y, rho=rho, lamb=lamb, kappa=kappa, epsilun=epsilun, p=p, description=data_name, extension=extension,
                max_iter=max_iter, burn_in=51, seed=1212)

SVM.print_model()
solution = SVM.fit_im(starting_V, printing=False)
print(solution[0])
