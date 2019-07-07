import numpy as np
import time
import sklearn.metrics

class My_GenSVM:
    """The main SVM object"""

    def __init__(self, x, y, rho="unweighted", lamb=10 ** -8, kappa=0, p=1, epsilun=10 ** -6, description="unspecified",
                 extension=False, max_iter=10000, burn_in=51, seed=124):
        self.x = x
        self.y = y
        # Sadly the lambda variable name is unavailable in python, since it has its own functionality, thus lamb
        self.lamb = lamb
        self.kappa = kappa
        self.p = p
        self.epsilun = epsilun
        self.description = description
        self.extension = extension
        self.max_iter = max_iter
        self.burn_in = burn_in
        np.random.seed(seed)

        # n = number of data instances, m = number of features/attributes
        self.n, self.m = x.shape
        # k = number of classes
        self.k = int(np.max(y))-int(np.min(y))+1
        # rho determines the weights of different classes as to how they impact the total error
        if rho == "unweighted":
            self.rho = np.ones(self.k)
        elif rho == "weighted":
            self.rho = self.weighted_roh()

        # Z is the n by (m+1) matrix of data including the intercept of ones
        self.Z = np.concatenate((np.ones((self.n, 1)), self.x), 1)

        # J is the m+1 diagonal matrix used to transform V to W
        self.J = np.diag(np.ones(self.m + 1))
        self.J[0][0] = 0

        # U_k is the K by (K-1) matrix of vertex coordinates
        self.U_k = self.generate_u_k(self.k)

    # The main iterative majorization algorithm
    def fit_im(self, starting_v=None, printing=True):
        starting_time = time.time()
        # V_hat is the first supporting point of variables t and W' of dimensions (M+1) by (K-1)
        if starting_v is not None:
            v_hat = starting_v
        else:
            v_hat = np.random.randn(self.m + 1, self.k - 1)

        # Initialize other starting values
        losses = []
        hitrates = []
        t = 1
        doubling = False
        loss = self.compute_loss(v_hat)
        loss_prev = (1 + 2 * self.epsilun) * loss
        print("Starting loss: ", loss)

        # Loop until convergence is reached
        while (loss_prev - loss)/loss > self.epsilun and t <= self.max_iter:
            alpha_is = np.zeros((self.n, 1))
            beta_is = np.zeros((self.n, self.k-1))
            for i in range(self.n):
                # Find class of instance i
                class_i = int(self.y[i])

                # Initialize local data structures
                supporting_qs = np.zeros((self.k, 1))
                hubers = np.zeros((self.k, 1))
                nonzeros = 0

                # Compute qs (projection distances), their huber hinges and determine epsilon (nonzeros)
                for j in range(1, self.k+1):
                    if j == class_i:
                        continue
                    supporting_q = self.compute_q(i, class_i, j, v_hat)
                    supporting_qs[j-1] = supporting_q
                    hubers[j-1] = self.huber(supporting_q)
                    if hubers[j-1] != 0:
                        nonzeros += 1
                if nonzeros > 1:
                    epsilon = 0
                else:
                    epsilon = 1

                # Initialize local data structures
                small_a = np.zeros((self.k, 1))
                small_b = np.zeros((self.k, 1))

                # Compute a, b and omega and subsequently alpha and beta
                if epsilon:
                    for j in range(1, self.k+1):
                        if j == class_i:
                            continue
                        small_a[j - 1], small_b[j - 1] = self.compute_a_b(supporting_qs[j - 1], 1)
                    alpha_is[i] = self.compute_alpha_simple(small_a, class_i)
                    beta_is[i][:] = self.compute_beta_simple(small_a, small_b, supporting_qs, class_i)
                else:
                    omega = self.compute_omega(hubers, self.p)
                    for j in range(1, self.k+1):
                        if j == class_i:
                            continue
                        small_a[j - 1], small_b[j - 1] = self.compute_a_b(supporting_qs[j - 1], self.p)
                    alpha_is[i] = self.compute_alpha_omega(small_a, omega, class_i)
                    beta_is[i][:] = self.compute_beta_omega(small_a, small_b, supporting_qs, omega, class_i)

            # Construct majorization matrices A, B and for the extension D
            A = np.diag(alpha_is.flatten())
            B = beta_is
            if self.extension:
                D = self.compute_D(v_hat)
                system_a = np.matmul(np.matmul(self.Z.T, A), self.Z) + np.multiply(self.lamb, D)
            else:
                system_a = np.matmul(np.matmul(self.Z.T, A), self.Z) + np.multiply(self.lamb, self.J)
            system_b = np.matmul(np.matmul(np.matmul(self.Z.T, A), self.Z), v_hat) + np.matmul(self.Z.T, B)

            # Solve the linear system to get V+ (new_V)
            new_V = np.linalg.solve(system_a, system_b)

            # Only start step doubling after 50 iterations burn-in
            if doubling:
                new_V = 2*new_V-v_hat

            # Update loss, store or print diagnostics and set new supporting point V_hat
            loss_prev = loss
            loss = self.compute_loss(new_V)
            losses.append(loss)
            hitrate = self.compute_hitrate(new_V, self.x, self.y)
            hitrates.append(hitrate)
            if printing or (np.mod(t-1, 100) == 0 and t>1):
                print("Iteration: ", t-1)
                print("loss: ", loss)
                print("In sample ARI: ", self.compute_ari(new_V, self.x, self.y))
            v_hat = new_V

            # Check if burn-in is over for step doubling
            if t == self.burn_in:
                doubling = True
            t += 1

        total_time = time.time()-starting_time
        print("Training time: ", total_time)
        print("Iterations: ", t-1)
        print("Final loss: ", loss)
        return v_hat, losses, hitrates, total_time

    # Given a solution V, predicts the labels of given x using the SVM
    def predict_data(self, V, x):
        t_star = V[0, :]
        W_star = np.delete(V, (0), axis=0)
        s_proj = np.matmul(x, W_star) + t_star

        distances = []
        for row in self.U_k:
            distances.append(np.linalg.norm(s_proj-row))
        label = np.argmin(distances) + 1
        return label

    def compute_ari(self, V, x, y):
        predictions = np.zeros(y.shape)
        for i in range(len(y)):
            predictions[i] = self.predict_data(V, x[i])
        ari = sklearn.metrics.adjusted_rand_score(y.flatten(), predictions.flatten())
        return ari

    # This function computes the hitrate between true and predicted labels
    # Kind of obsolete since packages can do it more efficiently
    def compute_hitrate(self, V, x, y):
        hits = 0
        misses = 0
        for i in range(len(y)):
            prediction = self.predict_data(V, x[i])
            if prediction == int(y[i]):
                hits += 1
            else:
                misses += 1
        hitrate = hits / (hits + misses)
        return hitrate

    # Computes the loss L_MSVM or L_GL-MSVM as described by formulas in the paper
    def compute_loss(self, V):
        summed_loss = 0
        for k in range(1, self.k+1):    # k from 1 to K
            class_range = self.find_class_indices(k)    # find G_k
            for index in class_range:                   # i in G_k
                norm_sum = 0
                for j in range(1, self.k+1):
                    if j == k:
                        continue
                    q_i = self.compute_q(index, k, j, V)
                    huber_q = self.huber(q_i)
                    norm_sum += huber_q**self.p
                summed_loss += self.rho[k - 1] * norm_sum ** (1 / self.p)
        if self.extension:
            # This line could give a sqrt warning, caused by negative non diagonal elements of VV', but since the
            # trace is taken (only over diagonal elements), this error can be ignored.
            regularizer = self.lamb * np.sqrt(np.matmul(self.J, np.matmul(V, V.T))).trace()
        else:
            regularizer = self.lamb * np.matmul(V.T, np.matmul(self.J, V)).trace()

        return summed_loss/self.n + regularizer

    def compute_alpha_simple(self, small_a, class_i):
        # epsilon = 1, we can use the simple majorization
        alpha = (1/self.n) * self.rho[class_i - 1] * np.sum(small_a)
        return alpha

    def compute_alpha_omega(self, small_a, omega, class_i):
        # epsilon = 0, we need to apply omega
        alpha = (1/self.n) * self.rho[class_i - 1] * np.sum(small_a * omega)
        return alpha

    def compute_beta_simple(self, small_a, small_b, supporting_qs, class_i):
        # One row of B, 1 by K-1
        def beta_map(a, b, q): return b-a*q
        sum = np.zeros((1, self.k-1))
        for j in range(1, self.k+1):
            if j == class_i:
                continue
            element = beta_map(small_a[j-1], small_b[j-1], supporting_qs[j-1])
            delta = self.U_k[class_i-1, :] - self.U_k[j-1, :]
            sum += np.multiply(delta, element)
        return np.multiply(1 / self.n * self.rho[class_i - 1], sum)

    def compute_beta_omega(self, small_a, small_b, supporting_qs, omega, class_i):
        # One row of B, 1 by K-1
        def beta_map(a, b, q): return omega*(b-a*q)
        sum = np.zeros((1, self.k-1))
        for j in range(1, self.k+1):
            if j == class_i:
                continue
            element = beta_map(small_a[j-1], small_b[j-1], supporting_qs[j-1])
            delta = self.U_k[class_i-1, :] - self.U_k[j-1, :]
            sum += np.multiply(delta, element)
        return np.multiply(1 / self.n * self.rho[class_i - 1], sum)

    def compute_a_b(self, x, p):
        # a and b are computed as from Table 4, Appendix C in Van den Burg and Groenen (2016)
        a = 0
        b = 0

        if p != 2 and x <= (p + self.kappa - 1) / (p - 2) :
            a = 1 / 4 * p ** 2 * (1 - x - (self.kappa + 1) / 2) ** (p - 2)                          # (22)
            b = a * x + 0.5 * p * (1 - x - (self.kappa + 1) / 2) ** (p - 1)                         # (20)
        elif x <= - self.kappa:
            a = 1 / 4 * p * (2 * p - 1) * ((self.kappa + 1) / 2) ** (p - 2)                         # (19)
            b = a * x + 0.5 * p * (1 - x - (self.kappa + 1) / 2) ** (p - 1)                         # (20)
        elif x <= 1:
            a = 1 / 4 * p * (2 * p - 1) * ((self.kappa + 1) / 2) ** (p - 2)                         # (19)
            b = a * x + p / (1 - x) * ((1 - x) / np.sqrt(2 * (self.kappa + 1))) ** (2 * p)          # (17)
        elif x > 1:
            if p == 2:
                a = 1 / 4 * p * (2 * p - 1) * ((self.kappa + 1) / 2) ** (p - 2)                     # (19)
                b = a*x                                                                             # given
            else:
                a = 1 / 4 * p ** 2 * (p / (p - 2) * (1 - x - (self.kappa + 1) / 2)) ** (p - 2)      # (23)
                b = a*((p*x+self.kappa-1)/(p-2)) + 0.5*p*(p/(p-2)*(1-x-(self.kappa+1)/2))**(p-1)    # (24)
        return a, b

    def compute_omega(self, hubers, p):
        def p_power(x): return x ** p
        omega = (1/p)*np.sum(np.apply_along_axis(p_power, 0, hubers))**(1/p-1)
        return omega

    def compute_q(self, i, y_i, j, V):
        return np.matmul(np.matmul(self.Z[i, :], V), self.U_k[y_i-1, :].T - self.U_k[j-1, :].T)

    # This computes the majorization matrix D for the Group Lasso penalty extension
    def compute_D(self, v_hat):
        diagonal_elements = np.zeros(self.m + 1)
        for i in range(1, self.m+1):
            # If the denominator is zero we need to set it so some positve number close to zero to prevent dividing
            # by zero.
            denom = max(10**-12, (2*np.linalg.norm(v_hat[i, :])))
            element = 1/denom
            diagonal_elements[i] = element

        D = np.diag(diagonal_elements)
        return D

    def find_class_indices(self, k):
        indices = []
        for i in range(len(self.y)):
            if self.y[i] == k:
                indices.append(i)
        return indices

    def huber(self, q):
        output = 0
        if q <= -self.kappa:
            output = 1 - q - (self.kappa+1)/2
        elif q <= 1:
            output = 1/(2*(self.kappa+1))*((1-q)**2)
        return output

    def weighted_roh(self):
        weights = np.zeros(self.k)
        new_y = self.y.flatten().astype(int)
        counts = np.bincount(new_y)
        if 0 not in new_y:
            counts = counts[1:]
        for i in range(self.k):
            weights[i] = self.n/(counts[i]*self.k)
        return weights

    @staticmethod
    def generate_u_k(classes):
        U = np.zeros((classes, classes-1))
        for k in range(1, classes+1):
            for l in range(1, classes):
                if k <= l:
                    U[k-1][l-1] = -1/(np.sqrt(2*(l**2+l)))
                elif k == l + 1:
                    U[k-1][l-1] = l/(np.sqrt(2*(l**2+l)))
        return U

    def print_model(self):
        print("This is the multiclass SVM for "+self.description)



