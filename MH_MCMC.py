import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator
import scipy.stats as stats
from scipy import integrate
from scipy.optimize import minimize
import tqdm
from IPython.display import display, Math
import corner

class MCMC:
    """
    Simple adaptive MCMC sampler:

    Constructor:
    MCMC(data, nParam, labels, modelFunction, priorFunction, indivMove, solveSigma, sigParams, seed)

          data[N, ncol]: N data points consisting of:
              data[:,0]: independent variable to modelFunction
              data[:,1]: dependent variable to be modeled by modelFunction
              data[:,2]: individual measurement sigma's for each data point
                         (used only if solveSigma=False)

                 nParam: number of parameters to be sampled (other than sigma if solveSigma=True)

                 labels: a numba.typed.List of strings for plot labels

    modelFunction(t, w): where t is the independent variable and w a vector of nParam
                         parameters. Returns a vector of N dependent variables.
                         Must be complied with @njit.

       priorFunction(w): prior pdf on parameters w; returns a probability (on [0,1]).
                         Must be complied with @njit.

              indivMove: if True, samples each component of w separately; if False, chooses move
                         from a multivariate normal distribution with a covariance matrix sampled
                         from the Markov chain. Default is False.

             solveSigma: if True, data[:,2] is ignored, and a single sigma on the data is added
                         as a parameter to sample. Default is False.

           sigParams[2]: sigParams[0] gives the width of the gamma function used to sample sigma
                         and
                         sigParams[1] gives the mean of the sampled sigma.
                         Both parameters are modified as the Markov chain proceeds.
                         Default parameters are [5.0, 1.0]

                   seed: the random number seed to start the rng. A default seed is provided.
    """
    def __init__(self, data,
                     nParam,
                     labels,
                     modelFunction,
                     priorFunction,
                     indivMove=False,
                     solveSigma=False,
                     sigParams=np.array([5.0, 1.0]), seed=9812379):

            self.data = data
            self.N = data.shape[0]
            self.labels = labels

            self.modelFunction = modelFunction
            self.priorFunction = priorFunction
            self.nParam = nParam

            self.indivMove = indivMove
            self.solveSigma = solveSigma
            self.sigParams = sigParams

            np.random.seed(seed)
      
    def multivariate_normal(self, mean, cov, *size):
        # choose the new positions of all parameters together for a multivariate normal distribution 
        n = cov.shape[0]
        eps = 1e-10
        norm = np.random.randn(*size,n).reshape(-1,n)
        return ((np.linalg.cholesky(cov + eps*np.eye(n))@norm.T).T + mean).reshape(*size, n)
    
    def logLikelihood(self, w):

        if self.solveSigma:
            # Solve for uncertainties in data
            self.sumOfSquares = ( (self.data[:,1] - self.modelFunction(self.data[:,0], w) )**2 ).sum()
            #print(self.sumOfSquares)
            logl = -0.5*self.N*np.log(2*np.pi)-0.5*self.N*np.log(self.sigma2) \
            - 0.5*self.sumOfSquares / (self.sigma2)

        else:
            # use individual sigmas in data
            ss = ( ( (self.data[:,1] - self.modelFunction(self.data[:,0], w))/self.data[:,2] )**2 ).sum()
            logl = -0.5*self.N*np.log(2*np.pi)- np.log(self.data[:,2]).sum() - 0.5 * ss
            
        return logl
    
    def logPrior(self, w, wrange):
        p =self.priorFunction(w,wrange)
        if p<=0.0:
            return -np.Inf
        else:
            return 0.0

    def propose(self, w, cov):
        wNew = np.zeros_like(w)
        wNew[:] = w

        if self.indivMove:
            j = int(np.random.rand()*self.nParam)
            wNew[j] = w[j] + (2.83**2/self.nParam) * np.random.randn() * np.sqrt(cov[j,j])
        else:
            wNew[:] = w + 0.5/self.nParam * self.multivariate_normal(np.zeros(self.nParam), cov, 1)

        return wNew, 1.0  # symmetric proposal distribution
    
    def covUpdate(self, x):

        n,p = x.shape  # number of samples, number of parameters

        w = np.ones(n) # possibly for weights...

        if self.cov.sum() > 0:

            for i in range(n):
                xi      = x[i,:]
                wsum    = w[i]
                xmeann  = xi
                xmean   = self.mean + wsum/(wsum+self.wsum)*(xmeann-self.mean);
                xcov    = (self.wsum-1)/(wsum+self.wsum-1)*self.cov + \
                         wsum*self.wsum/(wsum+self.wsum-1)/(wsum+self.wsum)*np.outer((xi-self.mean),(xi-self.mean))
                wsum    = wsum+self.wsum
                self.cov[:,:]  = xcov
                self.mean[:] = xmean
                self.wsum = wsum

        else:

            wsum = w.sum()
            self.mean[:].fill(0.0)
            self.cov[:,:].fill(0.0)
            for i in range(p):
                self.mean[i] = (x[:,i]*w).sum()/wsum;
            if wsum>1:
                for i in range(p):
                    for j in range(i+1):
                        self.cov[i,j] = (x[:,i]-self.mean[i]).T @ ((x[:,j]-self.mean[j])*w[i]) /(wsum-1)
                        if i != j:
                            self.cov[j,i] = self.cov[i,j]
            self.wsum = wsum
            
        
    def sampler(self, w0, sig0, wrange, iterations=10**5, sampleCov=100, startCov=100):
        """
            w0: starting point for the chain
          sig0: diagonal of the initial covariance matrix
    iterations: number of steps to perform
     sampleCov: interval in steps between updates of covariance matrix
      startCov: starting step after which covariance matrix is updated every sampleCov steps

        Results are in:

        self.chain[steps, nvar]
        self.logLchain[steps]
        self.acceptRatio[steps]

        """

        w = w0.copy()
        assert self.nParam == len(w)

        self.chain = np.zeros((iterations, self.nParam), dtype=np.float64)
        self.logLchain = np.zeros(iterations, dtype=np.float64)
        self.acceptRatio = np.zeros(iterations, dtype=np.float64)
        self.sigmaChain = np.zeros(iterations, dtype=np.float64)

        # Initialize covariance matrix to the diagonal form given as an argument
        self.wsum = 1
        self.cov = np.zeros((self.nParam,self.nParam))
        for i in range(self.nParam):
            self.cov[i,i] = sig0[i]**2
        self.mean = np.zeros(self.nParam)

        if self.solveSigma:
            # sample initial guess at data variance
            shape = 0.5*self.sigParams[0]
            scale = 1.0/(0.5*self.sigParams[0]*self.sigParams[1])
            self.sigma2 = 1.0/np.random.gamma(shape, scale, size=1)[0]
            #print(self.sigma2)

        # first sample of the chain
        self.chain[0,:] = w
        logp = self.logPrior(w, wrange)
        logl = self.logLikelihood(w)

        # run the chain
        ilast = startCov
        acceptSum = 0
        for i in tqdm.tqdm_notebook(range(iterations)):

            # propose a move and calculate prior
            wNew, qratio = self.propose(w, self.cov)
            logpNew = self.logPrior(wNew, wrange)


            
            # Only evaluate the likelihood if prior prob isn't zero
            loglNew = -np.inf
            if logpNew != -np.Inf:
                loglNew = self.logLikelihood(wNew)
                
            # only when prior and likelyhood function is valid than calculate the total probability    
            if logpNew == -np.Inf or np.isnan(loglNew):
                logRatio = -np.Inf
            else:
                # Log of acceptance ratio p(D|wNew)p(wNew) / ( p(D|w0)p(w0) )
                logRatio = (logpNew+loglNew) - (logp+logl)
            logRatio = min(0.0, logRatio)
            

            # Acceptance/rejection
            rand_ratio = np.random.rand()

            if rand_ratio <= np.exp(logRatio)*qratio:
                w = wNew
                logp = logpNew
                logl = loglNew
                acceptSum += 1

            if self.solveSigma:
                # Gibbs sampler for sigma^2: move is always accepted
                shape = 0.5*(self.sigParams[0] + self.N)
                scale = 0.5*(self.sigParams[0]*self.sigParams[1] + self.sumOfSquares)
                self.sigma2 = 1.0/np.random.gamma(shape, 1.0/scale, size=1)[0]
                #print(self.sigma2)
                self.sigmaChain[i] = np.sqrt(self.sigma2)

            self.chain[i, :] = w
            self.logLchain[i] = logl
            self.acceptRatio[i] = acceptSum/(i+1)

            # update covariance
            if i == ilast + sampleCov:
                self.covUpdate(self.chain[ilast:i,:])
                ilast = i
