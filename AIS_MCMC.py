#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import tqdm


class MCMC:
    """
    Simple adaptive MCMC sampler:

    Constructor:
    MCMC(nwalker, nParams, labels, model, prior, solveSigma, data, a=2, seed=12345678)

          data[N, ncol]: N data points consisting of:
              data[:,0]: independent variable to modelFunction
              data[:,1]: dependent variable to be modeled by modelFunction
              data[:,2]: individual measurement sigma's for each data point
                         (used only if solveSigma=False)

                 nParam:   number of parameters to be sampled (other than sigma if solveSigma=True)

                 labels:   a numba.typed.List of strings for plot labels

    modelFunction(t, w):   where t is the independent variable and w a vector of nParam
                           parameters. Returns a vector of N dependent variables.
                           Must be complied with @njit.

       priorFunction(w):   prior pdf on parameters w; returns a probability (on [0,1]).
                           Must be complied with @njit.

                      a:   adjustable scale parameter of the Stretch Move. Default value is provided                                according to Goodman, J., & Weare, J. 2010, Commun. Appl. Math. Comput.                                  Sci., 5, 65

                   seed: the random number seed to start the rng. A default seed is provided.
    """

    def __init__(self, 
                 nwalkers, 
                 nParam, 
                 labels,
                 modelFunction, 
                 priorFunction, 
                 data, 
                 a=2, 
                 seed=12345678):
        
        #reshape data for broadcasting

        #datax = data[:,0].reshape(-1,1)
        #datay = data[:,1].reshape(-1,1)
        #dataz = data[:,2].reshape(-1,1)

        self.data = data
        
         
        self.N = data.shape[0] 
        self.labels = labels

        self.modelFunction = modelFunction
        self.priorFunction = priorFunction
        self.nP = nParam

        self.a = a
        self.nW = nwalkers

        np.random.seed(seed)


    # Prior distribution: only give a limitation range here
    def logPrior(self, w, wrange):
        p =self.priorFunction(w,wrange)
        if p<=0.0:
            return -np.Inf
        else:
            return 0.0



    # likelihood function
    def logLikelihood(self, w):
        # use individual sigmas in data
        ss = ( ( (self.data[:,1] - self.modelFunction(self.data[:,0], w))/self.data[:,2] )**2 ).sum()
        logl = -0.5*self.N*np.log(2*np.pi)- np.log(self.data[:,2]).sum() - 0.5 * ss

        return logl



    def propose(self, w, k):

        """
        Uses an affine invariant transforation to propose a new position
        for one walker given the position of another.

        This function returns:

        * wNew: The new proposed position.

        * logpNew: The vector of log-probabilities at the positions
          given by 'wNew'.

        * zz: random scale factor

        """

        wNew    = np.zeros_like(w)
        wNew[:] = w #(nParams, nwalker)


        #draws a random number from the probability density g(z) = sqrt(z)
        zz = ((self.a - 1.) * np.random.rand() + 1) ** 2. / self.a


        # Generate the vectors of random numbers that will produce the
        # proposal.
        rint         = np.random.randint(self.nW-1)

        complEnsem   = np.zeros((self.nP,self.nW-1))
        complEnsem[:]= np.delete(w, k,1) #list of complementary walkers of p0


        # Calculate the proposed positions and the log-prior.
        wNew[:,k]    = complEnsem[:,rint] + zz * (w[:,k] - complEnsem[:,rint])
        

        return wNew, zz




    def sampler(self, w0, wrange, iterations, save_chain = False, sampleCov=100, startCov=100):
        """
        w0        : 2d array with the initial values; (nParams, nwalker)
        wrange    : ranges in parameter values
        iterations: number of steps to perform

        Results are in:

        self.chain[steps, nvar]
        self.logLchain[steps]
        self.acceptRatio[steps]

        """

        w = w0.copy()

        assert self.nP == len(w)

        self.chain       = np.zeros((self.nW, iterations, self.nP), dtype=np.float64)
        self.logLchain   = np.zeros((self.nW, iterations), dtype=np.float64)
        self.acceptRatio = np.zeros((self.nW, iterations), dtype=np.float64)
        
        
        logp= np.zeros(self.nW)
        logl= np.zeros(self.nW)
        for k in range(self.nW):
            logp[k] = self.logPrior(w[:,k], wrange)
            logl[k] = self.logLikelihood(w[:,k])

        # count the number of successful proposals
        acceptSum = 0

        #run sampler
        for i in tqdm.tqdm_notebook(range(iterations)):
            for k in range(self.nW):
                                                
                # propose a move and calculate prior
                wNew, zz = self.propose(w, k) #wNew (nParams, nwalker)
                logpNew = self.logPrior(wNew[:,k], wrange) 
                

                # Only evaluate the likelihood if prior prob isn't zero
                loglNew = -np.inf
                if logpNew != -np.Inf:
                    loglNew = self.logLikelihood(wNew[:,k])

                # only when prior and likelyhood function is valid than calculate the total probability    
                if logpNew == -np.Inf or np.isnan(loglNew):
                    logRatio = -np.Inf
                else:
                    # Log of acceptance ratio for the stretch move: Z^(nParams -1) p(wNew)/p(w)
                    logRatio = (self.nP - 1.) * np.log(zz) + (logpNew + loglNew ) - (logp[k] + logl[k])
                logRatio = min(0.0, logRatio)



                # Acceptance/rejection
                #accept = [alpha <= np.exp(logRatio)[i] for i in range(nParam)]
                if np.random.rand() <= np.exp(logRatio):
                    w[:,k] = wNew[:,k]
                    logp[k] = logpNew
                    logl[k] = loglNew


                    acceptSum += 1

            

                self.chain[k, i, :] = w[:,k]
                self.logLchain[k, i] = logl[k]
                self.acceptRatio[k, i] = acceptSum/(i+1)
        
        self.acceptRatio = self.acceptRatio/self.nW

        if save_chain == True:
            np.savetxt("AISChain_{}.txt".format(self.nW), self.chain, delimiter=',')

            

