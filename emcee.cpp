#include<vector>
#include<random>
#include<math.h>

// Simple class to store the current location in parameter space
class Walker{
    public:
        double h;
        double Omega_m;
        double Omega_k;

        // simple constructor
        Walker(double h, double Omega_m, double Omega_k):
                h(h), Omega_m(Omega_m), Omega_k(Omega_k){}

};

double GetZ(double a){
    double z = 1. / std::sqrt(a);
    return z;
}

void SeedEnsemble(std::vector<Walker> Ensemble){

    return;
}

/**
 * Runs the serial emcee MCMC algorithm
 * @param H vector to hold h values in the chain
 * @param OMEGA_M vector to hold h values in the chain
 * @param OMEGA_K vector to hold h values in the chain
 * @param K {int} the unber of walkers to use in one ensemble
 * @param M {int} the number of iterations to run
 * 
*/
void MCMC_Serial(std::vector<double> *H, std::vector<double> *OMEGA_M, 
                 std::vector<double> *OMEGA_K, int K, int M){

    double a_max = 2.0;
    // setup random number generator 
    std::mt19937_64 rng();
    std::uniform_real_distribution<double> a_dist(1. / a_max, a_max);

    // set up vector of ensemble walkers
    std::vector<std::vector<Walker>> Chain;
    std::vector<Walker> Ensemble, New_Ensemble;
    Ensemble.resize(K); New_Ensemble.resize(K);
    for(int k = 0; k < K; k++){
        //TODO: impliment sensible way to seed walkers
        Ensemble[k] = Walker(0.0, 0.0, 0.0);
        New_Ensemble[k] = Walker(0.0, 0.0, 0.0);
    }
    SeedEnsemble(Ensemble);
    Chain.push_back(Ensemble);

    for(int m = 0; m < M; m++){
        //loop thru the ensemble
        for(int k = 0; k < K; k++){
            // get a random walker that isn't walker k
            std::uniform_int_distribution<> j_dist(0, K-1);
            int j = j_dist(rng);
            while(j == k){j = j_dist(rng);}
            Walker X_k = Ensemble[k];
            Walker X_j = Ensemble[j];

            
        }
        //update ensemble to new values
        Ensemble = New_Ensemble;
    }

    return;
}

void MCMC_Parallel(){
    return;
}