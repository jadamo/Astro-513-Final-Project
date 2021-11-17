#include<vector>
#include<random>
#include<math.h>
#include"model.cpp"

// Simple class to store the current location in parameter space
class Walker{
    public:
        double h;
        double Omega_m;
        double Omega_k;

        // constructors
        Walker():h(0), Omega_m(0), Omega_k(0){}
        Walker(double h, double Omega_m, double Omega_k):
                h(h), Omega_m(Omega_m), Omega_k(Omega_k){}

        // overload operators to make adding and multiplying easier
        Walker operator+(Walker &A){
            Walker B(this->h + A.h, this->Omega_m + A.Omega_m,
                     this->Omega_k + A.Omega_k);
            return B;
        }
        Walker operator-(Walker &A){
            Walker B(this->h - A.h, this->Omega_m - A.Omega_m,
                     this->Omega_k - A.Omega_k);
            return B;
        }
        Walker operator*(double z){
            Walker Y(z*this->h, z*this->Omega_m, z*this->Omega_k);
            return Y;
        }
};

double GetZ(double a){
    double z = 1. / std::sqrt(a);
    return z;
}

double GetP(Walker &X){
    return 0.;
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
double MCMC_Serial(std::vector<double> *H, std::vector<double> *OMEGA_M, 
                 std::vector<double> *OMEGA_K, int K, int M){

    double a_max = 2.0;
    int accepted = 0;
    // setup random number generator 
    std::mt19937_64 rng();
    std::uniform_real_distribution<double> a_dist(1. / a_max, a_max);
    std::uniform_real_distribution<double> r_dist(0.0, 1.0);

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
            // get walker k and a random walker that isn't walker k
            std::uniform_int_distribution<int> j_dist(0, K-1);
            int j = j_dist(rng);
            while(j == k){j = j_dist(rng);}
            Walker X_k = Ensemble[k];
            Walker X_j = Ensemble[j];
            double z = GetZ(a_dist(rng));
            // get new proposed positoin for walker X_k
            Walker Y = X_k + ((X_k*z) - (X_j*z));
            // get 
            double q = std::pow(z, 2)*GetP(Y)*GetP(X_k);
            double r = r_dist(rng);
            if(r <= q){New_Ensemble[k] = Y; accepted++;}
            else       New_Ensemble[k] = X_k;

        }
        //update ensemble to new values
        Ensemble = New_Ensemble;
        Chain.push_back(Ensemble);
    }

    return 1.0*accepted / M;
}

void MCMC_Parallel(){
    return;
}