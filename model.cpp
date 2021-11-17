#include<math.h>
#include<float.h>

// This file has all the model functions from Jake Helton's original code, but converted to C++

// defines the general form of the integrand to be used throughout
double integrand(double z, double h, double omega_r, double omega_m, double omega_l, double omega_k){
    // speed of light given in units of km/s
    // Hubble constant given in units of km/s/Mpc
    double c = 2.998e5;
    double H_0 = h*100;
    double d_H = c/H_0;
    double integrand = d_H/std::sqrt(omega_r*std::pow(1.0 + z,4.0) + omega_m*std::pow(1.0 + z, 3.0) + 
                                     omega_l + omega_k*std::pow(1.0 + z, 2.0));
    return integrand;
}

// comoving distance as a function of redshift, Hubble constant, matter density, and spatial curvature
double d_C(double z, double h, double omega_m, double omega_k){
    // Hubble constant given in units of km/s/Mpc
    // comoving distance returned in units of Mpc
    double omega_r = 4.18e-5*std::pow(h, -2.0);
    double omega_l = 1.0 - omega_m - omega_k;
    double d_C = 0.0;
    //I have to write my own integration method here :(

    //double d_C = integrate.quad(integrand, 0.0, z, args=(h, omega_r, omega_m, omega_l, omega_k))[0];
    return d_C;
}

//comoving angular diameter distance as a function of redshift, Hubble constant, matter density, and spatial curvature
double d_M(double z, double h, double omega_m, double omega_k){
    // speed of light given in units of km/s
    // Hubble constant given in units of km/s/Mpc
    // angular diameter distance returned in units of Mpc
    double c = 2.998e5;
    double H_0 = h*100;
    double d_H = c/H_0;
    double d_M;
    if (omega_k < 0.0){ 
        d_M = (d_H/std::sqrt(std::abs(omega_k)))*std::sin(std::sqrt(std::abs(omega_k))*(d_C(z, h, omega_m, omega_k)/d_H));
    }
    else if (omega_k == 0.0){
        d_M = d_C(z, h, omega_m, omega_k);
    }
    else{
        d_M = (d_H/std::sqrt(std::abs(omega_k)))*std::sinh(std::sqrt(std::abs(omega_k))*(d_C(z, h, omega_m, omega_k)/d_H));
    }
    return d_M;
}

// angular diameter distance as a function of redshift, Hubble constant, matter density, and spatial curvature
double d_A(double z, double h, double omega_m, double omega_k){
    // Hubble constant given in units of km/s/Mpc
    // angular diameter distance returned in units of Mpc
    double d_A = d_M(z, h, omega_m, omega_k)/(1.0 + z);
    return d_A;
}

// ratio of sound horizon to comoving angular diameter distance as a function of redshift, Hubble constant, matter density, and spatial curvature
double ratio(double z, double h, double omega_m, double omega_k){
    // sound horizon given in units of Mpc
    // angular diameter distance given in units of Mpc
    // sound horizon coming from Planck Collaboration+2018
    double r_D = 147.09;
    return r_D/d_M(z, h, omega_m, omega_k);
}

//-----------------------------------

// log_prior
double log_prior(double h, double omega_m, double omega_k){
    // using uniform priors
    if ((h > 0.0 && h < 2.0) && (omega_m > 0.0 && omega_m < 1.0) && 
        (omega_k > -1.0 && omega_k < +1.0)){
        return 0.;
    }
    else return -DBL_MAX;
}

// log_likelihood
double log_likelihood(double h, double omega_m, double omega_k, double z, double ratio, double ratio_error){
    
    // # defining the model
    // model = np.copy(x_array)
    // for i, x in enumerate(x_array):
    //     model[i] = ratio(x, h, omega_m, omega_k)
    double model = ratio(z, h, omega_m, omega_k);

    // the likelihood is sum of the lot of normal distributions
    double ll = -0.5*std::sum((ratio - model)**(2.0)/ratio_error**(2.0) + np.log(ratio_error**(2.0)));
    return ll
}

// log_probability
double log_probability(double h, double omega_m, double omega_k, double z, double ratio, double ratio_error){
    double lp = log_prior(h, omega_m, omega_k);
    double ll = log_likelihood(h, omega_m, omega_k, z, ratio, ratio_error);
    if (lp != 0.0){
        return -DBL_MAX;
    }
    else return lp + ll;
}