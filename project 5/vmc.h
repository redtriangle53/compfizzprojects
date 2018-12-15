#ifndef VMC_H
#define VMC_H

#include <armadillo>
#include <cstdlib>

using namespace arma;

vec accuavg(vec array){ //finds array of running averages of elements of some input list, first element is first element, last element is total average
    int n = array.n_elem;
    vec avgs(n);
    double sum = 0.;
    for (int i = 0; i<n; i++){
        sum += array(i);
        avgs(i) = sum/(i+1.);
    }
    return avgs;
}

mat reduce(mat original, int size){ //Squash the columns to reduce resolution
    int b = original.n_cols;
    int a = original.n_rows;
    mat reduced(size, b);
    int skip = a/size;
    for (int i = 0; i<size; i++){
        reduced.row(i) = original.row(i*skip);
    }

    return reduced;
}


double PaccT1(vec posnew, vec posold, double omega, double alpha, double beta){ //Acceptance probability given current and proposed positions for trial function T1
    double normdiff = dot(posnew, posnew)-dot(posold, posold); //r1^2+r2^2 difference between proposed and current position
    double returnval;

    if (normdiff <= 0.){ //sneaky cost saving moved to inside the acceptance probability function. Metropolis function detects hard equality to 1 as guaranteed accept.
        returnval = 1.;
    }
    else{
        returnval = exp(-alpha*omega*normdiff);
    }

    return returnval;
}

double ELfuncT1norepulsion(vec pos, double omega, double alpha, double beta){ //Local Energy function for trial function T1 without electron repulsion

    return omega*(3.*alpha +omega*dot(pos, pos)*(1.-alpha*alpha)/2.);
}

double ELfuncT1(vec pos, double omega, double alpha, double beta){ //Local energy for T1 with electron repulsion

    return omega*(3.*alpha +omega*dot(pos, pos)*(1.-alpha*alpha)/2.)+ 1./norm(pos.subvec(0, 2)-pos.subvec(3, 5));
}

double PaccT2(vec posnew, vec posold, double omega, double alpha, double beta){ //acceptance probability for trial function T2 given current and proposed positions.
    double normterm = -alpha*omega*(dot(posnew, posnew)-dot(posold, posold)); //exponent in "standard" part
    double r12n = norm(posnew.subvec(0, 2)-posnew.subvec(3, 5)); //new/proposed relative distance
    double r12o = norm(posold.subvec(0, 2)-posold.subvec(3, 5)); //old relative distance
    double relterm = r12n/(1+beta*r12n)-r12o/(1+beta*r12o); //exponent in Jastrow factor
    double returnval;
    if (normterm+relterm >= 0.){ //sneaky cost saving moved to inside the acceptance probability function
        returnval = 1.;
    }
    else{
        returnval = exp(normterm+relterm);
    }

    return returnval;
}

double ELfuncT2(vec pos, double omega, double alpha, double beta){ //local energy function for T2
    double r12 = norm(pos.subvec(0, 2)-pos.subvec(3, 5));

    return omega*(3.*alpha +omega*dot(pos, pos)*(1.-alpha*alpha)/2.) + 1./r12 + (alpha*omega*r12-1./(2*pow(1+beta*r12, 2))-2./r12+2.*beta/(1+beta*r12))/(2*pow(1+beta*r12, 2));
}

double Vfunc(vec pos, double omega, double alpha, double beta){ //Potential energy function, applies to all cases

    return omega*omega*dot(pos, pos)/2. + 1./norm(pos.subvec(0, 2)-pos.subvec(3, 5));
}

//metropolis integration function. Takes all parameters + acceptance probability and local energy functions as argument.
vec metropolis(double Pacc(vec posnew, vec posold, double o, double a, double b), double ELfunc(vec pos, double o, double a, double b), double omega, double alpha, double beta, int n_cycles, int cycle, int burnin){
    //initialize things
    double sigma = 0.5; //initial step size parameter, a healthy middle ground.
    double P, EL;       //initializes probability and energy variables
    vec proposed(6, fill::zeros), current(6, fill::zeros), ELsamples(n_cycles), reldists(n_cycles), Vsamples(n_cycles); //various arrays
    mat randvecs(6, cycle); //contains random vectors for whole cycle(for proposal steps)
    int accepts = 0;        //number of accepted moves

    for (int j = -burnin; j<n_cycles; j++){ //burnin is number of equilibration cycles
        accepts = 0;
        randvecs.randu(); //set all elements to random numbers between 0 and 1.
        for (int i = 0; i<cycle; i++){ //loop through a cycle before taking a sample
            proposed = current + (randvecs.col(i)-0.5)*sigma; //proposed step
            P = Pacc(proposed, current, omega, alpha, beta); //acceptance probability for taking that step

            if ((P == 1.) || (randu() <= P)){ //acceptance check. If acceptance probability is set to 1 no check is needed
                current = proposed;
                accepts++;
            }
        }
        sigma += (double(accepts)/cycle-0.5)/100.;              //feedback loop which should make the acceptance rate converge to 50% in an ideal world

        if (j >= 0){                                            //if equilibration is complete
            ELsamples(j) = ELfunc(current, omega, alpha, beta); //take sample
            //reldists(j) = norm(current.subvec(0, 2) - current.subvec(3, 5)); //normally disabled through commenting. Enabled when collecting more complete data. Not very sophisticated on-off switch.
            //Vsamples(j) = Vfunc(current, omega, alpha, beta);
        }


    }

    //std::cout << "mean relative distance: " << accu(reldists)/n_cycles << endl;
    //std::cout << "mean potential energy: " << (accu(ELsamples)-accu(Vsamples))/accu(Vsamples) << endl;

   return ELsamples;
}


//Method for finding minima in reasonable time in one dimension
mat asearch(double start, double end, int m, int z, double Pacc(vec posnew, vec posold, double o, double a, double b), double ELfunc(vec pos, double o, double a, double b), double omega, double beta, int n_cycles, int cycle, int burnin){
    //init start point etc etc
    double min, temp;
    double length = end-start; //size of search domain
    double minpoint = (end+start)/2; //midpoint of domain
    vec domain(m);
    mat storage(m*z, 2);

    for (int o = 0; o < z; o++){ //zoom in by an order every time loop iterates
        //o represents order of zoom
        domain = linspace(abs(minpoint-length/pow(m, o)/2.), minpoint+length/pow(m, o)/2., m); //set up search domain based on coordinate which gives current minimum, an order of m smaller every time
        //loop over new domain
        for (int i = 0; i < m; i++){ //for ever point in domain
            storage(m*o+i, 1) = sum(metropolis(Pacc, ELfunc, omega, domain(i), beta, n_cycles, cycle, burnin))/n_cycles; //store expectation energy
            storage(m*o+i, 0) = domain(i);  //and value of variational parameter alpha
        }
        minpoint = storage(storage.col(1).index_min(), 0); //set midpoint to current optimal value of alpha
    }


    return storage;
}

//same as above except holding alpha constant and varying beta. Sad way to generalize to multiple dimensions. I'll change my ways when we get to three.
mat bsearch(double start, double end, int m, int z, double Pacc(vec posnew, vec posold, double o, double a, double b), double ELfunc(vec pos, double o, double a, double b), double omega, double alpha, int n_cycles, int cycle, int burnin){
    //init start point etc etc
    double min, temp;
    double length = end-start;
    double minpoint = (end+start)/2; //choose midpoint usually
    vec domain(m);
    mat storage(m*z, 2);
    for (int o = 0; o < z; o++){
        //o represents order of zoom
        domain = linspace(abs(minpoint-length/pow(m, o)/2.), minpoint+length/pow(m, o)/2., m); //something like this
        //loop over new domain
        for (int i = 0; i < m; i++){
            storage(m*o+i, 1) = sum(metropolis(Pacc, ELfunc, omega, alpha, domain(i), n_cycles, cycle, burnin))/n_cycles;
            storage(m*o+i, 0) = domain(i);
        }
        minpoint = storage(storage.col(1).index_min(), 0);
    }


    return storage;
}

//less sophisticated search algorithm, simply sweeps over search domain once and picks minimal value.
mat aplot(double start, double end, int m, double Pacc(vec posnew, vec posold, double o, double a, double b), double ELfunc(vec pos, double o, double a, double b), double omega, double beta, int n_cycles, int cycle, int burnin){
    mat storage(m, 2);
    vec E;
    storage.col(0) = linspace(start, end, m); //set domain as first column
    for (int i = 0; i < m; i++){
        E = metropolis(Pacc, ELfunc, omega, storage(i, 0), beta, n_cycles, cycle, burnin);
        storage(i, 1) = accu(E)/n_cycles; //store expectation energies next to the associated value of alpha
    }
    return storage;
}

//again, same as above but holding alpha constant while varying beta
mat bplot(double start, double end, int m, double Pacc(vec posnew, vec posold, double o, double a, double b), double ELfunc(vec pos, double o, double a, double b), double omega, double alpha, int n_cycles, int cycle, int burnin){
    mat storage(m, 2);
    vec E;
    storage.col(0) = linspace(start, end, m);
    for (int i = 0; i < m; i++){
        E = metropolis(Pacc, ELfunc, omega, alpha, storage(i, 0), n_cycles, cycle, burnin);
        storage(i, 1) = accu(E)/n_cycles;
    }
    return storage;
}



#endif // VMC_H
