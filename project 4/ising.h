#ifndef ISING_H
#define ISING_H

#include <armadillo>
using namespace arma;

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

vec accuavg(vec array){
    int n = array.n_elem;
    vec avgs(n);
    double sum = 0.;
    for (int i = 0; i<n; i++){
        sum += array(i);
        avgs(i) = sum/(i+1.);
    }
    return avgs;
}


int isingE(imat l){
    int E = 0;
    int size = l.n_cols;

    for (int i = 0; i<size; i++){
        for (int j = 0; j < size; j++){
            E -= l(i, j)*(l(i,((j+1) % size)) + l(((i+1) % size), j));
        }
    }
    return E;
}


mat isingMMCMC(int size, int steps, double T, int burnin, bool ordered){
    ivec particle(2);                                          //proposed flip coordinates
    mat stats(3, steps);                                       //samples stored here
    vec w = exp(linspace(-8, 8, 17)/T);                        //array of pre-calculated probabilities
    imat lattice(size, size);                                  //Initial configuration

    if (ordered == false){
        lattice = randi(size, size, distr_param(0, 1))*2-1;   //random configuration
    }
    else{
        lattice.fill(1);                                      //ordered configuration
    }

    int E = isingE(lattice);                                   //initial energy
    int M = accu(lattice);                                     //initial magnetization
    int dE;                                                    //proposed energy change
    int c = -burnin;                                           //sample counter
    int b = size*size;                                         //spacing
    int accepted = 0;                                          //counts number of accepted moves
    imat rparticles(2, b);

    while (c < steps){
        rparticles = randi<imat>(2, b, distr_param(0, size-1));
        for (int s = 0; s < b; s++){
            particle = rparticles.col(s);                      //a random particle to flip is proposed
            dE = 2*(lattice((particle(0)+1)%size, particle(1)) +
                    lattice(particle(0), (particle(1)+1)%size) +
                    lattice((size+particle(0)-1)%size, particle(1)) +
                    lattice(particle(0), (size+particle(1)-1)%size))*
                    lattice(particle(0), particle(1));         //energy change of going from current state to proposed state
            if (dE <= 0 || randu() <= w(-dE+8)){                //acceptance conditions
                lattice(particle(0), particle(1)) *= -1;       //update state
                E += dE;
                M += 2*lattice(particle(0), particle(1));
                accepted++;
            }
        }
        if (c >= 0){                                           //if burn-in/equilibration is over
            stats(0, c) = E;                                   //store samples
            stats(1, c) = M;
            stats(2, c) = accepted;
        }
        c++;
    }
    return stats;
}

void paralisingMMCMC(int size, int steps, double T, double& acE, double& acM, double& acE2, double& acM2, int seed, int burnin){
    arma_rng::set_seed(seed);
    ivec particle(2);                                          //proposed flip coordinates
    mat stats(2, steps);                                       //samples stored here
    vec w = exp(linspace(-8, 8, 17)/T);                        //array of pre-calculated probabilities
    imat lattice = randi(size, size, distr_param(0, 1))*2-1;   //random configuration

    int E = isingE(lattice);                                   //initial energy
    int M = accu(lattice);                                     //initial magnetization
    int dE;                                                    //proposed energy change
    int c = -burnin;                                           //sample counter
    int b = size*size;                                         //spacing
    imat rparticles(2, b);

    while (c < steps){
        rparticles = randi<imat>(2, b, distr_param(0, size-1));
        for (int s = 0; s < b; s++){
            particle = rparticles.col(s);                      //a random particle to flip is proposed
            dE = 2*(lattice((particle(0)+1)%size, particle(1)) +
                    lattice(particle(0), (particle(1)+1)%size) +
                    lattice((size+particle(0)-1)%size, particle(1)) +
                    lattice(particle(0), (size+particle(1)-1)%size))*
                    lattice(particle(0), particle(1));         //energy change of going from current state to proposed state
            if (dE <= 0 || randu() <= w(-dE+8)){                //acceptance conditions
                lattice(particle(0), particle(1)) *= -1;       //update state
                E += dE;
                M += 2*lattice(particle(0), particle(1));
            }
        }
        if (c >= 0){                                           //if burn-in/equilibration is over
            stats(0, c) = E;                                   //store samples
            stats(1, c) = M;
        }
        c++;
    }
    acE = accu(stats.row(0));
    acM = accu(abs(stats.row(1)));
    acE2 = accu(pow(stats.row(0), 2));
    acM2 = accu(pow(stats.row(1), 2));
    return;
}



#endif // ISING_H
