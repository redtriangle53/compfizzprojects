#include <QCoreApplication>
#include <iostream>
#include <armadillo>
#include <cstdlib>
#include <time.h>
#include <thread>
#include <vmc.h>

using namespace arma;


void test(){
    clock_t cstart, cstop;
    int n_cycles = 10000; //number of mc cycles
    int cycle = 20; //steps per cycle
    int burnin = 10; //equilibration period, quite optimistic
    double start = 0.1; //start of search domain over values of alpha
    double end = 10; //end of search domain
    int m = 70; //resolution of domain
    double omega = 1; //frequency

    cstart = clock();
    mat expEs = aplot(start, end, m, PaccT1, ELfuncT1norepulsion, omega, 1., n_cycles, cycle, burnin); //run full search over alphas for non-interactive particles in harmonic oscillator
    cstop = clock();

    std::cout << "Time taken: " << double (cstop-cstart)/CLOCKS_PER_SEC << " s" << endl;
    std::cout << "Time per step: " << double (cstop-cstart)/CLOCKS_PER_SEC/(expEs.n_rows*n_cycles*cycle) << " s" << endl;

    if (abs(expEs.col(1).min()-3) <= 1e-2){ //is the found minimum sufficiently close to 3(the analytical value)?
        std::cout << "Test Passed!" << endl;
    }
    else{
        std::cout << "Test Failed!" << endl;
    }

    return;
}


int main(){ //-----------------------------------------------------------MAIN PROGRAM----------------------------------------------------
    test();
    int n_cycles, cycle, m, zoom, burnin; //initialization of parameters
    double start, end, omega, alpha, beta;

    int mode; //mode selection branching incoming
    std::cout << "Select mode: 0 for stability, 1 for T1 search, 2 for T2 search, 3 for T2 evaluation" << endl;
    std::cin >> mode;

    if (mode == 0){ //----------------------------------------STABILITY - SIMPLE RUN OF T1 TO PRODUCE CONVERGENCE PLOT------------------------------
        n_cycles = 1000000; //perhaps I should have enabled user input for this one as well and put it together with the last "mode".
        cycle = 100;
        burnin = 1000;
        mat expEs(n_cycles, 2);
        omega = 0.01;

        expEs.col(1) = metropolis(PaccT1, ELfuncT1, omega, 0.4303, 1., n_cycles, cycle, burnin);

        std::cout << "Expectation energy: " << accu(expEs.col(1))/n_cycles << endl;
        std::cout << "Energy variance: " << accu(pow(expEs.col(1), 2))/n_cycles - pow(accu(expEs.col(1))/n_cycles, 2) << endl;


        expEs.col(1) = accuavg(expEs.col(1)); //list of running average
        expEs.col(1) = abs(expEs.col(1)-expEs(n_cycles-1, 1))/expEs(n_cycles-1, 1); //pseudo relative error, "exact" is final average
        expEs.col(0) = linspace(1, n_cycles, n_cycles); //set first column to be number of cycles

        expEs = reduce(expEs, 300); //reduce size of matrix for plotting

        //save to file not included, as anyone reading this would not be able to save it to my computer anyway

        std::cout << expEs(299, 1); //print final relative error instead
    }

    if (mode == 1){ //---------------------------------------------------------------T1-SEARCH---------------------------------------------------
        n_cycles = 100000;
        cycle = 50;
        burnin = 100;
        start = 0.2;
        end = 4;
        m = 100;
        std::cout << "input omega: " << endl;
        std::cin >> omega;

        mat expEs;

        expEs = aplot(start, end, m, PaccT1, ELfuncT1, omega, 1., n_cycles, cycle, burnin); //produce plot of expectation energy vs alpha
        int minindex = expEs.col(1).index_min();
        std::cout << expEs.row(minindex) << endl; //print optimal alpha and expectation energy


        //save to file
    }

    if (mode == 2){ //--------------------------------------------------------------T2-SEARCH---------------------------------------------------
        n_cycles = 100000;
        cycle = 50;
        burnin = 100;
        start = 0.0000001;
        end = 3;
        m = 10;
        zoom = 5;
        std::cout << "input omega: " << endl;
        std::cin >> omega;

        std::cout << "input alpha: " << endl;
        std::cin >> alpha;

        mat expEs;

        for (int i = 0; i<10; i++){ //search for optimal beta in slice defined by alpha, then vice versa. Repeat 10 times.
            expEs = bsearch(start, end, m, zoom, PaccT2, ELfuncT2, omega, alpha, n_cycles, cycle, burnin);
            std::cout << "b" << i+1 << " gives " << expEs.row(expEs.col(1).index_min()) << endl;
            beta = expEs(expEs.col(1).index_min(), 0);

            expEs = asearch(start, end, m, zoom, PaccT2, ELfuncT2, omega, beta, n_cycles, cycle, burnin);
            std::cout << "a" << i+1 << " gives " << expEs.row(expEs.col(1).index_min()) << endl;
            alpha = expEs(expEs.col(1).index_min(), 0);
        }



    }

    if (mode == 3){ //----------------------------------------------RUN T2 WITH ARBITRARY VARIATIONAL PARAMETERS AND FREQUENCIES------------------------------------
        n_cycles = 100000; //pretty self explanatory block
        cycle = 50;
        burnin = 100;

        std::cout << "input omega: " << endl;
        std::cin >> omega;

        std::cout << "input alpha: " << endl;
        std::cin >> alpha;

        std::cout << "input beta: " << endl;
        std::cin >> beta;

        mat expEs;

        expEs = metropolis(PaccT2, ELfuncT2, omega, alpha, beta, n_cycles, cycle, burnin);

        std::cout << "Expectation energy: " << accu(expEs)/n_cycles << endl;
        std::cout << "Energy variance: " << accu(pow(expEs, 2))/n_cycles - pow(accu(expEs)/n_cycles, 2) << endl;





    }





    return 0;
}
