#include <QCoreApplication>
#include <iostream>
#include <armadillo>
#include <cstdlib>
#include <time.h>
#include <thread>
#include <ising.h>

using namespace arma;


void test(){                                           //TEST FUNCTION
    int size = 2;                                      //Init basic sim variables
    int steps = 100000;
    double T = 1.;

    mat stats = isingMMCMC(size, steps, T, 0, false);  //Runs simulation

    double Z = 12 + 2*exp(8/T) + 2*exp(-8/T);          //Partition function for 2x2 lattice
    double Eanalytic = (-16*exp(8/T)+16*exp(-8/T))/Z;  //Analytic mean energy
    double Manalytic = (16 + 8*exp(8/T))/Z;            //and magnetization

    double Estat = sum(stats.row(0))/steps;            //Computed energy and magnetization
    double Mstat = sum(abs(stats.row(1)))/steps;

    if (abs((Estat-Eanalytic)/Eanalytic) < 1e-3){      //simple tests to see if they are about the same
        std::cout << "Energy within tolerance. \n";
    }
    else{
        std::cout << "Energy test failed. \n";
    }
    if (abs((Mstat-Manalytic)/Manalytic) < 1e-3){
        std::cout << "Magnetization within tolerance. \n";
    }
    else{
        std::cout << "Magnetization test failed. \n";
    }
    return;
}



int main(){ //---------------------------------------------------------MAIN PROGRAM-----------------------------------------------------------------------------
    test();                           //Tests the method

    arma_rng::set_seed(53);           //Sets random seed to 53. This seed was used for all results except for the phase transition ones(due to multiple threads)
    clock_t start, stop;              //timing variables

    double T;                         //Temperature scaled by k
    int mode, size, steps, burnin;    //mode variable for main program branching, size, steps and burnin simulation parameters.
    vec accuE, accuM, accux, accucv;  //arrays of accumulating averages based on data

                                      //Select your mode
    std::cout << "Mode selection: " << endl <<
                 "1 for 2x2 comparison with analytical values. " << endl <<
                 "2 for 20x20 equilibration. " << endl <<
                 "3 for 20x20 histogram. " << endl <<
                 "4 for phase transition computations" << endl;
    std::cin >> mode;

    if (mode == 1){ //------------------------------------------------ANALYTICAL COMPARISON----------------------------------------------------------------------
        size = 2;       //because 2x2
        steps = 100000; //number of cycles, reduced by factor of 100 from original
        T = 1.;         //temp
        burnin = 1000;  //equilibration cycles

        double Z = 12. + 2.*exp(8./T) + 2.*exp(-8./T);  //Analytical values
        double Eanalytic = (-16.*exp(8./T)+16.*exp(-8./T))/Z;
        double Manalytic = (16. + 8.*exp(8./T))/Z;
        double xanalytic = ((4.*8.+16.*2.*exp(8./T))/Z-pow(Manalytic, 2))/T;
        double cvanalytic = ((8.*8.*2.*(exp(-8./T)+exp(8./T))/Z) - pow(Eanalytic, 2))/pow(T, 2);

        start = clock();
        mat stats = isingMMCMC(size, steps, T, burnin, false);  //simulation
        stop = clock();
        std::cout << "Seconds per step: " << (stop-start)/(double)CLOCKS_PER_SEC/pow(size, 2)/steps << endl;

        accuE = accuavg(stats.row(0).t());  //Computed running averages
        accuM = accuavg(abs(stats.row(1).t()));
        accucv = (accuavg(pow(stats.row(0).t(), 2))-pow(accuE, 2))/pow(T, 2);
        accux = (accuavg(pow(stats.row(1).t(), 2))-pow(accuM, 2))/T;


        mat towrite(steps, 2); //This block is just to reduce the resolution so it can be better plotted
        towrite.col(0) = linspace(1, steps, steps);
        towrite.col(1) = abs((accuE-Eanalytic)/Eanalytic);
        mat reduced = reduce(towrite, 300);

        //In place of file writing:
        std::cout << "Energy relative error: " << abs(accuE(steps-1)/size/size-Eanalytic/steps/steps)/Eanalytic << endl <<
                     "Magnetization relative error: " << abs(accuM(steps-1)/size/size-Manalytic/steps/steps)/Manalytic << endl <<
                     "Heat capacity relative error: " << abs(accucv(steps-1)/size/size-cvanalytic/steps/steps)/cvanalytic << endl <<
                     "Susceptibility relative error: " << abs(accux(steps-1)/size/size-xanalytic/steps/steps)/xanalytic << endl;
    }

    if (mode == 2){  //---------------------------------------------------------EQUILIBRATION BIT---------------------------------------------------------------
        size = 20;  //20x20
        steps = 4000;  //number of cycles

        std::cout << "T = ? " << endl;  //Chose user-input temperature for this one, not sure why, as only two were used
        std::cin >> T;

        bool ordered;  //decides whether to randomize initial array or not
        mat Ms(steps, 2), Es(steps, 2), accepts(steps, 2), stats(steps, 3);
        for (int i = 0; i<2; i++){
            ordered = (i == 1);  //ensures only second run starts with an ordered lattice
            stats = isingMMCMC(size, steps, T, 0, ordered);  //simulation
            Es.col(i) = accuavg(stats.row(0).t())/pow(size, 2);  //collects computed running averages
            Ms.col(i) = accuavg(abs(stats.row(1).t()))/pow(size, 2);
            accepts.col(i) = stats.row(2).t();  //collects total number of accepted steps as function of cycle number
        }

        mat towrite(steps, 2); //Resolution decrease
        towrite.col(0) = linspace(1, steps, steps);
        towrite.col(1) = Es.col(0);
        mat reduced = reduce(towrite, 300);

        //Not sure how to do output from this one that somehow captures what the task asks for
        std::cout << "Average Energy random: " << Es(steps-1, 0) << " Average Energy ordered: " << Es(steps-1, 1) << endl;
        std::cout << "Average Magnetization random: " << Ms(steps-1, 0) << " Average Magnetization ordered: " << Ms(steps-1, 1) << endl;
    }

    if (mode == 3){  //--------------------------------------------------------HISTOGRAM BIT-------------------------------------------------
        size = 20;  //20x20
        steps = 100000;  //cycle number reduced by factor of 10 from original

        std::cout << "T = ? " << endl;  //user input temperature again
        std::cin >> T;;

        mat stats = isingMMCMC(size, steps, T, 1500, false);  //simulation


        vec Es = stats.row(0).t()/pow(size, 2);               //energy per spin
        Es = sort(Es);                                        //sorts the energies

        int c = 0;
        double currentE = 9999999999;
        for (int j = Es.n_elem-1; j > -1; j--){ //this blocks increases c by 1 every time a new energy is encountered
            if (Es(j) != currentE){
                c++;
                currentE = Es(j);
            }
        }
        std::cout << c;

        mat histogram(c, 2, fill::zeros);
        c = 0;  //c is reused
        histogram(0, 0) = currentE;
        for (int i = 0; i< Es.n_elem; i++){ //loops through the list
            if (Es(i) == currentE){         //counts up all identical values
                histogram(c, 1) += 1;
            }
            else{                           //changes to a new space when energy changes
                c++;
                currentE = Es(i);
                histogram(c, 0) = currentE;
                histogram(c, 1) += 1;
            }
        }
        histogram.col(1) /= Es.n_elem;      //Gets probability

        //Outputs the content of the histogram, may or may not be a mess.
        histogram.print("Energy -- Number of states with energy:");

    }

    if (mode == 4){  //---------------------------------------------------------PHASE TRANSITION---------------------------------------------------------
        int nT = 61;     //61 different temperatures
        int steps = 25;  //cycles per thread, originally 10 000 times more
        int burnin = 0; //equilibration cycles, originally 10 000.
        vec Ts = linspace(2., 2.3, nT); //temperature spacing is 0.05
        ivec sizes = {40, 60, 80, 100}; //array of different lattice sizes
        mat plots(nT, 16), stats;       //16 functions of T in one matrix! Come get yours today

        double E_t1, M_t1, E2_t1, M2_t1, //These will be passed by reference, so each thread needs their own variable
                E_t2, M_t2, E2_t2, M2_t2,
                E_t3, M_t3, E2_t3, M2_t3,
                E_t4, M_t4, E2_t4, M2_t4;

        start = clock();
        for (int is = 0; is < 4; is++){  //For each size
            size = sizes(is);
            for (int it = 0; it < nT; it++){  //and each temperature
                T = Ts(it);
                //Do a big thing
                std::thread one(paralisingMMCMC , size, steps, T, std::ref(E_t1), std::ref(M_t1), std::ref(E2_t1), std::ref(M2_t1), 1654, burnin);  //Give jobs to threads
                std::thread two(paralisingMMCMC , size, steps, T, std::ref(E_t2), std::ref(M_t2), std::ref(E2_t2), std::ref(M2_t2), 2565, burnin);  //next to last argument is
                std::thread three(paralisingMMCMC , size, steps, T, std::ref(E_t3), std::ref(M_t3), std::ref(E2_t3), std::ref(M2_t3), 337, burnin); //random seed for that thread
                std::thread four(paralisingMMCMC , size, steps, T, std::ref(E_t4), std::ref(M_t4), std::ref(E2_t4), std::ref(M2_t4), 46588, burnin);
                one.join();  //Wait for threads to finish their jobs
                two.join();
                three.join();
                four.join();

                //half an hour later you have 4 out of 244 values! This bit averages the contributions and adds to the relevant plots
                plots(it, 4*is) = (E_t1+E_t2+E_t3+E_t4)/(steps*4.);                                                //mean energy
                plots(it, 4*is+1) = (M_t1+M_t2+M_t3+M_t4)/(steps*4.);                                              //mean magnetization
                plots(it, 4*is+2) = ((E2_t1+E2_t2+E2_t3+E2_t4)/(steps*4.) - pow(plots(it, 4*is), 2))/pow(T, 2);    //heat capacity at constant volume
                plots(it, 4*is+3) = ((M2_t1+M2_t2+M2_t3+M2_t4)/(steps*4.) - pow(plots(it, 4*is+1), 2))/T;          //susceptibility
            }
            plots.submat(0, 4*is, nT-1, 4*is+3) /= pow(size, 2); //"per spin"
        }
        stop = clock();
        std::cout << (stop-start)/(double)CLOCKS_PER_SEC << endl;

        mat currentplot(nT, 2); //Loads up data into a table so it can be plotted
        currentplot.col(0) = Ts;
        currentplot.col(1) = plots.col(0);

        //Maybe a stupid output, but shows replicability compared to run examples. Also fits the theme of randomness if you need a justificaton for this laziness
        std::cout << "11th member of heat capacity plot: " << plots(10, 2);

    }

    return 0;
}
