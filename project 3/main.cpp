#include <QCoreApplication>
#include <iostream>
#include <armadillo>
#include <cstdlib>
#include <time.h>
#include <odesolve.h> //Classes for this project contained in this header

using namespace arma;


int main(){ //------------------------------------------------------------------MAIN----------------------------------------------------------
    clock_t start, stop; //Timing variables
    int n, n_frames, mode;  //integration points and number of recorded frames, and a mode variable for selecting which subprogram to run.
    mat recording, vel, analytic;  //recorded positions and velocities and analytic solution
    vec analyticx, analyticy, tvals, timeavg; //just some arrays used for intermediary steps
    double avtime, maxrelerr, h, maxenerr, maxLerr;  //quantities to be recorded and collected, also h is the step length


    //---------------------------------------------------Declaration of Solar system as per 16. oct. 2018------------------------------------
    double year = 365.25; //in days
    double solmassSI = 1.9884e+30;
    double earthmassSI = 5.972e+24;
    double mercurymassSI = 3.3e+23;
    double jupitermassSI = 1.9e+27;
    double marsmassSI = 6.6e+23;
    double venusmassSI = 4.9e+24;
    double saturnmassSI = 5.5e+26;
    double uranusmassSI = 8.8e+25;
    double neptunemassSI = 1.03e+26;

    // Conversion to units relative to the sun's mass
    double m_e = earthmassSI/solmassSI;
    double m_me = mercurymassSI/solmassSI;
    double m_j = jupitermassSI/solmassSI;
    double m_ma = marsmassSI/solmassSI;
    double m_ve = venusmassSI/solmassSI;
    double m_sa = saturnmassSI/solmassSI;
    double m_ur = uranusmassSI/solmassSI;
    double m_ne = neptunemassSI/solmassSI;


    // p for position, v for velocity
    vec solp = {-1.626563776599144E-04 , 7.254085928335599E-03};
    vec solv = {-7.590756043911132E-06 , 2.572902340170582E-06};
    vec earthp = {9.221407681304493E-01 , 3.859513377832802E-01};
    vec earthv = {-6.828707544910924E-03 , 1.585182505936633E-02};
    vec jupp = {-2.647556619196740E+00 ,-4.665803344838063E+00};
    vec jupv = {6.472894738201937E-03 ,-3.365002410907006E-03};
    vec venp = {6.964312859826739E-01 , 2.062367790477971E-01};
    vec venv = {-5.634304924120689E-03 , 1.936063620693106E-02};
    vec mercp = {-1.108553727766392E-01 ,-4.453177109185471E-01};
    vec mercv = {2.167391218861134E-02 ,-5.267049408145004E-03};
    vec marsp = {1.381399662147958E+00 ,-1.174156225640810E-01};
    vec marsv = {1.784568140790161E-03 , 1.513529311587753E-02};
    vec uranp = {1.717189890129584E+01 , 1.000409507836852E+01};
    vec uranv = {-2.008698097731207E-03 , 3.215097265770461E-03};
    vec satp = {1.559567382615958E+00 ,-9.933507349870982E+00};
    vec satv = {5.203446520658728E-03 , 8.475354203872724E-04};
    vec nepp = {2.892187745513884E+01 ,-7.716436088896167E+00};
    vec nepv = {7.887188993004740E-04 , 3.051602067298480E-03};

    // velocities converted from AU/day to AU/year
    solv *= year;
    earthv *= year;
    jupv *= year;
    venv *= year;
    mercv *= year;
    marsv *= year;
    uranv *= year;
    satv *= year;
    nepv *= year;

    //------------------------------------------------------------------Program Start----------------------------------------------------

    
    //----------------------------------------------------------------DISCLAIMER------------------------------------------------
    //I made the mistake of thinking "let's not overdo it with the object orientation since it's the first time I try it", and so many
    //of the modifications I did were manual workarounds, leading to an implementation that is janky at best. Lesson learned. For future
    //reference I will go all the way and generalize the classes further to avoid this. It's better to have a messier class and a clean
    //implementation than having to rewrite code for different use cases. This is the reason why many of the results are not directly
    //replicable with the code here; Only the standard gravitational force(and without a static sun) is available without modifying the
    //system class in odesolve.h. Strictly speaking I should have rewritten everything in a cleaner way and just made sure it still worked
    //as intended, but it turned out to be too hard to make that call with the deadline approaching and a lot of time already invested.
    //Enough excuses! The first 2 modes are alright, although the first requires modification of the system class to get the relativistic
    //correction. Not every case used in the report is still here due to having to hack together code to compensate for my fear of
    //overgeneralizing the classes. Mode 3 should give you an idea of what the code which is now overwritten would look like if I had kept it.

    std::cout << "Select mode: 1 for precession of mercury, 2 for full solar system sim, 3 for benchmarking code.\n";
    std::cin >> mode;

    if (mode == 1){ //-----------------------------------------MERCURY'S PRECESSION----------------------------------------------------
    int n_years = 10; //should be 100 years, but I have modified it to 10 to not make you sit through 100 real years of waiting for the sim to finish.
    int nperyear = 2000000;
    n = n_years*nperyear;
    int n_mercyear = n/n_years*88/365; //approximate number of frames per mercury year(around 0.24 earth years)


    mercp = vec{0.3075, 0.};
    mercv = vec{0., 12.44};
    solp = vec{0.,0.};
    solv = solp;


    System sys; //standard constructor means it will record every frame for every object you add with the "true" bool. Watch your memory!
    sys.add(solp, solv, 1., true);
    sys.add(mercp, mercv, m_me, true);

    VerletSolver Solver(n, 0., n_years, sys);
    Solver.solve();

    recording = Solver.sys.recording; //apparently I need to access a version of sys stored inside Solver, despite calling by reference in the Solver class.
    double dist, angle;
    double mindist = 1.;
    n_frames = recording.n_cols;
    vec currentrelpos, minpos;

    for (int i = 1; i<n_mercyear; i++){ //iterates backwards one mercury year from end of simulation. Simple minimum search algorithm.
        currentrelpos = recording.col(n_frames-i).subvec(2, 3)-recording.col(n_frames-i).subvec(0, 1);
        dist = norm(currentrelpos);
        if (dist < mindist){
            mindist = dist;
            minpos = currentrelpos;
            angle = atan2(currentrelpos(1), currentrelpos(0));
        }
    }
    std::cout << "Latest perihelion angle(in radians): " << angle << endl;
    std::cout << "Corresponding position: " << endl << minpos << endl << "And distance " << mindist << " AU \n";
    }



    if (mode==2){  //------------------------------------------------------FULL SOLAR SYSTEM SIM-----------------------------------------------------
    n = 12000000; //12 years of 1M points each.
    n_frames = 200; //This is why the recording system is nice. Plot looks nice, and I didn't have to break my computer with 9 different trillion point arrays to make it.

    System sys(2, n/n_frames);
    sys.add(solp, solv, 1., true);
    sys.add(earthp, earthv, m_e, true);
    sys.add(jupp, jupv, m_j, true);
    sys.add(venp, venv, m_ve, true);
    sys.add(mercp, mercv, m_me, true);
    sys.add(marsp, marsv, m_ma, true);
    sys.add(uranp, uranv, m_ur, true);
    sys.add(satp, satv, m_sa, true);
    sys.add(nepp, nepv, m_ne, true);


    VerletSolver Solver(n, 0., 12., sys);
    Solver.solve();

    recording = Solver.sys.recording;
    mat earthpos = recording.submat(2, 0, 3, n_frames-1);
    mat juppos = recording.submat(4, 0, 5, n_frames-1);
    mat solpos = recording.submat(0, 0, 1, n_frames-1);
    mat venuspos = recording.submat(6, 0, 7, n_frames-1);
    mat mercurypos = recording.submat(8, 0, 9, n_frames-1);
    mat marspos = recording.submat(10, 0, 11, n_frames-1);
    mat uranuspos = recording.submat(12, 0, 13, n_frames-1);
    mat saturnpos = recording.submat(14, 0, 15, n_frames-1);
    mat neptunepos = recording.submat(16, 0, 17, n_frames-1);
    } //I don't know what to output here for the final file. An I/O block was here, but I figure it's appropriate to not write files to your disk.



    if (mode==3){  //-------------------------------------------------BENCHMARKING-------------------------------------------------------------------
    vec p1 = vec{0., 0.};
    vec p2 = vec{1., 0.};
    vec v2 = vec{0., 2*datum::pi};


    for (int i = 2; i<7; i++){ // reduced maximal power of 10 for n from 8 to 6
        n = pow(10, i);
        n_frames = n;
        h = 1./n;

        { //-------------------------------------------------------------EULER-----------------------------------------------------------------------
        { //The exact same code is repeated for the verlet method below(but in a slightly different way). I should have written this part into a function.
          //forgive me for this messy patchwork code. I decided to include it in the final file just to convince you that I did more than two parts of the project.
          //Also, the results will not match the ones in the report unless the acceleration term in the system class is modified to treat the sun as being in origo.
        System sys(2, n/n_frames);
        sys.add(p1, p1, 1., false);
        sys.add(p2, v2, m_e, true);

        EulerSolver Solver(n, 0., 1., sys);
        Solver.solve();
        recording = Solver.sys.recording;
        n_frames = recording.n_cols;
        }

            tvals = linspace(0, 1., n_frames); //for some reason this bit, which sets up the analytical solution, is indented.
            analytic = mat(2, n_frames);
            analytic.row(0) = cos(2*datum::pi*tvals).t();
            analytic.row(1) = sin(2*datum::pi*tvals).t();

        vec relerr(n_frames);  //This block makes a vector of relative errors, and then finds the largest element.
        mat difference = analytic-recording;
        for (int k = 0; k<n_frames; k++){
            relerr(k) = abs(norm(difference.col(k))/norm(analytic.col(k)));
        }
        maxrelerr = relerr.max();

        // This block finds the velocities, despite the fact that later the system class was modified to record them. This is also terrible since recording is not likely to correspond with h.
        vel = mat(2, n_frames-2);
        for (int j = 1; j<n_frames-1; j++){
            vel.col(j-1) = (recording.col(j+1)-recording.col(j-1))/(2*h);
        }

        //Same type of code as further above but for finding maximal relative deviation in mechanical energy and angular momentum
        mat truncpos = recording.submat(0, 1, 1, n_frames-2);
        vec velsquared(n_frames-2);
        vec dist(n_frames-2);
        vec L(n_frames-2);
        for (int k=0; k<n_frames-2; k++){
            velsquared(k) = dot(vel.col(k), vel.col(k));
            dist(k) = norm(truncpos.col(k));
            L(k) = truncpos(0, k)*vel(1, k)-truncpos(1, k)*vel(0, k);
        }
        mat energy = velsquared/2. - 4.*pow(datum::pi, 2)/dist;
        maxenerr = abs((energy-energy(0))/energy(0)).max();

        //L
        maxLerr = abs((L-L(0))/L(0)).max();

        std::cout << "Step length: " << h << endl; //This part originally just wrote everything to file.
        std::cout << "Euler maximal relative error: " << maxrelerr << endl;
        std::cout << "Euler maximal relative energy deviation: " << maxenerr << endl;
        std::cout << "Euler maximal relative angular momentum deviation: " << maxLerr << endl;
        }
        
        { //-----------------------------------------------------------------VERLET-----------------------------------------------------------------------
        { //The same embarrassing mess is repeated below.
        System sys(2, n/n_frames);
        sys.add(p1, p1, 1., true);
        sys.add(p2, v2, m_e, true);

        VerletSolver Solver(n, 0., 1., sys);
        Solver.solve();
        recording = Solver.sys.recording;
        n_frames = recording.n_cols;
        }
        recording = recording.submat(2, 0, 3, n_frames-1);


            tvals = linspace(0, 1, n_frames);
            analytic = mat(2, n_frames);
            analytic.row(0) = cos(2*datum::pi*tvals).t();
            analytic.row(1) = sin(2*datum::pi*tvals).t();

        vec relerr(n_frames);
        mat difference = analytic-recording;
        for (int k = 0; k<n_frames; k++){
            relerr(k) = abs(norm(difference.col(k))/norm(analytic.col(k)));
        }
        maxrelerr = relerr.max();

        // NOW FOR ENERGY AND ANGMOM
        vel = mat(2, n_frames-2);
        for (int j = 1; j<n_frames-1; j++){
            vel.col(j-1) = (recording.col(j+1)-recording.col(j-1))/(2*h);
        }
        //ENERGY

        mat truncpos = recording.submat(0, 1, 1, n_frames-2);
        vec velsquared(n_frames-2);
        vec dist(n_frames-2);
        vec L(n_frames-2);
        for (int k=0; k<n_frames-2; k++){
            velsquared(k) = dot(vel.col(k), vel.col(k));
            dist(k) = norm(truncpos.col(k));
            L(k) = truncpos(0, k)*vel(1, k)-truncpos(1, k)*vel(0, k);
        }
        mat energy = velsquared/2. - 4.*pow(datum::pi, 2)/dist;
        maxenerr = abs((energy-energy(0))/energy(0)).max();

        //L
        maxLerr = abs((L-L(0))/(L(0))).max();


        std::cout << "Verlet maximal relative error: " << maxrelerr << endl;
        std::cout << "Verlet maximal relative energy deviation: " << maxenerr << endl;
        std::cout << "Verlet maximal relative angular momentum deviation: " << maxLerr << endl;
        }
        
    }
    }
    
    return 0; //This has been a learning experience.
}




















