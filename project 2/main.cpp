#include <QCoreApplication>
#include <iostream>
#include <armadillo>
#include <cstdlib>
#include <time.h>

using namespace arma;

Col<int> maxoff(mat M){                         //Search algorithm for finding maximal off-diagonal element of symmetric matrix. Returns a 2-tuple of the element's indices.
    int d = M.n_cols;                           //Dimension of matrix
    Col<int> ind(2, fill::ones);                //2-tuple of indices set to 1 so if it fails to find a maximal element it does not return a diagonal element.

    double max = 0;
    double c;
    for(int i=1; i < d; i++){                   //Search loop iterates over d-1 rows, and for each row only i elements(the strict lower triagonal of the matrix).
        for(int j=0; j < i; j++){
            c = abs(M(i, j));
            if(c > max){
                max = c;
                ind(0) = i;
                ind(1) = j;
            }
        }
    }
    return ind;
}

void jacobi(mat& M, mat& ST){                   //Jacobi rotation algorithm. Modifies M to (approximate)diagonal form, and ST to its unitary similarity transformation.
    int d = M.n_cols;                           //Dimension of matrix
    int k, l;                                   //Indices of maximal element.
    double T, t, c, s;                          //Trigonometric quantities cot 2x, tan x, cos x and sin x.
    double aik, ail, akk, all, akl, stik, stil; //Matrix elements A_(index index) and ST_(index index) so the matrices can be directly modified after reading current elements.
    Col<int> maxind;                            //2-tuple of indices of maximal element.
    double max;                                 //absolute value of maximal element.
    ST.eye(d, d);                               //M is similar to M through the identity matrix. Each unitary factor will be multiplied to this to accumulate the total transformation.

    maxind = maxoff(M);                         //Finds indices of the maximal off-diagonal element of M.
    max = abs(M(maxind(0), maxind(1)));
    while(pow(max, 2) > 1e-8){                  //Main loop iterates through transformations until no off-diagonal elements are larger than a threshold.
        k = maxind(0);                          //Loads useful elements into regular variables.
        l = maxind(1);
        akk = M(k, k);
        all = M(l, l);
        akl = M(k, l);

        T = (all - akk)/(2.*akl);               //Calculates the trigonometric quantities corresponding to the angle that eliminates the maximal element.
        if(T > 0){                              //Selects largest absolute value of t to avoid subtraction of similar large numbers.
            t = -T+sqrt(1.+pow(T, 2));
        }
        else{
            t = -T-sqrt(1.+pow(T, 2));
        }
        c = 1./sqrt(1.+pow(t, 2));
        s = t*c;

        for(int i = 0; i<d; i++){               //Specialized matrix multiplication loop.
            aik = M(i, k);                      //Loads quantities whose matrix element will be rewritten.
            ail = M(i, l);
            stik = ST(i, k);
            stil = ST(i, l);
            M(i, k) = aik*c-ail*s;              //Applies similarity transformation to M and modifies ST to represent the total transformation matrix.
            M(i, l) = ail*c+aik*s;
            ST(i, k) = stik*c - stil*s;
            ST(i, l) = stil*c + stik*s;
            M(l, i) = M(i, l);
            M(k, i) = M(i, k);

        }
        M(k, k) = akk*pow(c, 2) - 2*akl*c*s + all*pow(s, 2); //Remaining steps that do not fit into the loop.
        M(l, l) = all*pow(c, 2) + 2*akl*c*s + akk*pow(s, 2);
        M(k, l) = (akk-all)*c*s + akl*(pow(c, 2)-pow(s, 2));
        M(l, k) = M(k, l);

        maxind = maxoff(M);
        max = abs(M(maxind(0), maxind(1)));
    }
    return;
}

int a(double l, mat  A){                        //Function which takes in a value l and a matrix A. Returns the number of eigenvalues of A lower than l.
    int count = 0;
    double q = A(0, 0)-l;
    if (q < 0){
        count += 1;
    }
    for (int i=0; i<A.n_cols-1; i++){
        q = A(i+1, i+1)-l-pow(A(i, i+1), 2)/q;
        if (q < 0){
            count +=1;
        }
    }
    return count;
}

vec bisect(mat A){                              //Bisection method. Takes in a symmetric matrix A, returns a vector of eigenvalues.
    int d = A.n_cols;                           //Dimension of matrix.
    double m = 2;                               //Number of subdomains the domain is split into per iteration.
    double tmax, tmin;                          //Temporary upper and lower bound candidates.
    double c, b;                                //c, b current on- and off-diagonal elements.

    double lmax = A(0, 0)+1;
    double lmin = A(0, 0)-1;
    for (int i=0; i<d-1; i++){                  //Loops through all pairs of on- and off-diagonals to find the lower and upper bounds.
        c = A(i+1, i+1);
        b = A(i, i+1);
        tmax = c+b*b+1;
        tmin = c-b*b-1;
        if (tmax > lmax){
            lmax = tmax;
        }
        if (tmin < lmin){
            lmin = tmin;
        }
    }

    vec ind(d);                                 //The "indices" or "coordinates" of each search area(values depending on how many iterations have gone).
    ind.fill(-1);                               //Program will not recognize negative values as valid indices and ignore them.
    vec indtemp = ind;                          //Each iteration will store its current indices here, then update ind at the end of the iteration.
    ind(0) = 0;                                 //First index is zero, representing that the whole interval [lmin, lmax] is the only area with eigenvalues.

    double ai, aprev, h, ltemp;                 //ai and aprev are evaluations of a for current and previous points. h is the step length and ltemp is the "lmin" for subintervals.
    double e = 1e-8;                            //Precision threshold.
    double kbound = (log10(lmax-lmin)-log10(e))/log10(m); //Number of iterations required for each area to be narrower than the threshold.
    int s = 0;                                  //Index of indtemp which is to be modified.
    int r;                                      //r just represents the index of the current area.

    for (int k=1; k<kbound+1; k++){            //Main loop
        h = (lmax-lmin)/pow(m, k);
        for (int i=0; i < d; i++){             //Each iteration runs over every area noted in ind and investigates them all.
            r = ind(i);
            if (r >= 0){
                ltemp = lmin+r*h*m;
                aprev = a(ltemp, A);
                for (int j=1; j<=m; j++){
                    ai = a(ltemp+j*h, A);
                    if (ai != aprev){
                        aprev = ai;
                        indtemp(s) = r*m+j-1;
                        s += 1;
                    }
                }
            }
        }
        ind = indtemp;
        indtemp.fill(-1.);
        s=0;

    }
    return lmin + ind*h;
}

mat lanczos(mat A){                             //Lanczos' algorithm. Finds an incomplete tridiagonal form and returns similarity transformation matrix Q. Prints out approx eigenvalues.
    int d = A.n_cols;                           //Dimension of matrix.
    int k = 0;                                  //Dimension of incomplete tridiagonal submatrix. Increases as it is built up.
    double a, b;                                //Current on- and off-diagonal elements.
    mat I(d, d, fill::eye), T(d, d, fill::zeros), Q(d, d, fill::zeros), eigvec(d, d); //Identity, transformed, unitary matrices + eigenvector matrix.
    vec r, eigval;                              //r is b_k q_{i+1}, eigval is vector of eigenvalues.

    Q.col(0).randu();                           //Sets the first column of Q to be a random normalized vector.
    Q.col(0) /= norm(Q.col(0));

    b = 1;
    a = dot(Q.col(0), A*Q.col(0));
    r = (A-a*I)*Q.col(0);
    b = norm(r);
    T(0, 0) = a; T(0, 1) = b; T(1, 0) = b;
    while ((abs(b) >= 1e-12) && (k < d-1)){     //Main loop.
        Q.col(k+1) = r/b;                       //Tridiagonalization part.
        k += 1;
        a = dot(Q.col(k), A*Q.col(k));
        r = (A-a*I)*Q.col(k)-T(k-1, k)*Q.col(k-1);
        b = norm(r);
        T(k, k) = a;

        eig_sym(eigval, eigvec, T);             //Finds eigenvalues and vectors of incomplete tridiagonal.
        eigval = eigval.subvec(d-1-k, d-1);
        eigvec = eigvec.submat(0, d-1-k, d-1, d-1);

        for (int i = 0; i < k; i++){            //Checks for convergence of eigenpairs through condition |s_ik b_k| -> 0, in which case it terminates.
            if (abs(eigvec.col(i)(k)*b) < 1e-10){
                std::cout << eigvec.col(i)(k) << endl;
                eigval.subvec(0, 9).print("Approximated Eigenvalues:");
                std::cout << "First convergence of eigenvalue after " << k << " iterations.\n";
                return Q;
            }
        }
        if (k < d-1){
            T(k, k+1) = b; T(k+1, k) = b;
        }
    }
    eigval.print("Approximated Eigenvalues:");
    return Q;
}



void test(){                                       //Unit test function.
    std::cout << "Testing testing 123..\n";


    mat A(5, 5, fill::eye), ST;                    //Checks if the search algorithm finds the largest off-diagonal element.
    A(4, 2) = 0.5;                                 //This element is lower than the diagonal elements to ensure that it ignores those.
    A(2, 4) = 0.5;

    Col<int> ind = maxoff(A);
    if(A(ind(0), ind(1)) = 0.5){
        std::cout << "Search test passed!\n";
    }
    else{
        std::cout << "Search test failed! \n";
    }


    jacobi(A, ST);                                  //Checks if the inner product is conserved, ie if ST is unitary.
    mat a(5, 1, fill::zeros), b(5, 1, fill::zeros); //Since all inner products can be reduced to a combination of inner products that are either parallel or orthogonal,
    a(1) = 1;                                       //Checking that orthonormality is conserved is sufficient.
    b(3) = 1;

    double i = dot((ST*a), (ST*b));
    if(i < 1e-6){
        std::cout << "Orthogonality conserved!\n";
    }
    else{
        std::cout << "Orthogonality test failed!\n";
    }

    i = dot((ST*a), (ST*a));
    if(i-1. < 1e-6){
        std::cout << "Normality conserved!\n";
    }
    else{
        std::cout << "Normality test failed!\n";
    }


    vec E(5);                                       //Initializes simple symmetric tridiagonal matrix for which we have analytic solutions. E contains them.
    for(int i = 0; i < 4; i++){
        A(i, i) = 2.;
        A(i, i+1) = -1.;
        A(i+1, i) = -1.;
        E(i) = 2-2*cos((i+1)*datum::pi/6.);
    }
    A(4, 4) = 2.;
    E(4) = 2.-2*cos(5*datum::pi/6);


    mat J = A;     //Tests if the jacobi-computed eigenvalues of a simple example of our tridiagonal matrix of interest match up with the analytical ones.
    jacobi(J, ST);
    vec eigerror = abs(sort(diagvec(J))-E);

    std::cout << "Jacobi eigenvalue test:\n";
    for(int i = 0; i<4; i++){
        if(eigerror(i) < 1e-8){
            std::cout << "eigenvalue " << i << " is correct!\n";
        }
        else{
            std::cout << "eigenvalue " << i << " is incorrect!\n";
        }
    }


    mat B = A;    //Tests if the bisect-computed eigenvalues of a simple example of our tridiagonal matrix of interest match up with the analytical ones.
    vec bisecteig = bisect(B);
    eigerror = abs(bisecteig-E);

    std::cout << "Bisect eigenvalue test:\n";
    for(int i = 0; i<4; i++){
        if(eigerror(i) < 1e-8){
            std::cout << "eigenvalue " << i << " is correct!\n";
        }
        else{
            std::cout << "eigenvalue " << i << " is incorrect!\n";
        }
    }

    return;
}



int main(){ //-------------------------------------------------------MAIN PROGRAM----------------------------------------------------------------------
    test(); //unit tests run at startup to catch potential errors.
    clock_t start, stop;
    int n;                                      //dimension of matrix.
    double h;                                   //step length.
    int mode;

    std::cout << "input n:\n";                  //user input value for n.
    std::cin >> n;

    mat A(n, n, fill::zeros), ST;               //Initializes main matrix A and similarity transform matrix ST.

    std::cout << "1 for buckling beam, 2 for single electron harmosc, 3 for double electron harmosc\n";
    std::cin >> mode;                           //user input mode selection.


    if(mode == 1){ //--------------------------------------------Buckling beam problem-----------------------------------------------------------------
        h = 1./(n+1.);                          //step length.

        vec E(n);                               //Block setting up A and E(vector containing analytical eigenvalues).
        for(int i = 0; i < n-1; i++){
            A(i, i) = 2.;
            A(i, i+1) = -1.;
            A(i+1, i) = -1.;
            E(i) = (2-2*cos((i+1)*datum::pi/(n+1.)))/pow(h, 2);
        }
        A(n-1, n-1) = 2.;
        E(n-1) = (2.-2*cos(n*datum::pi/(n+1)))/pow(h, 2);

        mat J = A;                              //Jacobi block.
        start = clock();
        jacobi(J, ST);
        stop = clock();
        J /= pow(h, 2);
        vec eig = diagvec(J);
        eig = sort(eig);
        std::cout << eig.subvec(0, 3) << endl;
        std::cout << "Jacobi algorithm time: " << stop-start << endl;

        mat B = A;                              //Bisect block.
        start = clock();
        vec bisecteig = bisect(B);
        stop = clock();
        bisecteig /= pow(h, 2);
        std::cout << bisecteig.subvec(0, 3) << endl;
        std::cout << "Bisect algorithm time: " << stop-start << endl;

    }
    if(mode == 2){ //---------------------------------------------3D radial harmonic oscillator--------------------------------------------------------
        double rhomax = 30.;                    //Sets boundary point and step length.
        h = rhomax/(n+1.);

        vec rho = linspace(h, rhomax-h, n);     //Constructs matrix.
        vec V = pow(rho, 2)*pow(h, 2);
        for(int i = 0; i < n-1; i++){
            A(i, i) = 2. + V(i);
            A(i, i+1) = -1.;
            A(i+1, i) = -1.;
        }
        A(n-1, n-1) = 2. + V(n-1);

        mat J = A;                              //Jacobi block.
        start = clock();
        jacobi(J, ST);
        stop = clock();
        J /= pow(h, 2);
        std::cout << "Jacobi algorithm time: " << stop-start << endl;
        vec eig = sort(diagvec(J));
        eig.subvec(0, 9).print("Eigenvalues for 3d harmosc potential:");

        mat L = A/pow(h, 2);                    //Lanczos block.
        start = clock();
        mat Q = lanczos(L);
        stop = clock();
        std::cout << "Lanczos algorithm time: " << stop-start << endl;

    }
    if(mode==3){ //---------------------------------------------2-electron 3D harmonic oscillator------------------------------------------------------
        double wr;                              //User input of frequency and boundary point
        std::cout << "input value for wr:\n";
        std::cin >> wr;
        double rhomax;
        std::cout << "input value for rhomax:\n";
        std::cin >> rhomax;
        h = rhomax/(n+1.);

        vec rho = linspace(h, rhomax-h, n);     //Constructs matrix.
        vec V = pow(wr, 2)*pow(rho, 2)*pow(h, 2)+ pow(h, 2)/rho;
        for(int i = 0; i < n-1; i++){
            A(i, i) = 2. + V(i);
            A(i, i+1) = -1.;
            A(i+1, i) = -1.;
        }
        A(n-1, n-1) = 2. + V(n-1);

        start = clock();                        //Jacobi block.
        jacobi(A, ST);
        stop = clock();
        A /= pow(h, 2);
        std::cout << "Jacobi algorithm time: " << stop-start << endl;
        vec eig = diagvec(A);
        std::cout << eig(eig.index_min()) << endl;
    }

    return 0;
}

