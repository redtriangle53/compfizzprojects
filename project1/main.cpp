#include <iostream>
#include <armadillo>
#include <cstdlib>
#include <time.h>

using namespace arma;

// Solver for general tridiagonal matrices. Takes in vectors A, B and C describing the upper, middle and lower diagonal,
// as well as a vector F, being the equivalent of b in Ax = b. Returns a vector V, equivalent to the x.
vec general(vec A, vec B, vec C, vec F){
    double m;
    int n = B.size();
    vec V(n, fill::zeros);

    // Forward substitution algorithm.
    for(int i = 1; i <= n-1; i++){
        if(abs(A(i-1)) >= 1e-7){
        m = B(i-1)/A(i-1);
        B(i) = m*B(i)-C(i-1);
        C(i) *= m;
        F(i) = F(i)*m-F(i-1);
        }
    }

    // Backward substitution algorithm.
    V(n-1) = F(n-1)/B(n-1);
    for(int i = n-2; i >= 0; i--){
        V(i) = (F(i)-C(i)*V(i+1))/B(i);
    }
    return V;
}

// Solver for special tridiagonal matrices with uniform diagonals(where the main diagonal has a distinct value from the other two).
// In this case we need neither A or C at all, only the value of their original elements, a.
vec special(double a, vec B, vec F){
    double b = B(0);
    double m = b/a;
    int n = B.size();
    vec V(n, fill::zeros);

    // Forward substitution algorithm.
    B(1) = m*b - a;
    F(1) = m*F(1) - F(0);
    for(int i = 2; i <= n-1; i++){
        m = B(i-1)/a;
        B(i) = m*b- B(i-2);
        F(i) = m*F(i)- F(i-1);
    }

    // Backward substitution algorithm.
    V(n-1) = F(n-1)/B(n-1);
    for(int i = n-2; i >= 1; i--){
        V(i) = (F(i)-B(i-1)*V(i+1))/B(i);
    }
    V(0) = (F(0)-a*V(1))/B(0);
    return V;
}

int main()
{
    //Variable declaration and initialization block with user input. ------------------------------------------------------------------------------------------
    clock_t timestart, timestop;                                                  //Time variables for recording CPU time.
    int n, mode, algo, matrixtest;                                                //Mode variables for user defined branching.
    double b, a;                                                                  //Element values for special tridiagonal matrices.

    std::cout << "Enter n:\n"; //User input dimension of matrix.
    std::cin >> n;

    double h = 1./((double)n + 1.);                                               //Step length, +1 to account for element n+1.
    vec B(n, fill::ones), A(n, fill::ones), C(n, fill::ones), V(n), Vstandard(n); //A, B and C vectors representing the diagonals, V's will have the solutions.
    vec X = linspace(0.+h, 1.-h, n);                                              //Domain which we are solving for. We already know the boundary conditions(0).
    vec F = 100*exp(-10*X);                                                       //The function which we are integrating. This vector will be the b in Ax = b.
    vec U = 1 - (1-exp(-10))*X-exp(-10*X);                                        //Analytic solution to compare our numerical solution to.

    //User input for determining whether special or general matrix type.
    std::cout << "1 for matrix with uniform diagonals, 2 for general tridiagonal matrix:\n";
    std::cin >> mode;

    //Special matrix init. -------------------------------------------------------------------------------------------------------------------------------------
    if(mode == 1){
        std::cout << "1 for general, 2 for special:\n";                           //User input for determining which algorithm to apply.
        std::cin >> algo;

        std::cout << "Enter a(nondiagonal) first, then b(diagonal):\n";           //User input determining the values the main and secondary diagonals take, respectively.
        std::cin >> a;
        std::cin >> b;

        B *= b;
        A *= a;
        C *= a;
    }

    //General matrix init. User input for every single main and secondary diagonal element. Do not ever select this branch for n=100 and above.
    else{
        algo = 1; //Automatically chooses the general solver branch.
        std::cout << "Enter " << n-1 << " vector elements for A:\n";
        for(int i = 0; i <= A.size()-2; i++){
            std::cin >> A(i);
        }
        std::cout << "Enter " << n << " vector elements for B:\n";
        for(int i = 0; i <= B.size()-1; i++){
            std::cin >> B(i);
        }
        std::cout << "Enter " << n-1 << " vector elements for C:\n";
        for(int i = 0; i <= C.size()-2; i++){
            std::cin >> C(i);
        }
    }

    //User input option for testing against standard solution algorithms for general nxn matrices.
    std::cout << "1 to test against standard matrix algorithms, anything else to not:\n";
    std::cin >> matrixtest;
    std::cout << "Running...\n";

    //Optional standard matrix solver comparison branch. ------------------------------------------------------------------------------------------------------
    if(matrixtest == 1){
        mat M(n, n), L(n, n), U(n, n); //M for main, L and U are what will be lower and upper triagonals, for formality's sake.

        // Loop putting the vector elements in the right spots in the matrix.
        for(int i = 0; i <= n-1; i++){
            M(i, i) = B(i);
            if(i != n-1){
            M(i, i+1) = C(i);
            M(i+1, i) = A(i);
            }
        }

        timestart = clock(); //Applies general matrix solver algorithm, takes time.
        Vstandard = solve(M, F);
        timestop = clock();
        std::cout << "Normal solver time: " << timestop-timestart << endl;

        timestart = clock(); //Applies general LU-decomposition algorithm.
        lu(L, U, M);
        timestop = clock();
        std::cout << "LU solver time: " << timestop-timestart << endl;
    }

    //General tridiagonal algorithm branch. -------------------------------------------------------------------------------------------------------------------
    if(algo == 1){
        timestart = clock();
        V = general(A, B, C, F);
        timestop = clock();
    }

    //Special tridiagonal algorithm branch. -------------------------------------------------------------------------------------------------------------------
    if(algo == 2) {
        timestart = clock();
        V = special(a, B, F);
        timestop = clock();
    }


    //Different outputs depending on whether we are interested in general matrix solutions, or differential equation solutions.
    //Special matrix branch. Compares to differential equation analytical solution(will not make much sense if a and b are not -1 and 2 I guess).
    if(mode == 1){
        double errormax = max(abs((V*pow(h, 2)-U)/U));
        std::cout << "Custom algorithm CPU time:" << timestop-timestart << endl;
        std::cout << "Relative error for h = " << h << ": " << errormax << endl;
    }
    //General matrix branch. Compares accuracy to the standard solver function for testing purposes.
    if(mode == 2){
        double errormax = max(abs((V-Vstandard)/Vstandard));
        std::cout << "Custom algorithm CPU time:" << timestop-timestart << endl;
        std::cout << "Relative error: " << errormax << endl;
    }


    return 0;
}
