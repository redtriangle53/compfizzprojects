#ifndef ODESOLVE_H
#define ODESOLVE_H
#include <armadillo>

using namespace arma;

class System {  //System class contains all parameters, interactions and states of the system, and makes a recording of the system.
    public:
    int obnum = 0;  //number of physical objects in the system
    int d, c, e, frameskip, framenum; //dimensionality, counters, number of frames between each recorded frame, and number of recorded frames
    vec masses, recordbool;  //vector of masses, an array of truth values. The ith object will be recorded if the ith bool is true
    mat poss, prevposs, vels; //current positions, previous positions and current velocities. This should be generalized to a single matrix containing the states some number of steps back.
    mat recording, recordv; //recordings of the positions and velocities for full sim.

    System(){ //Standard constructor. Standard dimensionality is 2 and standard recording resolution is max.
        d=2;
        frameskip = 1;
    }

    System(int dimensions, int fskip){ //Constructor with custom dimensionality and recording resolution.
        d = dimensions;
        frameskip = fskip;
        if (fskip == 0){
            frameskip = 1;
        }
    }

    void add(vec p, vec v, double mass, bool rec){ //adds a physical object to the system by simply loading up the new info with the old in new, larger arrays.
        if (obnum == 0){
            masses = vec(1);
            recordbool = vec(1);
            poss = mat(d, 1);
            prevposs = mat(d, 1);
            vels = mat(d, 1);

            masses(0) = mass;
            recordbool(0) = rec;
            poss.col(0) = p;
            prevposs.col(0) = p;
            vels.col(0) = v;

            obnum +=1;
            return;
        }
        mat temppos(d, 1+obnum);
        mat tempvel(d, 1+obnum);
        vec tempmasses(1+obnum);
        vec temprecbool(1+obnum);

        temppos.submat(0, 0, d-1, obnum-1) = poss;
        tempvel.submat(0, 0, d-1, obnum-1) = vels;
        tempmasses.subvec(0, obnum-1) = masses;
        temprecbool.subvec(0, obnum-1) = recordbool;

        temppos.col(obnum) = p;
        tempvel.col(obnum) = v;
        tempmasses(obnum) = mass;
        temprecbool(obnum) = rec;

        poss = temppos;
        prevposs = temppos;
        vels = tempvel;
        masses = tempmasses;
        recordbool = temprecbool;

        obnum +=1;
        return;
    }

    void update(vec p, vec v, int k){ //rewrites the state arrays with new information at the end of a step
        prevposs.col(k) = poss.col(k);
        poss.col(k) = p;
        vels.col(k) = v;
    }


    //These three functions just give you the current or previous position, or the current valocity, of some object in the system.
    vec pos(int k){
        return poss.col(k);
    }

    vec prevpos(int k){
        return prevposs.col(k);
    }

    vec vel(int k){
        return vels.col(k);
    }

    //Returns the acceleration affecting object k by summing up contributions from every other object. This is standard gravity, I should find a way to customize this
    //in-program since I either have to write child classes for every alternative or modify it manually for each use case(like I have) if this is not possible.
    vec acc(int k){
        vec r;
        vec a = vec(d, fill::zeros);
        for (int i=0; i<obnum; i++){
            if (i != k){
                r = pos(k)-pos(i);
                a -= 4.*pow(datum::pi, 2)*masses(i)/pow(norm(r), 3)*r;
            }
        }
        return a;
    }

    void record(int j, int n){  //Records the positions of every object with it activated after some number of steps.
        if (j==0){
            framenum = n/frameskip;
            c = 0;
            for (int i=0; i<obnum; i++){
                if (recordbool(i)){
                    c +=1;
                }
            }
            recording = mat(d*c, framenum, fill::zeros);
            recordv = mat(d*c, framenum, fill::zeros);
            c = 0;
            e = 0;
        }
        if (j%frameskip == 0){
            for (int i=0; i<obnum; i++){
                if (recordbool(i)){
                    recording.col(e).subvec(d*c, d*c+1) = poss.col(i);
                    recordv.col(e).subvec(d*c, d*c+1) = vels.col(i);
                    c += 1;
                }
            }
            c = 0;
            e += 1;
        }
    }

};


class ODESolver { //General solver class for physical systems. Takes in a system object like a console takes in a cartridge(back in the day), and plays it.
    vec r, prevr, a, v;  //position, previous position, acceleration and velocity
    double h, t0, tf;  //step length, start time, final time
    int n;  //number of steps

    public:
    System sys;

    ODESolver(int m, double t1, double t2, System& s){  //No standard constructor here. Integration points, times and which system to simulate must be given.
        sys = s;
        n = m;
        t0 = t1; tf = t2;
        h = (tf-t0)/m;
    }

    virtual void step(vec& r, vec& prevr, vec& a, vec& v, double h){  //Not a very accurate numerical method.
        return;
    }


    void solve(){  //iterates through every object in the system and updates them for each time step. Is it good or bad that positions are updated mid-step?
        for (int i = 0; i < n; i++){
            for (int j = 0; j<sys.obnum; j++){
                v = sys.vel(j);
                a = sys.acc(j);
                r = sys.pos(j);
                prevr = sys.prevpos(j);
                step(r, prevr, a, v, h);
                sys.update(r, v, j);
            }
            sys.record(i, n);
        }
        return;
    }
};

class EulerSolver : public ODESolver {  //Child class of ODESolver with a slightly more accurate step method.
    public:
    EulerSolver(int m, double t1, double t2, System& s) : ODESolver(m, t1, t2, s){}
    virtual void step(vec& r, vec& prevr, vec& a, vec& v, double h){
        r += v*h;
        v += a*h;
        return;
    }
};

class VerletSolver : public ODESolver {  //Even more accurate, wowza.
    public:
    VerletSolver(int m, double t1, double t2, System& s) : ODESolver(m, t1, t2, s){}
    virtual void step(vec& r, vec& prevr, vec& a, vec& v, double h){
        if (norm(r-prevr) == 0.){
            r += v*h;
            v += a*h;
            return;
        }
        r = 2*r-prevr+a*h*h;
        v = (r-prevr)/(2*h);
        return;
    }
};



#endif // ODESOLVE_H
