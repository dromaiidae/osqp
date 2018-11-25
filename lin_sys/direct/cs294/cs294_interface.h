#ifndef CS294_H
#define CS294_H

#ifdef __cplusplus
extern "C" {
#endif

#include "lin_alg.h"
#include "kkt.h"

/**
 * Cs294 solver structure
 *
 * NB: If we use Cs294, we suppose that EMBEDDED is not enabled
 */
typedef struct cs294 cs294_solver;

struct cs294 {
    enum linsys_solver_type type;

    /**
     * @name Functions
     * @{
     */
    c_int (*solve)(struct cs294 * self, c_float * b, const OSQPSettings * settings);

    void (*free)(struct cs294 * self); ///< Free workspace (only if desktop)

    c_int (*update_matrices)(struct cs294 * self, const csc *P, const csc *A, const OSQPSettings *settings); ///< Update solver matrices
    c_int (*update_rho_vec)(struct cs294 * self, const c_float * rho_vec, const c_int m); ///< Update solver matrices

    c_int nthreads;
    /** @} */


    /**
     * @name Attributes
     * @{
     */
    // Attributes
    csc *KKT;         ///< KKT matrix (in CSR format!)
    c_int *KKT_i;     ///< KKT column ondeces in 1-indexing for Cs294
    c_int *KKT_p;     ///< KKT row pointers in 1-indexing for Cs294
    c_float *bp;      ///< workspace memory for solves (rhs)

    // Cs294 variables
    void *pt[64];     ///< internal solver memory pointer pt
    c_int iparm[64];  ///< Cs294 control parameters
    c_int n;          ///< dimension of the linear system
    c_int mtype;      ///< matrix type (-2 for real and symmetric indefinite)
    c_int nrhs;       ///< number of right-hand sides (1 for our needs)
    c_int maxfct;     ///< maximum number of factors (1 for our needs)
    c_int mnum;       ///< indicates matrix for the solution phase (1 for our needs)
    c_int phase;      ///< control the execution phases of the solver
    c_int error;      ///< the error indicator (0 for no error)
    c_int msglvl;     ///< Message level information (0 for no output)
    c_int idum;       ///< dummy integer
    c_float fdum;     ///< dummy float

    // These are required for matrix updates
    c_int * Pdiag_idx, Pdiag_n;  ///< index and number of diagonal elements in P
    c_int * PtoKKT, * AtoKKT;    ///< Index of elements from P and A to KKT matrix
    c_int * rhotoKKT;            ///< Index of rho places in KKT matrix

    /** @} */
};


/**
 * Initialize Cs294 Solver
 *
 * @param  P      Cost function matrix (upper triangular form)
 * @param  A      Constraints matrix
 * @param  sigma   Algorithm parameter. If polish, then sigma = delta.
 * @param  rho_vec Algorithm parameter. If polish, then rho_vec = OSQP_NULL.
 * @param  polish Flag whether we are initializing for polish or not
 * @return        Initialized private structure
 */
cs294_solver *init_linsys_solver_cs294(const csc * P, const csc * A, c_float sigma, c_float * rho_vec, c_int polish);


/**
 * Solve linear system and store result in b
 * @param  s        Linear system solver structure
 * @param  b        Right-hand side
 * @param  settings OSQP solver settings
 * @return          Exitflag
 */
c_int solve_linsys_cs294(cs294_solver * s, c_float * b, const OSQPSettings * settings);


/**
 * Update linear system solver matrices
 * @param  s        Linear system solver structure
 * @param  P        Matrix P
 * @param  A        Matrix A
 * @param  settings Solver settings
 * @return          Exitflag
 */
c_int update_linsys_solver_matrices_cs294(cs294_solver * s,
		const csc *P, const csc *A, const OSQPSettings *settings);


/**
 * Update rho parameter in linear system solver structure
 * @param  s   Linear system solver structure
 * @param  rho new rho value
 * @param  m   number of constraints
 * @return     exitflag
 */
c_int update_linsys_solver_rho_vec_cs294(cs294_solver * s, const c_float * rho_vec, const c_int m);


/**
 * Free linear system solver
 * @param s linear system solver object
 */
void free_linsys_solver_cs294(cs294_solver * s);

#ifdef __cplusplus
}
#endif

#endif
