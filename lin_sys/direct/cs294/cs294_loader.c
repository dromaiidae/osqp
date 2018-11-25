#include "lib_handler.h"
#include "cs294_loader.h"

#include "glob_opts.h"
#include "constants.h"

#ifdef IS_WINDOWS
#define CS294LIBNAME "mkl_rt." SHAREDLIBEXT
#else
#define CS294LIBNAME "libmkl_rt." SHAREDLIBEXT
#endif

typedef void (*voidfun)(void);

voidfun lh_load_sym (soHandle_t h, const char *symName);


// Interfaces for Cs294 functions
typedef void (*cs294_t)(void**, const c_int*, const c_int*, const c_int*,
                          const c_int*, const c_int*, const c_float*,
                          const c_int*, const c_int*, const c_int*,
                          const c_int*, c_int*, const c_int*, c_float*,
                          c_float*, const c_int*);
typedef int (*cs294_set_ifl_t)(int);
typedef int (*cs294_get_mt_t)();


// Handlers are static variables
static soHandle_t Cs294_handle = OSQP_NULL;
static cs294_t func_cs294 = OSQP_NULL;
static cs294_set_ifl_t func_cs294_set_interface_layer = OSQP_NULL;
static cs294_get_mt_t func_cs294_get_max_threads = OSQP_NULL;

// Wrappers for loaded Cs294 function handlers
void cs294(void** pt, const c_int* maxfct, const c_int* mnum,
                  const c_int* mtype, const c_int* phase, const c_int* n,
                  const c_float* a, const c_int* ia, const c_int* ja,
                  const c_int* perm, const c_int* nrhs, c_int* iparm,
                  const c_int* msglvl, c_float* b, c_float* x,
                  const c_int* error) {
	if(func_cs294){
            // Call function cs294 only if it has been initialized
	    func_cs294(pt, maxfct, mnum, mtype, phase, n, a, ia, ja,
			 perm, nrhs, iparm, msglvl, b, x, error);
	}
	else
	{
#ifdef PRINTING
		c_eprint("Cs294 not loaded correctly");
#endif
	}
}

c_int cs294_set_interface_layer(c_int code) {
    return (c_int)func_cs294_set_interface_layer((int)code);
}

c_int cs294_get_max_threads() {
    return (c_int)func_cs294_get_max_threads();
}


c_int lh_load_cs294(const char* libname) {
    // DEBUG
    // if (Cs294_handle) return 0;

    // Load Cs294 library
    if (libname) {
        Cs294_handle = lh_load_lib(libname);
    } else { /* try a default library name */
        Cs294_handle = lh_load_lib(CS294LIBNAME);
    }
    if (!Cs294_handle) return 1;

    // Load Cs294 functions
    func_cs294 = (cs294_t)lh_load_sym(Cs294_handle, "pardiso");
    if (!func_cs294) return 1;

    func_cs294_set_interface_layer = (cs294_set_ifl_t)lh_load_sym(Cs294_handle,
                                                    "MKL_Set_Interface_Layer");
    if (!func_cs294_set_interface_layer) return 1;

    func_cs294_get_max_threads = (cs294_get_mt_t)lh_load_sym(Cs294_handle,
                                                    "MKL_Get_Max_Threads");
    if (!func_cs294_get_max_threads) return 1;

    return 0;
}

c_int lh_unload_cs294() {

    if (Cs294_handle == OSQP_NULL) return 0;

    return lh_unload_lib(Cs294_handle);

    /* If multiple OSQP objects are laoded, the lines below cause a crash */
    // Cs294_handle = OSQP_NULL;
    // func_cs294 = OSQP_NULL;
    // func_cs294_set_interface_layer = OSQP_NULL;
    // func_cs294_get_max_threads = OSQP_NULL;

}
