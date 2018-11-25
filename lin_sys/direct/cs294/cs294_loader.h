#ifndef CS294LOADER_H
#define CS294LOADER_H

#ifdef __cplusplus
extern "C" {
#endif



/**
 * Tries to load a shared library with Cs294.
 * Return a failure if the library cannot be loaded or not all Cs294 symbols are found.
 * @param libname The name under which the Cs294 lib can be found, or OSQP_NULL to use a default name (mkl_rt.SHAREDLIBEXT).
 * @return Zero on success, nonzero on failure.
 */
c_int lh_load_cs294(const char* libname);

/**
 * Unloads the loaded Cs294 shared library.
 * @return Zero on success, nonzero on failure.
 */
c_int lh_unload_cs294();


#ifdef __cplusplus
}
#endif

#endif /*CS294LOADER_H*/
