import numpy as np
import datetime

FILENAME = "../data/matrix2.mat"
DIMENSION = 450

dense_array_to_decomp = np.fromfile(FILENAME)
array = np.reshape(dense_array_to_decomp, (DIMENSION, DIMENSION))

for i in range(5):
    start = datetime.datetime.now()
    np.linalg.cholesky(array)
    end = datetime.datetime.now()

    print("Nanoseconds 2 elapsed matrix:", end.microsecond - start.microsecond)
