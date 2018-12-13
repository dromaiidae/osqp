import numpy as np
import datetime

FILENAME = "../data/matrix_4k.mat"
DIMENSION = 4000

dense_array_to_decomp = np.fromfile(FILENAME)
array = np.reshape(dense_array_to_decomp, (DIMENSION, DIMENSION))

start = datetime.datetime.now()
np.linalg.cholesky(array)
end = datetime.datetime.now()

print("Nanoseconds elapsed matrix2:", end.microsecond - start.microsecond)