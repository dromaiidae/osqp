import numpy as np
import datetime
from sksparse.cholmod import cholesky
from scipy.sparse import csc_matrix

FILENAME = "../data/matrix_160.mat"
DIMENSION = 160

dense_array_to_decomp = np.fromfile(FILENAME)
array = csc_matrix(np.reshape(dense_array_to_decomp, (DIMENSION, DIMENSION)))

start = datetime.datetime.now()
cholesky(array)
end = datetime.datetime.now()

print("Cholesky Nanoseconds elapsed matrix2:", end.microsecond - start.microsecond)
