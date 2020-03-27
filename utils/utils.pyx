''''
 __author__: bishwarup
 created: Wednesday, 25th March 2020 11:11:08 pm
 ''''

cimport cython
from libc.math cimport sqrt
import numpy
cimport numpy

cpdef int calc_area((int, int, int, int) coord):
    cdef long int area
    area = (coord[2] - coord[0]) * (coord[3] - coord[1])
    return area

cpdef double distance((int, int) x1, (int, int) x2):
    cdef double d
    d = sqrt((x1[0] - x2[0]) * ((x1[0] - x2[0])) + (x1[1] - x2[1]) * ((x1[1] - x2[1])) )
    return d

ctypedef numpy.uint8_t DTYPE_t
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef spawn((int, int) shape, numpy.ndarray[DTYPE_t, ndim = 2, mode = 'c'] grid, int fill):
    cpdef bint done = False
    
    cdef Py_ssize_t start_x, start_y, i, j
    cdef Py_ssize_t max_x = grid.shape[0]
    cdef Py_ssize_t max_y = grid.shape[1]
    cdef long int area
    cdef (int, int, int, int) bbox
    
    while not done:
        start_x = numpy.random.randint(0, max_x - shape[0])
        start_y = numpy.random.randint(0, max_y - shape[1])
        
        for i in range(start_x, max_x - shape[0]):
            for j in range(start_y, max_y - shape[1]):
                if grid[i, j] == 1: 
                    continue
                else:
                    area = numpy.sum(grid[i:(i+shape[0]), j:(j+shape[1])])
                    if area == 0:
                        grid[i:(i+shape[0]), j:(j+shape[1])] = fill
                        done = True
                        break
                    else:
                        continue
            if done:
                break
        bbox = (i, j, i + shape[0], j + shape[1])
        return bbox, grid