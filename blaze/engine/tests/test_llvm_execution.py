import operator

import blaze
from blaze import *
from blaze.datashape import datashape
from blaze.engine import pipeline, llvm_execution

from unittest import skip

def convert_graph(lazy_blaze_graph):
    # Convert blaze graph to ATerm graph
    p = pipeline.Pipeline(have_numbapro=True)
    context, aterm_graph = p.run_pipeline(lazy_blaze_graph)
    return context["instructions"], context["executors"], context["symbols"]

def test_conversion():
    """
    >>> test_conversion()
    [LLVMExecutor(chunked, (op0 + (op1 * op2)))('%0' '%1' '%2')]
    ['%0', '%1', '%2']
    """
    a = NDArray([1, 2, 3, 4], datashape('2, 2, int32'))
    b = NDArray([5, 6, 7, 8], datashape('2, 2, float32'))
    c = NDArray([9, 10, 11, 12], datashape('2, 2, int32'))

    graph = a + b * c
    instructions, executors, symbols = convert_graph(graph)

    assert len(instructions) == len(executors) == 1
    print instructions
    print sorted(symbols)

def test_execution_simple():
    """
    >>> test_execution_simple()
    [  46.   62.   80.  100.]
    """
    a = NDArray([1, 2, 3, 4], datashape('4, float32'))
    b = NDArray([5, 6, 7, 8], datashape('4, float32'))
    c = NDArray([9, 10, 11, 12], datashape('4, float32'))
    out = NDArray([0, 0, 0, 0], datashape('4, float32'))

    graph = a + b * c
    out[:] = graph

    # print list(out.data.ca), hex(out.data.ca.leftover_array.ctypes.data)
    # print "*" * 100
    # print out.data.ca.leftover_array.dtype
    print out.data.ca

def test_execution():
    """
    >>> test_execution()
    """
    for size in [1, 3, 100, 100000]:
        for op in [operator.add, operator.mul]:
            for dtype in ["float32", "float64", "int32", "int64"]:
                dshape = datashape("%d, %s" % (size, dtype))

                a = NDArray(range(size), dshape)
                b = NDArray(range(size, size * 2), dshape)

                # blaze.zeros(dshape) returns an Array
                out = NDArray([0] * size, dshape)
                out[:] = op(a, b)

                c1 = np.arange(size, dtype=dtype)
                c2 = np.arange(size, size * 2, dtype=dtype)
                out2 = carray.carray(op(c1, c2), dtype=np.dtype(dtype))

                assert blaze.allclose(out, out2), (size, op, dshape)


if __name__ == '__main__':
#   test_conversion()
    test_execution()

    import doctest
    doctest.testmod()
