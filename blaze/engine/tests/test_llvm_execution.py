from blaze import *
from blaze.datashape import datashape
from blaze.engine import pipeline, llvm_execution

from unittest import skip

def convert_graph(lazy_blaze_graph):
    # Convert blaze graph to ATerm graph
    p = pipeline.Pipeline(have_numbapro=True)
    context, aterm_graph = p.run_pipeline(lazy_blaze_graph)
    return context["instructions"], context["executors"], context["symbols"]

#@skip("Unstable")
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

#@skip("Unstable")
def test_execution():
    """
    >>> test_execution()
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

if __name__ == '__main__':
#   test_conversion()
#   test_execution()

    import doctest
    doctest.testmod()
