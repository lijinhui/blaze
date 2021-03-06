import numpy as np
from math import sqrt

from blaze.datashape import dshape
from blaze.rts.funcs import PythonFn, install, lift
from blaze.engine import executors
from blaze import metadata as md

from numexpr import evaluate

from blaze.expr.ops import array_like
from blaze.metadata import aligned

from blaze.ts.ucr_dtw import ucr
#from blaze.ooc import linalg

# evaluating this function over a term has a cost, in the future this
# might be things like calculations for FLOPs associated with the
# operation. Could be a function of the shape of the term Most of the
# costs of BLAS functions can calculated a priori

zerocost = lambda term: 0

#------------------------------------------------------------------------
# Function Library
#------------------------------------------------------------------------

# Anatomy of a Blaze Function Def
# -------------------------------

#   ATerm Pattern Matcher
#            |                      +-- Type Signature
#            |                      |
#            v                      v
# @lift('Mul(<term>,<term>)', '(a,a) -> a', {
#     'types'   : {'a': array_like},     <- Type Constraints
#     'metadata': {'a': aligned, local}, <- Metadata Constraint
# }, costfn)
# def multiply(a, b):
#     return np.multipy(a, b)
#    |                     |
#    +---------------------+
#    Function Implementation
#
#
# If we were to "read" this definition in English. It would read:
#
#  > Here is a function called Mul, it matches any expression of two
#  > graph nodes. In addition the two graph nodes must be of the same
#  > datashapes (a) or unify/broadcast to the same type (a,a). It
#  > returns a result of the same type of the operands. In addition the
#  > two arguments must have metadata annotating them both as aligned
#  > memory and in system local memory. If the expression in question
#  > matches these criterion than I can tell you that you can perfom
#  > this operation in (n*m) FLOPS where n and m are the size of your
#  > input array, if you can't find a better implementaion than that use
#  > this function implementation!

#------------------------------------------------------------------------

@lift('dtw(<term>, <term>, <term>, <term>)', '(a,a,float,int) -> b', {
    'types' : {'a': array_like},
    'metadata': {},
})
def dtw(d1, d2, s, n):
    return ucr.dtw(d1, d2, s, n)

@lift('dot(<term>, <term>)', '(a,a) -> a', {
    'types'    : {'a' : array_like},
    'metadata' : {'a' : md.c_contigious},
})
def dot(a1, a2):
    return linalg.dot(a1, a2)

#------------------------------------------------------------------------

@lift('Add(<term>,<term>)', '(a,a) -> a')
def add(a, b):
    return np.add(a, a)

@lift('Mul(<term>,<term>)', '(a,a)-> a', {
    'types'   : {'a': array_like},
    'metadata': {},
})
def multiply(a, b):
    return np.multipy(a, b)


@lift('Pow(<term>,<term>)', '(a,a) -> a', {
    'types'   : {'a': array_like},
    'metadata': {},
})
def power(a, b):
    return np.power(a, b)

@lift('Abs(<term>)', 'a -> a', {
    'types'   : {'a': array_like},
    'metadata': {},
})
def abs(a, b):
    return np.abs(a, b)
