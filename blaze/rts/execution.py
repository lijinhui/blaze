# a + b * c

# ATerm Graph
# ===========
#
#   Arithmetic(
#     Add
#   , Array(){dshape("3, int64"), 45340864}
#   , Arithmetic(
#         Mul
#       , Array(){dshape("3, int64"), 45340792}
#       , Array(){dshape("3, int64"), 45341584}
#     ){dshape("3, int64"), 45264528}
#   ){dshape("3, int64"), 45264432}

# Execution Plan
# ==============

# vars %a %b %c
# %0 := ElemwiseNumpy{np.mul,nogil}(%b, %c)
# %0 := ElemwiseNumpy{np.add,nogil,inplace}(%0, %a)

# Responsibilities
# - allocate memory blocks on Blaze heap for LHS
# - determine whether to do operation inplace or to store the
#   output in a temporary
#
# - Later: handle kernel fusion
# - Much Later: handle GPU access & thread control

from blaze.rts.storage import Heap

# =================================
# The main Blaze RTS execution loop
# =================================

# Invokes Executor functions and handles memory management from external
# sources to allocate on, IOPro allocators, SQL Queries, ZeroMQ...

# TOOD: write in Cython
def execplan(context, plan, symbols):
    """ Takes a list of of instructions from the Pipeline and
    then allocates the necessary memory needed for the
    intermediates are temporaries. Then executes the plan
    returning the result. """

    instructions = context["instructions"]  # [ Instruction(...) ]
    symbols = context["symbols"]            # { %0 -> Array(...){...}
    operands = context["operand_dict"]      # { Array(...){...} -> Blaze Array }

    def getop(symbol):
        term = symbols[symbol]
        term_id = term.annotation.meta[0].label
        op = operands[term_id]
        return op

    h = Heap()
    ret = None

    for instruction in instructions:
        ops = map(getop, instruction.args)

        if not instruction.lhs:
            lhs = h.allocate(instruction.lhs.size())
        else:
            lhs = getop(instruction.lhs)

        ret = instruction.execute(ops, lhs)

    h.finalize()
    return ret
