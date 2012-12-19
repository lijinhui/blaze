import ast

import numpy as np

import numba
from numba import transforms as numba_transforms
from numba import decorators
from numba.ufunc_builder import UFuncBuilder
from numba.minivect import minitypes

import blaze
import blaze.idx
from blaze.expr import visitor
from blaze.expr import ops
from blaze.expr import paterm
from blaze.datashape import coretypes
from blaze.engine import pipeline
from blaze.engine import executors
from blaze.sources import canonical
from blaze import plan

from blaze.datashape import datashape
from blaze import Table, NDTable, Array, NDArray

from numbapro.vectorize import Vectorize

class GraphToAst(visitor.BasicGraphVisitor):
    """
    Convert a blaze graph to a Python AST.
    """

    binop_to_astop = {
        ops.Add: ast.Add,
        ops.Mul: ast.Mult,
    }

    def __init__(self):
        super(GraphToAst, self).__init__()
        self.ufunc_builder = UFuncBuilder()

    def App(self, app):
        if app.operator.arity == 2:
            op = binop_to_astop.get(type(app.operator), None)
            if op is not None:
                left, right = self.visit(app.operator.children)
                return ast.BinOp(left=left, op=op(), right=right)

        return self.Unknown(app)

    def Unknown(self, tree):
        return self.ufunc_builder.register_operand(tree)


class ATermToAstTranslator(visitor.GraphTranslator):
    """
    Convert an aterm graph to a Python AST.
    """

    opname_to_astop = {
        'add': ast.Add,
        'mul': ast.Mult,
    }

    nesting_level = 0

    def __init__(self, executors, blaze_operands):
        super(ATermToAstTranslator, self).__init__()
        self.ufunc_builder = UFuncBuilder()
        self.executors = executors
        self.blaze_operands = blaze_operands # term -> blaze graph object

    def get_blaze_op(self, term):
        term_id = term.annotation.meta[0].label
        return self.blaze_operands[term_id]

    def register(self, graph, result, lhs=None):
        if lhs is not None:
            assert self.nesting_level == 0

        if self.nesting_level == 0:
            # Bottom of graph that we can handle
            operands = self.ufunc_builder.operands
            pyast_function = self.ufunc_builder.build_ufunc_ast(result)
            # print getsource(pyast_function)
            py_ufunc = self.ufunc_builder.compile_to_pyfunc(pyast_function,
                                                            globals={'np': np})

            executor = build_executor(py_ufunc, pyast_function, operands, graph)
            self.executors[id(executor)] = executor

            if lhs is not None:
                operands.append(lhs)
                datashape = lhs.annotation.ty #self.get_blaze_op(lhs).datashape
            else:
                # blaze_operands = [self.get_blaze_op(op) for op in operands]
                # datashape = coretypes.broadcast(*blaze_operands)
                datashape = graph.annotation.ty

            annotation = paterm.AAnnotation(
                ty=datashape,
                annotations=[id(executor), 'numba', bool(lhs)]
            )
            appl = paterm.AAppl(paterm.ATerm('Executor'), operands,
                                annotation=annotation)
            return appl

        self.result = result

        # Delete this node
        return None

    def match_assignment(self, app):
        """
        Handles slice assignemnt, e.g. out[:, :] = non_trivial_expr
        """
        assert self.nesting_level == 0

        lhs, rhs = app.args

        #
        ### Visit rhs
        #
        self.nesting_level += 1
        self.visit(rhs)
        rhs_result = self.result
        self.nesting_level -= 1

        #
        ### Visit lhs
        #
        # TODO: extend paterm.matches
        is_simple = (lhs.spine.label == 'Slice' and
                     lhs.args[0].spine.label == 'Array' and
                     all(v.label == "None" for v in lhs.args[1:]))
        if is_simple:
            self.nesting_level += 1
            lhs = self.visit(lhs)
            self.nesting_level -= 1
            lhs = self.ufunc_builder.operands.pop() # pop LHS from operands
        else:
            # LHS is complicated, let someone else (or ourselves!) execute
            # it independently
            # self.nesting_level is 0 at this point, so it will be registered
            # independently
            state = self.ufunc_builder.save()
            lhs = self.visit(lhs)
            lhs_result = self.result
            self.ufunc_builder.restore(state)

        #
        ### Build and return kernel if the rhs was an expression we could handle
        #
        if rhs_result:
            return self.register(app, rhs_result, lhs=lhs)
        else:
            app.args = [lhs, rhs]
            return app

    def handle_math_or_arithmetic(self, app, is_arithmetic):
        """
        Rewrite math and arithmetic operations.
        """
        opname = app.args[0].label.lower()
        if is_arithmetic:
            op = self.opname_to_astop.get(opname, None)
        else:
            # TODO: unhack
            if hasattr(np, opname):
                op = opname
            else:
                op = None

        type = plan.get_datashape(app)

        # Only accept scalars if we are already nested
        is_array = type.shape or self.nesting_level

        if op and is_array: # args = [op, ...]
            self.nesting_level += 1
            self.visit(app.args[1:])
            self.nesting_level -= 1

            # handle_arithmetic/handle_math
            if is_arithmetic:
                return self.handle_arithmetic(app, op)
            else:
                return self.handle_math(app, op)

        return self.unhandled(app)

    def handle_arithmetic(self, app, ast_op):
        """
        Handle unary and binary arithmetic
        """
        left, right = self.result
        result = ast.BinOp(left=left, op=ast_op(), right=right)
        return self.register(app, result)

    def handle_math(self, app, math_func_name):
        """
        Handle math calls by generate a call like `np.sin(x)`
        """
        operand = self.result
        func = ast.Attribute(value=ast.Name(id='np', ctx=ast.Load()),
                             attr=math_func_name,
                             ctx=ast.Load())
        math_call = ast.Call(func=func, args=[operand], keywords=[],
                             starargs=None, kwargs=None)
        return self.register(app, math_call)

    def AAppl(self, app):
        "Look for unops, binops and reductions and anything else we can handle"
        is_arithmetic = paterm.matches('Arithmetic;*', app.spine)
        if is_arithmetic or paterm.matches("Math;*", app.spine):
            return self.handle_math_or_arithmetic(app, is_arithmetic)

        elif paterm.matches('Slice;*', app.spine):
            array, start, stop, step = app.args
            if all(paterm.matches("None;*", op) for op in (start, stop, step)):
                return self.visit(array)

        elif paterm.matches("Assign;*", app.spine):
            return self.match_assignment(app)

        elif paterm.matches("Array;*", app.spine) and self.nesting_level:
            self.maybe_operand(app)
            if self.nesting_level:
                return None
            return app

        return self.unhandled(app)

    def AInt(self, constant):
        self.result = ast.Num(n=constant.n)
        return constant

    AFloat = AInt

    def maybe_operand(self, aterm):
        if self.nesting_level:
            self.result = self.ufunc_builder.register_operand(aterm)

    def unhandled(self, aterm):
        "An term we can't handle, scan for sub-trees"
        nesting_level = self.nesting_level
        state = self.ufunc_builder.save()

        self.nesting_level = 0
        self.visitchildren(aterm)
        self.nesting_level = nesting_level
        self.ufunc_builder.restore(state)

        self.maybe_operand(aterm)
        return aterm


def build_executor(py_ufunc, pyast_function, operands,
                   aterm_subgraph_root, strategy='chunked'):
    """ Build a ufunc and an wrapping executor from a Python AST """
    result_dtype = unannotate_dtype(aterm_subgraph_root)
    operand_dtypes = map(unannotate_dtype, operands)

    vectorizer = Vectorize(py_ufunc)
    vectorizer.add(restype=minitype(result_dtype),
                   argtypes=map(minitype, operand_dtypes))
    ufunc = vectorizer.build_ufunc()

    # Get a string of the operation for debugging
    return_stat = pyast_function.body[0]
    operation = getsource(return_stat.value)

    # TODO: build an executor tree and substitute where we can evaluate
    executor = executors.ElementwiseLLVMExecutor(
        strategy,
        ufunc,
        operand_dtypes,
        result_dtype,
        operation=operation,
    )

    return executor

def getsource(ast):
    from meta import asttools
    return asttools.dump_python_source(ast).strip()

def unannotate_dtype(aterm):
    """ Takes a term with a datashape annotation and returns the NumPy
    dtype associate with it

    >>> term
    x{dshape("2, 2, int32")}
    >>> unannotate_dtype(term)
    int32
    """
    # unpack the annotation {'s': 'int32'}
    unpack = paterm.matches('dshape(s);*', aterm.annotation['type'])
    ds_str = unpack['s']
    dshape = datashape(ds_str.s)

    dtype = coretypes.to_dtype(dshape)
    return dtype

def minitype(dtype):
    return minitypes.map_dtype(dtype)

def substitute_llvm_executors(aterm_graph, executors, operands):
    translator = ATermToAstTranslator(executors, operands)
    return translator.visit(aterm_graph)
