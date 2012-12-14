"""
Cython bindings for libAterm, a small no-dependency C library for
manipulating and parsing ATerm expressions.

Docs
----

* http://strategoxt.org/Tools/ATermLibrary
* http://www.meta-environment.org/doc/books/technology/aterm-guide/aterm-guide.html

"""

#------------------------------------------------------------------------
# See caterm.pxd for type definitions
#------------------------------------------------------------------------

from libc.stdlib cimport malloc, free
from blaze.blaze cimport *

from copy import copy

# singleton empty ATerm
cdef ATerm ATEmpty


#------------------------------------------------------------------------
# Dumb aterm library wrappers that handle errors
#------------------------------------------------------------------------


cdef inline ATerm ATreadFromString(char *value) except NULL:
    print "reading...", value
    cdef ATerm result = _ATreadFromString(value)
    if result == ATEmpty:
        raise InvalidATerm(value)
    return result

cdef inline ATerm ATmakeInt(int value) except NULL:
    cdef ATerm result = _ATmakeInt(value)
    if result == ATEmpty:
        raise InvalidATerm(value)
    return result

cdef inline ATerm ATmakeReal(double value) except NULL:
    cdef ATerm result = _ATmakeReal(value)
    if result == ATEmpty:
        raise InvalidATerm(value)
    return result


#------------------------------------------------------------------------
# Python ATerm wrapper
#------------------------------------------------------------------------


cdef class PyATerm:

    cdef ATerm a
    cdef char* _repr
    cdef object pattern

    def __init__(self, pattern):
        self.pattern = pattern
        self._repr = pattern

    @classmethod
    def from_pattern(cls, pattern):
        cdef ATerm a

        if isinstance(pattern, basestring):
            a = ATreadFromString(pattern)
            # self._repr = ATwriteToString(a) # this segfaults
        elif isinstance(pattern, int):
            a = ATmakeInt(pattern)
        elif isinstance(pattern, float):
            a = ATmakeReal(pattern)

        return cls.from_aterm(<Py_uintptr_t> a, pattern)

    @classmethod
    def from_aterm(cls, Py_uintptr_t aterm, pattern):
        cdef ATerm a = <ATerm> aterm
        cdef PyATerm result = cls(pattern)
        result.a = <ATerm> aterm
        ATprotect(&result.a)
        return result

    @property
    def typeof(self):
        return ATgetType(self.a)

    def __iter__(self):
        cdef int arity = subterms(self.a)
        cdef ATerm* ptr
        accum = []

        for i in range(arity):
            ptr = next_subterm(self.a, i)
            accum.append(PyATerm(<int>ptr))

        return iter(accum)

    @property
    def annotations(self):
        return PyATerm(<int>annotations(self.a))

    def aset(self, bytes key, bytes value):
        """ Return a new ATerm annotated with the given key,
        value pair """
        cdef ATerm label = ATreadFromString(key)
        cdef ATerm anno = ATreadFromString(value)
        cdef ATerm copy = ATsetAnnotation(self.a, label, anno)
        return self.from_aterm(<Py_uintptr_t> copy, value)

    def aget(self, bytes key):
        """ Query a annotation of the term.  """
        cdef ATerm label = ATreadFromString(key)
        cdef ATerm value = ATgetAnnotation(self.a, label)
        if value == ATEmpty:
            raise NoAnnotation(key)
        else:
            return ATwriteToString(value)

    def amatch(self, char* pattern):
        """ Pattern match on annotations """
        for a in self.annotations:
            return a.matches(pattern)

    def amatch_all(self, list patterns):
        """ Pattern match on annotations """
        cdef char **cp = <char**>malloc(len(patterns) * sizeof(char*))

        try:
            for i,p in enumerate(patterns):
                cp[i] = <char*>patterns[i]

            for i in xrange(len(patterns)):
                matches = False
                for a in self.annotations:
                    matches |= a.matches(cp[i])

                # if not the nfall out
                if not matches:
                    return False

            # matched all patterns
            return True

        finally:
            free(cp)

    def __richcmp__(PyATerm self, PyATerm other, int op):
        cdef ATbool res

        if op == 2:
            if isinstance(other, PyATerm):
                res = ATisEqual(self.a, other.a)
        else:
            # TODO: lexicographic ordering from aterm2.h
            raise NotImplementedError

        if res == ATtrue:
            return True
        if res == ATfalse:
            return False

    def __dealloc__(self):
        "Mark the aterm deletable"
        # TODO: what if this thing is shared by someone else?
        ATunprotect(&self.a)

    def matches(self, pattern, capture=None):
        """
        Matches against ATerm patterns.

        >>> aterm('x').matches('<term>')
        True
        >>> aterm('f(1)').matches('<appl(1)>', [APPL])
        ('f',)
        >>> aterm('f("bar")').matches('f(<str>)', [STR])
        ('bar',)
        >>> aterm('f("bar", "awk")').matches('f(<str>, <str>)', [STR, STR])
        ('bar', 'awk')

        Alas, no metadata annotation ... at least out of the box.
        Ergo the reason for my half baked query language. Think
        I can roll it in here though
        """
        cdef object pcopy = copy(pattern)[:]
        cdef char* cpattern = PyString_AsString(pcopy)

        cdef ATbool res
        cdef char *c1, *c2, *c3, *c4, *c5
        #cdef ATerm *a1, *a2, *a3, *a4, *a5

        #if len(pattern) > 0:
            #raise ValueError("Empty pattern match")

        if capture is None:
            res = ATmatch(self.a, cpattern)

            if res == ATtrue:
                return True
            if res == ATfalse:
                return False

        # yeah, good stuff
        elif len(capture) == 1:
            res = ATmatch(self.a, pattern, &c1)
            return (c1,)
        elif len(capture) == 2:
            res = ATmatch(self.a, pattern, &c1, &c2)
            return (c1,c2)
        elif len(capture) == 3:
            res = ATmatch(self.a, pattern, &c1, &c2, &c3)
            return (c1,c2,c3)
        elif len(capture) == 4:
            res = ATmatch(self.a, pattern, &c1, &c2, &c3, &c4)
            return (c1,c2,c3,c4)
        elif len(capture) == 5:
            res = ATmatch(self.a, pattern, &c1, &c2, &c3, &c4, &c5)
            return (c1,c2,c3,c4,c5)
        else:
            raise ValueError("Up to 5 captures variables")

    def __repr__(self):
        return "aterm('%s')" % ATwriteToString(self.a)


#------------------------------------------------------------------------
# Exceptions
#------------------------------------------------------------------------

class InvalidATerm(SyntaxError):
    pass

class NoAnnotation(KeyError):
    pass

#------------------------------------------------------------------------
# Error Handling
#------------------------------------------------------------------------

cdef void error(char *format, va_list args) with gil:
    raise InvalidATerm(format)

# execute at module init
cdef ATerm bottomOfStack
ATinit(1, [], &bottomOfStack)

# Register error handlers
ATsetErrorHandler(error)
ATsetWarningHandler(error)
ATsetAbortHandler(error)

#------------------------------------------------------------------------
# Constants
#------------------------------------------------------------------------

FREE        = AT_FREE
APPL        = AT_APPL
INT         = AT_INT
REAL        = AT_REAL
LIST        = AT_LIST
PLACEHOLDER = AT_PLACEHOLDER
BLOB        = AT_BLOB
SYMBOL      = AT_SYMBOL
STR         = BLOB

#------------------------------------------------------------------------
# Toplevel
#------------------------------------------------------------------------

def aterm(pattern):
    return PyATerm.from_pattern(pattern)

def make(template, params=None):
    cdef char* ctemplate = PyString_AsString(template)

    cdef ATerm a1, a2, a3, a4, a5
    cdef ATerm res

    if len(params) == 0:
        raise ValueError("Use aterm() for construction with no arguments")
    if len(params) == 1:
        pa1 = PyATerm(params[0])
        res = ATmake(template, pa1.a)
    if len(params) == 2:
        pa1 = PyATerm(params[0])
        pa2 = PyATerm(params[1])
        res = ATmake(template, pa1.a, pa2.a)
    if len(params) == 3:
        pa1 = PyATerm(params[0])
        pa2 = PyATerm(params[1])
        pa3 = PyATerm(params[2])
        res = ATmake(template, pa1.a, pa2.a, pa3.a)

    return PyATerm(<int>res)

def matches(str s, PyATerm term):
    return term.matches(s)
