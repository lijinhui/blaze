cdef extern from "Python.h":
    char* PyString_AsString(object string)

cdef extern from "stdarg.h":
    ctypedef struct va_list:
        pass
    ctypedef struct fake_type:
        pass
    void va_start(va_list, void* arg)
    void* va_arg(va_list, fake_type)
    void va_end(va_list)
    fake_type int_type "int"

cdef extern from "stdio.h":
    ctypedef FILE
    enum: stdout

cdef extern from "aterm1.h":
    enum: AT_FREE
    enum: AT_APPL
    enum: AT_INT
    enum: AT_REAL
    enum: AT_LIST
    enum: AT_PLACEHOLDER
    enum: AT_BLOB
    enum: AT_SYMBOL
    enum: MAX_ARITY

    ctypedef long MachineWord
    ctypedef unsigned long HashNumber
    ctypedef unsigned long header_type

    ctypedef struct __ATerm:
        header_type header
        void *next

    ctypedef union _ATerm:
        header_type header
        __ATerm aterm
        void* subaterm[MAX_ARITY+3]
        MachineWord  word[MAX_ARITY+3]

    ctypedef _ATerm *ATerm
    ctypedef ATerm ATermInt
    ctypedef ATerm ATermReal

    ctypedef int *FILE

    void ATinit (int argc, char *argv[], ATerm *bottomOfStack)
    ATbool ATmatch(ATerm t, char *pattern, ...)

    #--------------------------------------------------------------------
    # Memory management
    #--------------------------------------------------------------------

    void ATprotect(ATerm *TrmPtr)
    void ATunprotect(ATerm *TrmPtr)

    #--------------------------------------------------------------------
    # Parsing and ATerm building
    #--------------------------------------------------------------------

    int ATgetType(ATerm t)

    int ATprintf(char *format, ...)
    int ATfprintf(int stream, char *format, ...)
    char* ATwriteToString(ATerm t)

    ATerm ATmake(char *pattern, ...)
    ATerm ATmakeTerm(ATerm pat, ...)
    ATerm _ATmakeInt "ATmakeInt" (int val)
    ATerm _ATmakeReal "ATmakeReal" (double val)

    ATerm ATvmake(char *pat)
    ATerm ATvmakeTerm(ATerm pat)
    void  AT_vmakeSetArgs(va_list *args)

    ATerm _ATreadFromString "ATreadFromString" (char *string)
    ATerm ATreadFromSharedString(char *s, int size)

    ATerm ATsetAnnotation(ATerm t, ATerm label, ATerm anno)
    ATerm ATgetAnnotation(ATerm t, ATerm label)

    void ATsetWarningHandler(void (*handler)(char *format, va_list args))
    void ATsetErrorHandler(void (*handler)(char *format, va_list args))
    void ATsetAbortHandler(void (*handler)(char *format, va_list args))

    ATbool ATisEqual(ATerm t1, ATerm t2)
    ATbool AT_isDeepEqual(ATerm t1, ATerm t2)
    ATbool ATisEqualModuloAnnotations(ATerm t1, ATerm t2)

    ctypedef enum ATbool:
        ATfalse = 0
        ATtrue  = 1

cdef extern from "utils.h":
    int subterms(ATerm t)
    ATerm * next_subterm(ATerm t, int i)
    ATerm * annotations(ATerm t)