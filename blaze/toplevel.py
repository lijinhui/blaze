import os, os.path

from urlparse import urlparse
from params import params, to_cparams
from params import params as _params
from sources.sql import SqliteSource
from sources.chunked import CArraySource, CTableSource

from table import NDArray, Array, NDTable, Table
from blaze.datashape.coretypes import from_numpy, to_numpy, TypeVar, Fixed
from blaze import carray, dshape as _dshape

import numpy as np

__all__ = ["open", "zeros", "ones", "fromiter", "loadtxt", "allclose"]

def open(uri=None, mode='a'):
    """Open a Blaze object via an `uri` (Uniform Resource Identifier).

    Parameters
    ----------
    uri : str
        Specifies the URI for the Blaze object.  It can be a regular file too.

    mode : the open mode (string)
        Specifies the mode in which the object is opened.  The supported
        values are:

          * 'r' for read-only
          * 'w' for emptying the previous underlying data
          * 'a' for allowing read/write on top of existing data

    Returns
    -------
    out : an Array or Table object.

    """
    ARRAY = 1
    TABLE = 2

    if uri is None:
        source = CArraySource()
    else:
        uri = urlparse(uri)

        if uri.scheme == 'carray':
            path = os.path.join(uri.netloc, uri.path[1:])
            parms = params(storage=path)
            source = CArraySource(params=parms)
            structure = ARRAY

        if uri.scheme == 'ctable':
            path = os.path.join(uri.netloc, uri.path[1:])
            parms = params(storage=path)
            source = CTableSource(params=parms)
            structure = TABLE

        elif uri.scheme == 'sqlite':
            path = os.path.join(uri.netloc, uri.path[1:])
            parms = params(storage=path or None)
            source = SqliteSource(params=parms)
            structure = TABLE

        else:
            # Default is to treat the URI as a regular path
            parms = params(storage=uri.path)
            source = CArraySource(params=parms)
            structure = ARRAY

    # Don't want a deferred array (yet)
    # return NDArray(source)
    if structure == ARRAY:
        return Array(source)
    elif structure == TABLE:
        return NDTable(source)

# These are like NumPy equivalent except that they can allocate
# larger than memory.

def zeros(dshape, params=None):
    """ Create an Array and fill it with zeros.

    Parameters
    ----------
    dshape : str, blaze.dshape instance
        Specifies the datashape of the outcome object.
    params : blaze.params object
        Any parameter supported by the backend library.

    Returns
    -------
    out : an Array object.

    """
    if isinstance(dshape, basestring):
        dshape = _dshape(dshape)
    shape, dtype = to_numpy(dshape)
    cparams, rootdir, format_flavor = to_cparams(params or _params())
    if rootdir is not None:
        carray.zeros(shape, dtype, rootdir=rootdir, cparams=cparams)
        return open(rootdir)
    else:
        source = CArraySource(carray.zeros(shape, dtype, cparams=cparams),
                              dshape=dshape,
                              params=params)
        return Array(source)

def ones(dshape, params=None):
    """ Create an Array and fill it with ones.

    Parameters
    ----------
    dshape : str, blaze.dshape instance
        Specifies the datashape of the outcome object.
    params : blaze.params object
        Any parameter supported by the backend library.

    Returns
    -------
    out : an Array object.

    """
    if isinstance(dshape, basestring):
        dshape = _dshape(dshape)
    shape, dtype = to_numpy(dshape)
    cparams, rootdir, format_flavor = to_cparams(params or _params())
    if rootdir is not None:
        carray.ones(shape, dtype, rootdir=rootdir, cparams=cparams)
        return open(rootdir)
    else:
        source = CArraySource(carray.ones(shape, dtype, cparams=cparams),
                              params=params)
        return Array(source)

def fromiter(iterable, dshape, params=None):
    """ Create an Array and fill it with values from `iterable`.

    Parameters
    ----------
    iterable : iterable object
        An iterable object providing data for the carray.
    dshape : str, blaze.dshape instance
        Specifies the datashape of the outcome object.  Only 1d shapes
        are supported right now. When the `iterator` should return an
        unknown number of items, a ``TypeVar`` can be used.
    params : blaze.params object
        Any parameter supported by the backend library.

    Returns
    -------
    out : an Array object.

    """
    if isinstance(dshape, basestring):
        dshape = _dshape(dshape)
    shape, dtype = dshape.parameters[:-1], dshape.parameters[-1]
    # Check the shape part
    if len(shape) > 1:
        raise ValueError("shape can be only 1-dimensional")
    length = shape[0]
    count = -1
    if type(length) == TypeVar:
        count = -1
    elif type(length) == Fixed:
        count = length.val

    dtype = dtype.to_dtype()
    # Now, create the Array itself (using the carray backend)
    cparams, rootdir, format_flavor = to_cparams(params or _params())
    if rootdir is not None:
        carray.fromiter(iterable, dtype, count=count,
                        rootdir=rootdir, cparams=cparams)
        return open(rootdir)
    else:
        ica = carray.fromiter(iterable, dtype, count=count, cparams=cparams)
        source = CArraySource(ica, params=params)
        return Array(source)

def loadtxt(filetxt, storage):
    """ Convert txt file into Blaze native format """
    Array(np.loadtxt(filetxt), params=params(storage=storage))

def _ndarray(a):
    if not isinstance(a, NDArray):
        a = NDArray(a)
    return a

def allclose(a, b, rtol=1e-05, atol=1e-08):
    """
    Returns True if two arrays are element-wise equal within a tolerance.

    The tolerance values are positive, typically very small numbers.  The
    relative difference (`rtol` * abs(`b`)) and the absolute difference
    `atol` are added together to compare against the absolute difference
    between `a` and `b`.

    If either array contains one or more NaNs, False is returned.
    Infs are treated as equal if they are in the same place and of the same
    sign in both arrays.

    Parameters
    ----------
    a, b : array_like
        Input arrays to compare.
    rtol : float
        The relative tolerance parameter (see Notes).
    atol : float
        The absolute tolerance parameter (see Notes).

    Returns
    -------
    allclose : bool
        Returns True if the two arrays are equal within the given
        tolerance; False otherwise.

    See Also
    --------
    all, any, alltrue, sometrue

    Notes
    -----
    If the following equation is element-wise True, then allclose returns
    True.

     absolute(`a` - `b`) <= (`atol` + `rtol` * absolute(`b`))

    The above equation is not symmetric in `a` and `b`, so that
    `allclose(a, b)` might be different from `allclose(b, a)` in
    some rare cases.

    Examples
    --------
    >>> blaze.allclose([1e10,1e-7], [1.00001e10,1e-8])
    False
    >>> blaze.allclose([1e10,1e-8], [1.00001e10,1e-9])
    True
    >>> blaze.allclose([1e10,1e-8], [1.0001e10,1e-9])
    False
    >>> blaze.allclose([1.0, np.nan], [1.0, np.nan])
    False
    """
    a, b = _ndarray(a), _ndarray(b)
    return blaze_all(blaze_abs(a - b) <= atol + rtol * blaze_abs(b))

def blaze_all(a, axis=None, out=None):
    """
    Test whether all array elements along a given axis evaluate to True.
    """
    a = _ndarray(a)
    return a.all(axis=axis, out=out)

def blaze_any(a, axis=None, out=None):
    """
    Test whether any array elements along a given axis evaluate to True.
    """
    a = _ndarray(a)
    return a.any(axis=axis, out=out)

def blaze_abs(a, axis=None, out=None):
    """
    Returns the absolute value element-wise.
    """
    a = _ndarray(a)
    return a.abs(axis=axis, out=out)