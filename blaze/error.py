class ChunkingException(BaseException):
    pass

class NoSuchChecksum(ValueError):
    pass

class ChecksumMismatch(RuntimeError):
    pass

class FileNotFound(IOError):
    pass

class BlazeException(Exception):
    "Base class for blaze exceptions"

# for Numba
class ExecutionError(BlazeException):
    """
    Raised when we are unable to execute a certain lazy or immediate
    expression.
    """

# for the RTS
class NoDispatch(BlazeException):
    def __init__(self, aterm):
        self.aterm = aterm
    def __str__(self):
        return "No implementation for '%r'" % self.aterm

# for the RTS
class InvalidLibraryDefinton(BlazeException):
    pass

class NotNumpyCompatible(BlazeException):
    """
    Raised when we try to convert a datashape into a NumPy dtype
    but it cannot be ceorced.
    """
    pass
