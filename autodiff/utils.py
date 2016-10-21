import gc
import opcode
import inspect
import theano
import numpy as np

from collections import OrderedDict
from inspect import getcallargs

#import theano
#from theano.sandbox.cuda import cuda_ndarray
#cuda_ndarray = cuda_ndarray.cuda_ndarray


def orderedcallargs(fn, *args, **kwargs):
    """
    Returns an OrderedDictionary containing the names and values of a
    function's arguments. The arguments are ordered according to the function's
    argspec:
        1. named arguments
        2. variable positional argument
        3. variable keyword argument
    """
    callargs = getcallargs(fn, *args, **kwargs)
    argspec = inspect.getargspec(fn)

    o_callargs = OrderedDict()
    for argname in argspec.args:
        o_callargs[argname] = callargs[argname]

    if argspec.varargs:
        o_callargs[argspec.varargs] = callargs[argspec.varargs]

    if argspec.keywords:
        o_callargs[argspec.keywords] = callargs[argspec.keywords]

    return o_callargs


def expandedcallargs(fn, *args, **kwargs):
    """
    Returns a tuple of all function args and kwargs, expanded so that varargs
    and kwargs are not nested. The args are ordered by their position in the
    function signature.
    """
    return tuple(flatten(orderedcallargs(fn, *args, **kwargs)))


def as_seq(x, seq_type=None):
    """
    If x is not a sequence, returns it as one. The seq_type argument allows the
    output type to be specified (defaults to list). If x is a sequence and
    seq_type is provided, then x is converted to seq_type.

    Arguments
    ---------
    x : seq or object

    seq_type : output sequence type
        If None, then if x is already a sequence, no change is made. If x
        is not a sequence, a list is returned.
    """
    if x is None:
        # None represents an empty sequence
        x = []
    elif not isinstance(x, (list, tuple, set, frozenset, dict)):
        # if x is not already a sequence (including dict), then make it one
        x = [x]

    if seq_type is not None and not isinstance(x, seq_type):
        # if necessary, convert x to the sequence type
        x = seq_type(x)

    return x


def itercode(code):
    """Return a generator of byte-offset, opcode, and argument
    from a byte-code-string
    """
    i = 0
    extended_arg = 0
    n = len(code)
    while i < n:
        c = code[i]
        num = i
        op = ord(c)
        i = i + 1
        oparg = None
        if op >= opcode.HAVE_ARGUMENT:
            oparg = ord(code[i]) + ord(code[i + 1]) * 256 + extended_arg
            extended_arg = 0
            i = i + 2
            if op == opcode.EXTENDED_ARG:
                extended_arg = oparg * 65536

        delta = yield num, op, oparg
        if delta is not None:
            abs_rel, dst = delta
            assert abs_rel == 'abs' or abs_rel == 'rel'
            i = dst if abs_rel == 'abs' else i + dst


def flatten(container):
    """Iterate over the elements of a [nested] container in a consistent order,
    unpacking dictionaries, lists, and tuples.

    Returns a list.

    Note that unflatten(container, flatten(container)) == container    """
    rval = []
    if isinstance(container, (list, tuple)):
        for d_i in container:
            rval.extend(flatten(d_i))
    elif isinstance(container, dict):
        if isinstance(container, OrderedDict):
            sortedkeys = container.keys()
        else:
            try:
                sortedkeys = sorted(container.keys())
            except TypeError:
                sortedkeys = container.keys()

        for k in sortedkeys:
            # if isinstance(k, (tuple, dict)):
                # # -- if keys are tuples containing ndarrays, should
                # #    they be traversed also?
                # raise NotImplementedError(
                    # 'potential ambiguity in container key', k)
            rval.extend(flatten(container[k]))
    else:
        rval.append(container)
    return rval


def unflatten(container, flat):
    """Iterate over a [nested] container, building a clone from the elements of
    flat.

    Returns object with same type as container.

    Note that unflatten(container, flatten(container)) == container
    """
    def unflatten_inner(container, pos):
        if isinstance(container, (list, tuple)):
            rval = []
            for d_i in container:
                d_i_clone, pos = unflatten_inner(d_i, pos)
                rval.append(d_i_clone)
            # check for namedtuple, which has a different __new__ signature
            if hasattr(container, '_fields'):
                rval = type(container)(*rval)
            else:
                rval = type(container)(rval)

        elif isinstance(container, dict):
            rval = type(container)()
            if isinstance(container, OrderedDict):
                sortedkeys = container.keys()
            else:
                try:
                    sortedkeys = sorted(container.keys())
                except TypeError:
                    sortedkeys = container.keys()
            for k in sortedkeys:
                v_clone, pos = unflatten_inner(container[k], pos)
                rval[k] = v_clone

        else:
            rval = flat[pos]
            pos += 1
        return rval, pos
    return unflatten_inner(container, 0)[0]


def isvar(x):
    """
    Type test for Theano variables.
    """
    vartypes = (theano.tensor.sharedvar.SharedVariable,
                theano.tensor.TensorConstant,
                theano.tensor.TensorVariable)
    return isinstance(x, vartypes)


def clean_int_args(*args, **kwargs):
    """
    Given args and kwargs, replaces small integers with numpy int16 objects, to
    allow tracing.
    """
    flatargs = flatten(args)
    for i, a in enumerate(flatargs):
        if type(a) is int and -5 <= a <= 256:
            flatargs[i] = np.int16(a)
    clean_args = unflatten(args, flatargs)

    flatkwargs = flatten(kwargs)
    for i, a in enumerate(flatkwargs):
        if type(a) is int and -5 <= a <= 256:
            flatkwargs[i] = np.int16(a)
    clean_kwargs = unflatten(kwargs, flatkwargs)
    return clean_args, clean_kwargs


# -- picklable decorated function
class post_collect(object):
    def __init__(self, f):
        self.f = f

    def __call__(self, *args, **kwargs):
        try:
            return self.f(*args, **kwargs)
        finally:
            gc.collect()
            #mem_info = cuda_ndarray.mem_info()
            #om = cuda_ndarray.outstanding_mallocs()
            #print 'Post-gc: %s %s' % (mem_info, om)
