import builtins
import logging
import copy
import meta
from ast import *
import types
import inspect
import numpy as np
import theano
import theano.tensor as T
import autodiff
import autodiff.utils as utils
import autodiff.functions
import collections


logger = logging.getLogger('autodiff')

# XXX FIXME This will not do - seed must be exposed.
# from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor.shared_randomstreams import RandomStreams
global_randomstreams = RandomStreams(seed=12345)
# seed = np.random.randint(1, 999999))


#########################
#########################
# from numba source

import linecache
import textwrap

try:
    from meta.decompiler import decompile_func
except Exception as exn:
    def decompile_func(*args, **kwargs):
        raise Exception("Could not import Meta -- Cannot recreate source "
                        "from bytecode")


def fix_ast_lineno(tree):
    # NOTE: A hack to fix assertion error in debug mode due to bad lineno.
    #       Lineno must increase monotonically for co_lnotab,
    #       the "line number table" to work correctly.
    #       This script just set all lineno to 1 and col_offset = to 0.
    #       This makes it impossible to do traceback, but it is not possible
    #       anyway since we are dynamically changing the source code.
    for node in ast.walk(tree):
        # only ast.expr and ast.stmt and their subclass has lineno and
        # col_offset.
        # if isinstance(node,  ast.expr) or isinstance(node, ast.stmt):
        node.lineno = 1
        node.col_offset = 0

    return tree


## Fixme:
##  This should be changed to visit the AST and fix-up where a None object
##  is present as this will likely not work for all AST.
def _fix_ast(myast):
    import _ast
    # Remove Pass nodes from the end of the ast
    while len(myast.body) > 0 and isinstance(myast.body[-1], _ast.Pass):
        del myast.body[-1]
    # Add a return node at the end of the ast if not present
    if len(myast.body) < 1 or not isinstance(myast.body[-1], _ast.Return):
        name = _ast.Name(id='None', ctx=_ast.Load(), lineno=0, col_offset=0)
        myast.body.append(Return(name))
    # remove _decorator list which sometimes confuses ast visitor
    try:
        indx = myast._fields.index('decorator_list')
    except ValueError:
        return
    else:
        myast.decorator_list = []


def get_ast(func):
    if func.__name__ == '<lambda>':
        func_def = decompile_func(func)
        if isinstance(func_def, Lambda):
            func_def = FunctionDef(name='<lambda>',
                                   args=func_def.args,
                                   body=[Return(func_def.body)],
                                   decorator_list=[])
        assert isinstance(func_def, FunctionDef)
        return func_def
    try:
        linecache.checkcache(inspect.getsourcefile(func))
        source = inspect.getsource(func)
        source_module = inspect.getmodule(func)
    except IOError:
        return decompile_func(func)
    else:
        # Split off decorators
        # TODO: This is not quite correct, we can have comments or strings
        # starting at column 0 and an indented function !
        source = textwrap.dedent(source)
        decorators = 0
        # decorator can have multiple lines
        while not source.lstrip().startswith('def'):
            assert source
            decorator, sep, source = source.partition('\n')
            decorators += 1
        source_file = getattr(source_module, '__file__', '<unknown file>')
        module_ast = compile(source, source_file, "exec", PyCF_ONLY_AST, True)

        lineoffset = func.__code__.co_firstlineno + decorators - 1
        increment_lineno(module_ast, lineoffset)

        assert len(module_ast.body) == 1
        func_def = module_ast.body[0]
        _fix_ast(func_def)
        assert isinstance(func_def, FunctionDef)
        # remove docstrings (really any unassigned strings)
        for node in func_def.body:
            if isinstance(node, Expr) and isinstance(node.value, Str):
                func_def.body.remove(node)
        return func_def

#########################
#########################


def get_source(ast):
    if hasattr(ast, '__code__'):
        ast = get_ast(ast)
    elif isinstance(ast, collections.Callable):
        ast = get_ast(ast.__call__)
    return meta.asttools.dump_python_source(ast)


def print_ast(ast):
    if hasattr(ast, '__code__'):
        ast = get_ast(ast)
    elif isinstance(ast, collections.Callable):
        ast = get_ast(ast.__call__)
    meta.asttools.print_ast(ast)


def print_source(ast):
    if hasattr(ast, '__code__'):
        ast = get_ast(ast)
    elif isinstance(ast, collections.Callable):
        ast = get_ast(ast.__call__)
    meta.asttools.python_source(ast)


def simple_Call(func, args=None):
    """
    Simple alias for building Call nodes that doesn't require specification of
    keywords, kwargs or starargs.
    """
    args = utils.as_seq(args)
    call = Call(args=args,
                func=func,
                keywords=[],
                kwargs=None,
                starargs=None)
    return call


def isvar_ast(name):
    """
    Wraps a Name node in a call to utils.isvar.
    """
    isvar = simple_Call(args=utils.as_seq(name),
                        func=Attribute(attr='isvar',
                                       ctx=Load(),
                                       value=Name(ctx=Load(), id='_utils__')))
    return isvar


class Context(object):

    def __init__(self,
                 borrowable=None,
                 force_floatX=False,
                 ignore=None,
                 infer_updates=False,
                 escape_on_error=False):
        self.sym_vars = dict()
        self.tags = dict()
        # FIXME do we need to hold on to all of these itermediates?
        # ensure these id's do not get recycled by garbage collection
        self._nogc = []
        self._top_def = None
        self.infer_updates = infer_updates
        self.updates = collections.OrderedDict()
        self.borrowable = [id(b) for b in utils.as_seq(borrowable)]
        self.force_floatX = force_floatX
        self.ignore = utils.as_seq(ignore, tuple)
        self.escape_on_error = escape_on_error
        self.shadowed_containers = dict()

    def recompile(self, f, nested=False):
        """
        Accepts a function f that operates on numerical objects and
        returns a function that operates on Theano objects.

        nested : bool
            `recompile` resets the context and sets the 'top_node' of the
            function, which helps in tracing arguments. By passing nested=True,
            this reset can be bypassed. This is used, for example, when
            transforming nested functions. In this case, we want to use the
            same context but keep it when calling recompile.
        """
        transformer = TheanoTransformer(context=self)

        f_ast = get_ast(f)

        if not nested:
            self._top_def = f_ast
            self.tags.clear()

        transformed_ast = fix_missing_locations(transformer.visit(f_ast))

        f_globals = f.__globals__.copy()
        f_globals.update(dict(_ctx__=transformer,
                              _functions__=autodiff.functions,
                              _T__=theano.tensor,
                              _utils__=autodiff.utils))
        if f.__closure__:
            f_globals.update((v, transformer.shadow(c.cell_contents))
                             for v, c in
                             zip(f.__code__.co_freevars, f.__closure__))

        for name in f.__code__.co_names:
            if name in f_globals.keys():
                f_globals[name] = transformer.shadow(f_globals[name])

        try:
            new_f = meta.decompiler.compile_func(ast_node=transformed_ast,
                                                 filename='<Context-AST>',
                                                 globals=f_globals)
        except SyntaxError as err:
            if "'return' with argument inside generator" in err.message:
                if isinstance(transformed_ast.body[-1], Return):
                    transformed_ast.body.pop(-1)
                    new_f = meta.decompiler.compile_func(
                        ast_node=transformed_ast,
                        filename='<Context-AST>',
                        globals=f_globals)
            else:
                raise
        except:
            raise

        # add defaults, if necessary (meta erases them and won't recompile!)
        if f.__defaults__:
            new_f.__defaults__ = utils.clean_int_args(*f.__defaults__)[0]

        # recreate method, if necessary
        if isinstance(f, types.MethodType):
            new_f = types.MethodType(new_f, f.__self__)

        return new_f

    def get_symbolic(self, x):
        """
        Attempts to retrieve the symbolic version of x.

        if x is an numeric object (int, float, numpy array), it must have been
        traced by the context during recompiled function execution.

        if x is a string, it must have been tagged with
        autodiff.functions.tag().
        """
        if isinstance(x, str):
            if x in self.sym_vars:
                return self.sym_vars[x]
            elif x in self.tags:
                return self.tags[x]
            else:
                raise ValueError(
                    'Requested the symbolic variable of tag `{0}`'
                    ', but `{0}` was not tagged.'.format(x))
        elif utils.isvar(x):
            return x
        elif id(x) in self.sym_vars:
            return self.sym_vars[id(x)]
        elif isinstance(x, int) and not isinstance(x, bool) and -5 <= x <= 256:
            raise ValueError(
                'Small integers (-5 <= x <= 256) can not be shadowed due to '
                'CPython caching. Try casting the variable as a NumPy int '
                'type or array before tracing: {0}'.format(x))
        elif np.asarray(x).dtype == 'object':
            raise ValueError(
                'Requested the symbolic variable shadowing object {0}, but '
                'it was not traced because it is not compatible with any '
                'Theano type.'.format(x))
        else:
            raise ValueError(
                'Requested the symbolic variable shadowing object {0}, but '
                'it was not traced because it did not appear in the '
                'function.'.format(x))

    def reset(self):
        self.sym_vars.clear()
        self.tags.clear()
        self._nogc = []
        self._top_node = None
        self.shadowed_containers.clear()


class TheanoTransformer(NodeTransformer):

    def __init__(self, context):
        super(TheanoTransformer, self).__init__()
        self.context = context

    def ast_wrap(self, method_name, args):
        """
        Allows Python methods to be applied to AST nodes at runtime.

        `method_name` is a method of the TheanoTransformer class that accepts
        Python objects as arguments.

        `args` are the AST nodes representing the arguments for `method_name`
        (not including `self`!).

        ast_wrap returns an `ast.Call()` node which calls the method on the
        specified arguments at runtime.
        """
        wrapped = simple_Call(func=Attribute(attr=method_name,
                                             ctx=Load(),
                                             value=Name(ctx=Load(),
                                                        id='_ctx__')),
                              args=args)

        return wrapped

    # ** --------------------------------------------------------
    # ** Direct Manipulation (Methods)

    def shadow(self, args):
        """
        Helper function for `_shadow` that calls it on a flattened version of
        its argument.
        """
        shadow_vars = [self._shadow_inner(x) for x in utils.flatten(args)]
        new_args = utils.unflatten(args, shadow_vars)
        if isinstance(new_args, (list, dict, tuple, set)):
            self.context.shadowed_containers[id(new_args)] = args
            # add to _nogc to ensure that the id won't be reused
            self.context._nogc.append(new_args)
        return new_args

    def _shadow_inner(self, x):

        """
        Given a numerical variable x, return an equivalent Theano shared
        variable and store the relationship in self.sym_vars. Otherwise return
        x.
        """
        # try checking if x is ignored (will fail for NumPy arrays)
        try:
            if x in self.context.ignore:
                return x
        except:
            pass

        # skip Python builtins and ignored id's
        if (id(x) in self.context.ignore
                or x is None
                or isinstance(x, (str, bool))):
            return x

        # skip ignored types
        elif isinstance(x,
                tuple(i for i in self.context.ignore if isinstance(i, type))):
            return x

        # transform compatible numeric values into Theano variables
        elif isinstance(x, (int, float, np.number, np.ndarray)):
            # take special care with small ints, because CPython caches them.
            if isinstance(x, int) and -5 <= x <= 256:
                x = np.int_(x)

            if getattr(x, 'dtype', None) == bool:
                logger.info('Note: Theano has no bool type; '
                            'upcasting bool to int8.')
                x = x.astype('int8')

            if id(x) not in self.context.sym_vars:

                # store id because x will be changed if force_floatX is True
                id_x = id(x)

                # add to _nogc to ensure that the id won't be reused
                self.context._nogc.append(x)

                # check if symbolic variable should be copied or borrowed
                borrow = id_x in self.context.borrowable

                # cast x if requested
                if self.context.force_floatX:
                    x = np.array(x, dtype=theano.config.floatX)

                # create symbolic version
                try:
                    sym_x = theano.shared(x, borrow=borrow)
                except:
                    sym_x = theano.shared(x)

                # store symbolic version
                self.context.sym_vars[id_x] = sym_x

                # return symbolic version
                return sym_x
            else:
                return self.context.sym_vars[id(x)]

        else:
            return x


    # ==================================================
    # ==================================================
    #
    # Runtime Modifications
    #
    # ==================================================
    # ==================================================

    @staticmethod
    def handle_escape(x):
        """
        Handles escaping variables
        """
        def escape(x):
            if isinstance(x, theano.tensor.sharedvar.SharedVariable):
                return x.get_value()
            elif utils.isvar(x):
                try:
                    return x.eval()
                except Exception as e:
                    raise ValueError(
                        'Could not escape {}. \nThe following error was '
                        'raised when trying to call eval():\n{}'.format(x, e))
            else:
                return x
        return utils.unflatten(x, [escape(i) for i in utils.flatten(x)])

    def handle_int(self, x, escape=False):
        if escape:
            x = self.handle_escape(x)

        if utils.isvar(x) and x.ndim == 0 and 'float' in x.dtype:
            return x.astype('int64')
        elif np.asarray(x).ndim == 0 and np.asarray(x).dtype.kind == 'f':
            return int(x)
        else:
            return x

    def handle_assign_updates(self, args):
        target, value = args
        self.shadow(target)
        if id(target) in self.context.sym_vars and utils.isvar(value):
            target_var = self.context.sym_vars[id(target)]
            self.context.updates[target_var] = value
        elif (isinstance(target, T.sharedvar.SharedVariable)
                and target in self.context.sym_vars.values()
                and utils.isvar(value)):
            self.context.updates[target] = value
        return value

    def handle_escaped_call(self, fn, *args, **kwargs):
        esc_args = utils.unflatten(
            args, [TheanoTransformer.handle_escape(a) for a in utils.flatten(args)])
        esc_kwargs = utils.unflatten(
            kwargs, [TheanoTransformer.handle_escape(a) for a in utils.flatten(kwargs)])
        escaped_result = fn(*esc_args, **esc_kwargs)
        return self.shadow(escaped_result)

    def handle_subscript(self, x):
        """
        Theano doesn't have a bool type, but we can track certain variables
        that we know must be boolean and possibly use that information (for
        advanced indexing, for example).

        We also cast non-integer scalar indices to ints (they may be coerced
        to floats by the force_floatX option, for example).
        """
        if isinstance(x, (list, tuple)):
            # check for namedtuples, which need their __new__ args expanded
            if hasattr(x, '_fields'):
                return type(x)(*[self._handle_subscript_inner(xi) for xi in x])
            else:
                return type(x)(self._handle_subscript_inner(xi) for xi in x)
        else:
            return self._handle_subscript_inner(x)

    def _handle_subscript_inner(self, x):
        if utils.isvar(x):
            if x.ndim > 0 and x.dtype == 'int8':
                return x.nonzero()
            elif x.ndim == 0 and 'int' not in x.dtype:
                return x.astype('int64')
            else:
                return x
        else:
            return x

    def handle_tag(self, obj, tag):
        if not isinstance(tag, str):
            raise ValueError('Tag must be a string. Received: {0}'.format(tag))
        if tag in self.context.tags:
            logger.warning(
                '{0} was tagged as {1}, but the tag {1} was already '
                'assigned. Note that the new tag will overwrite '
                'the old one.'.format(obj, tag))
        else:
            self.context.tags[tag] = obj
            if utils.isvar(obj):
                obj.name = tag
        return obj

    def handle_tag_function_arg(self, obj, tag):
        """
        A version of tagging called only by visit_FunctionDef, which tags
        top-level function arguments and stores the tags in sym_vars. These
        tags can not be overwritten.
        """
        self.context.sym_vars[tag] = obj
        if utils.isvar(obj):
            obj.name = tag

    def handle_functions(self, func):
        """
        Given some function for, return another function.

        Generally used to exchange NumPy functions for Theano equivalents.
        """

        # ** ======================= first handle functions defined here!

        if getattr(func, '__module__', None) == __name__:
            return func

        if func in self.context.ignore:
            return func

        # ** ======================= special autodiff functions

        elif func is autodiff.functions.escape:
            # escapes a variable from Tensor representation
            return self.handle_escape

        elif func is autodiff.functions.escaped_call:
            # call a function on escaped arguments without transforming the AST
            return self.handle_escaped_call

        elif func is autodiff.functions.tag:
            # tag a variable
            return self.handle_tag

        elif func is autodiff.functions.shadow:
            return self.shadow

        # ** ======================= autodiff classes

        elif isinstance(func, autodiff.symbolic.Symbolic):
            return func.symfn

        # ** ======================= __theano_op__

        elif hasattr(func, '__theano_op__'):
            return func.__theano_op__

        # ** ======================= array methods (with tensor instances)

        elif utils.isvar(getattr(func, '__self__', None)):
            return self.handle_methods(func.__self__, func.__name__)

        # ** ======================= Theano function

        elif (getattr(func, '__module__', None) and
                getattr(func, '__module__', '').startswith('theano')):
            return func

        elif isinstance(func, T.elemwise.Elemwise):
            return func

        # ** ======================= type/casting functions and new builtins

        elif type(func) is type:

            # range
            if func is range:
                def range_(*args):
                    int_args = (self.handle_int(a, escape=True) for a in args)
                    return func(*int_args)
                return range_

            # zip
            elif func is zip:
                def zip_(*args):
                    if any(utils.isvar(a) for a in args):
                        raise TypeError(
                            'Called zip() on Tensor but Tensors '
                            'do not support iteration. Maybe try escaping '
                            'the tensor?')
                    else:
                        return zip(*args)
                return zip_

            # casts
            elif func in(bool, np.bool_, np.bool8):
                logger.info('Warning: Theano has no bool type; '
                            'upgrading to int8.')

                def bool_(x):
                    return T.neq(x, 0)
                return bool_

            elif func.__name__ in T.basic._cast_mapping.keys():
                def cast(x):
                    return T.cast(x, dtype=func.__name__)
                return cast

            elif func is float:
                def float_(x):
                    return T.cast(x, dtype=theano.config.floatX)
                return float_

            elif func is int:
                def int_(x):
                    return T.cast(x, dtype='int' + theano.config.floatX[-2:])
                return int_

            # enumerate
            elif func is enumerate:
                def enumerate_(iterable, start=0):
                    if utils.isvar(iterable):
                        raise TypeError(
                            'Called enumerate() on Tensor {0} but Tensors '
                            'do not support iteration. Maybe try escaping '
                            'the tensor?'.format(iterable))
                    else:
                        return enumerate(iterable, start=start)
                return enumerate_

            # any other builtin function (tuple, list, set, Exception)
            elif func in builtins.__dict__.values():
                return func

            else:
                def new_type(*args, **kwargs):
                    try:
                        return self.shadow(func(*args, **kwargs))
                    except:
                        raise ValueError('Unsupported type: {0}'.format(func))
                return new_type

        # ** ======================= numpy functions

        elif (inspect.getmodule(func) is np
              or (getattr(func, '__module__', None)
                  and getattr(func, '__module__').startswith('numpy'))
              or isinstance(func, np.ufunc)
              or func in (min, max)):

            # abs
            if func in (np.abs, np.absolute):
                return abs

            # ones/zeros
            # FIXME submitted a PR to Theano to make syntax more
            # like Numpy; this change shouldn't be needed afterward.
            elif func in (np.ones, np.zeros):
                def alloc(shp, dtype=None):
                    if (not isinstance(shp, (list, tuple))
                            and not utils.isvar(shp)):
                        shp = [shp]
                    return getattr(T, func.__name__)(shp, dtype)
                return alloc

            # handle asarray
            elif func is np.asarray:
                def _asarray(x):
                    if not utils.isvar(x):
                        return np.asarray(x)
                    else:
                        return x
                return _asarray

            # atleast_1d
            elif func is np.atleast_1d:
                def _atleast_1d(x):
                    if x.ndim == 0:
                        return x.dimshuffle('x')
                    else:
                        return x
                return _atleast_1d

            # atleast_2d
            elif func is np.atleast_2d:
                def _atleast_2d(x):
                    if x.ndim == 0:
                        return x.dimshuffle('x', 'x')
                    elif x.ndim == 1:
                        return x.dimshuffle('x', 0)
                    else:
                        return x
                return _atleast_2d

            # atleast_3d
            elif func is np.atleast_3d:
                def _atleast_3d(x):
                    if x.ndim == 0:
                        return x.dimshuffle('x', 'x', 'x')
                    elif x.ndim == 1:
                        return x.dimshuffle('x', 'x', 0)
                    elif x.ndim == 2:
                        return x.dimshuffle('x', 0, 1)
                    else:
                        return x
                return _atleast_3d

            # reshape
            elif func is np.reshape:
                def _reshape(*args, **kwargs):
                    callargs = inspect.getcallargs(T.reshape, *args, **kwargs)
                    x, newshape = callargs['x'], callargs['newshape']
                    if isinstance(newshape, (list, tuple)):
                        newshape = [self.handle_int(s) for s in newshape]
                    else:
                        newshape = self.handle_int(newshape)
                    return T.reshape(x, newshape)
                return _reshape

            # vstack
            elif func is np.vstack:
                def _vstack(tup):
                    return T.vertical_stack(*tup)
                return _vstack

            # hstack
            elif func is np.hstack:
                def _hstack(tup):
                    return T.horizontal_stack(*tup)
                return _hstack

            # transpose
            elif func is np.transpose:
                def _transpose(a, axes=None):
                    if axes is not None:
                        axes = [self.handle_int(a, escape=True) for a in axes]
                    return T.transpose(x=a, axes=axes)
                return _transpose

            # functions taking axis as an argument -- make sure to escape it
            elif func in (np.argmax,
                          np.argmin,
                          np.argsort,
                          np.concatenate,
                          np.max,
                          np.mean,
                          np.min,
                          np.prod,
                          np.std,
                          np.sum,
                          np.var):

                def reduce_(*args, **kwargs):

                    func_name = func.__name__
                    if func_name == 'amax':
                        func_name = 'max'
                    elif func_name == 'amin':
                        func_name = 'min'

                    theano_func = getattr(T, func_name)
                    if 'axis' in kwargs:
                        kwargs['axis'] = self.handle_int(
                            kwargs['axis'], escape=True)
                    elif len(args) >= 2:
                        args = list(args)
                        args[1] = self.handle_int(args[1], escape=True)

                    # sometimes Theano uses 'a', sometimes it uses 'x'
                    if func not in (np.concatenate,):
                        np_first_arg = inspect.getargspec(func).args[0]
                        t_first_arg = inspect.getargspec(theano_func).args[0]
                        if np_first_arg in kwargs:
                            if np_first_arg != t_first_arg:
                                kwargs[t_first_arg] = kwargs.pop(np_first_arg)

                    return theano_func(*args, **kwargs)
                return reduce_

            # get equivalent Theano function
            elif hasattr(T, func.__name__):
                return getattr(T, func.__name__)

            else:
                raise ValueError(
                    'Autodiff unsupported function: {0}'.format(func))

        # ** ======================= ignore the inspect module

        elif inspect.getmodule(func) is inspect:
            return func

        # ** ======================= built-ins

        elif '<built-in' in str(func):

            # def escaped_random(*args, **kwargs):
            #     return self.handle_escaped_call(func, *args, **kwargs)
            # return escaped_random



            def handle_size(size):
                if not utils.isvar(size):
                    if not isinstance(size, (list, tuple)):
                        size = [size]
                    size = [self.handle_int(s) for s in size]
                else:
                    if size.ndim == 0:
                        size = size.dimshuffle('x')
                    size = size.astype('int64')
                return size

            # uniform random numbers (np.random.uniform)
            if func is np.random.uniform:
                def rand_u(low=0.0, high=1.0, size=1):
                    size = handle_size(size)
                    return global_randomstreams.uniform(low=low,
                                                        high=high,
                                                        size=size)
                return rand_u

            # standard uniform random numbers (np.random.random, np.random.rand)
            elif func in (np.random.random, np.random.rand):
                def rand_u(size):
                    size = handle_size(size)
                    return global_randomstreams.uniform(size=size)
                return rand_u

            # normal random numbers (np.random.normal)
            elif func is np.random.normal:
                def rand_n(loc=0.0, scale=1.0, size=1):
                    size = handle_size(size)
                    return global_randomstreams.normal(avg=loc,
                                                       std=scale,
                                                       size=size)
                return rand_n

            # standard normal random numbers (np.random.randn)
            elif func is np.random.randn:
                def rand_n(*size):
                    size = [self.handle_int(s) for s in size]
                    return global_randomstreams.normal(size=size)
                return rand_n

            # binomial random numbers (np.random.binomial)
            elif func is np.random.binomial:
                def rand_b(n, p, size=1):
                    size = handle_size(size)
                    return global_randomstreams.binomial(n=n, p=p, size=size)
                return rand_b

            # isinstance
            elif func is isinstance:
                def isinstance_(obj, types):
                    # if self.context.force_floatX:
                    #     if int in utils.as_seq(types):
                    #         logger.debug(
                    #             'You are trying to check for ints but '
                    #             'force_floatX is True, so the check may fail. '
                    #             'Consider escaping the call.')
                    escaped_obj = self.handle_escape(obj)
                    if (isinstance(escaped_obj, (np.ndarray, np.number))
                            and obj.ndim == 0):
                        escaped_obj = np.asscalar(escaped_obj)
                    return isinstance(escaped_obj, self.handle_escape(types))
                return isinstance_

            # inplace list methods
            elif isinstance(
                getattr(func, '__self__', None), (list, dict, set, tuple)):
                def _inplace(*args):
                    # check if the container is shadowing a different one
                    if id(func.__self__) in self.context.shadowed_containers:
                        c = self.context.shadowed_containers[id(func.__self__)]
                        tmp = getattr(c, func.__name__)(*args)
                        if tmp is None:
                            return c
                        else:
                            return tmp
                    else:
                        return func(*args)
                return _inplace

            # anything else
            else:
                return func

        # ** ======================= A bound method not covered yet

        # elif isinstance(func, types.MethodType):
            # return func

        # ** ======================= Misc

        elif (('ipdb' in (getattr(func, '__module__', '') or [])
              or 'pdb' in (getattr(func, '__module__', '') or []))
              and func.__name__ == 'set_trace'):
            return func

        # ** ======================= Special handling for OrderedDict views

        elif func in (collections.abc.ValuesView,
                      collections.abc.KeysView,
                      collections.abc.ItemsView):
            return func

        # ** ======================= Anything else

        else:
            try:
                return self.context.recompile(func, nested=True)
            except Exception as err:
                if self.context.escape_on_error:
                    logger.warning(
                        'Error when recompiling {0}. Calling escaped version '
                        'because escape_on_error is True.'.format(func))
                    def escapedfunc(*args, **kwargs):
                        return self.handle_escaped_call(func, *args, **kwargs)
                    return escapedfunc
                else:
                    raise ValueError(
                        'Unsupported function: {}. The following error was '
                        'raised: {}'.format(func, err))

        # ** ======================= Catchall (shouldn't be called)

        raise ValueError(
            'handle_functions: No case matched function {0}. Something is '
            'wrong -- should not reach this point!'.format(func))

    def handle_methods(self, var, method_name):
        """
        This method is called whenever:
            1. An array method is requested that doesn't exist for Theano
               variables (like _.swapaxes()). `handle_methods` is used
               to supply a replacement method. Note that in this case,
               `handle_methods` is called directly.
            2. A method is requested that DOES exist for Theano variables. In
               this case, `handle_methods` is called by
               `handle_functions` prior to calling the method.
               `handle_methods` is used to supply a replacement function
               that properly handles the supplied arguments (since they are
               compliant with the Numpy signature, not the Theano one).
        """
        # if we're not dealing with a Theano variable, nothing to do here.
        if not utils.isvar(var):
            return getattr(var, method_name)

        # ** ======================= Reshape

        # Theano's reshape requires dim to be in a collection, unlike Numpy.
        if method_name == 'reshape':
            def reshape(*args, **kwargs):
                if 'shape' in kwargs:
                    args = [kwargs.pop('shape')] + list(args)


                if args:
                    if not isinstance(args[0], (list, tuple)):
                        args = [args]
                else:
                    args = ((),)

                # Theano doesn't handle (), as an arg, which NumPy interprets
                # as casting length-1 vectors to scalars
                if args == ((),):
                    if var.ndim > 1:
                        raise ValueError(
                            'Reshape with `()` as an arg can only be used '
                            'with vectors of length 1.')
                    return var[0]
                else:
                    if args:
                        args = [self.handle_int(a) for a in args[0]]
                        if len(args) > 1:
                            args = [args]
                    return var.reshape(*args, **kwargs)
            return reshape

        # ** ======================= repeat

        elif method_name == 'repeat':
            def repeat(repeats, axis=None):
                if isinstance(repeats, (list, tuple)):
                    repeats = [self.handle_int(r) for r in repeats]
                else:
                    repeats = self.handle_int(repeats)

                axis = self.handle_int(axis, escape=True)
                return var.repeat(repeats, axis)
            return repeat

        # ** ======================= swapaxes

        # Theano has no swapaxes method
        elif method_name == 'swapaxes':
            def swapaxes(*args, **kwargs):
                axis1, axis2 = (int(self.handle_escape(a)) for a in args)
                dims = list(range(var.ndim))
                dims[axis1], dims[axis2] = dims[axis2], dims[axis1]
                return var.dimshuffle(*dims)
            return swapaxes

        # ** ======================= astype

        # Theano doesn't process numpy dtype objects or 'bool'
        elif method_name == 'astype':
            def astype(*args, **kwargs):
                dtype = kwargs.pop('dtype', None)
                if not dtype:
                    dtype = args[0]
                if not isinstance(dtype, str):
                    # get numpy dtype objects like np.float32
                    try:
                        dtype = dtype.__name__
                    except:
                        raise NotImplementedError(
                            'Unsupported dtype: {0}'.format(dtype))
                if 'bool' in dtype:
                    dtype = 'int8'
                    logger.info('Warning: Theano has no bool type; '
                                'upgrading to int8.')
                return var.astype(dtype)
            return astype

        # ** ======================= sort

        elif method_name == 'sort':
            def sort_(*args, **kwargs):
                raise ValueError(
                    'Calling an array\'s `sort()` method is not supported '
                    'because in NumPy it is an inplace operation, but in '
                    'Theano it is not. Please use numpy.sort() instead.')
            return sort_

        # ** ======================= reductions

        elif method_name in ('argmax',
                             'argmin',
                             'argsort',
                             'concatenate',
                             'max',
                             'mean',
                             'min',
                             'norm',
                             'prod',
                             'std',
                             'sum',
                             'var'):
            def reduce_(*args, **kwargs):
                method = getattr(var, method_name)
                all_args = inspect.getcallargs(method, *args, **kwargs)
                for k, v in list(all_args.items()):
                    if v is method.__self__:
                        all_args.pop(k)
                all_args['axis'] = self.handle_escape(all_args['axis'])
                if all_args['axis'] is not None:
                    all_args['axis'] = int(all_args['axis'])
                return method(**all_args)
            return reduce_

        # ** ======================= anything else

        # ...Otherwise, try to access the method on the Theano variable
        else:
            return getattr(var, method_name)

    def handle_comparison(self, operator, left, right):
        """
        This method is called whenever an operator is encountered with a single
        rhs comparator, since tensors do not properly them.
        """
        if utils.isvar(left) or utils.isvar(right):
            return getattr(T, operator)(left, right)
        elif operator == 'gt':
            return left > right
        elif operator == 'ge':
            return left >= right
        elif operator == 'lt':
            return left < right
        elif operator == 'le':
            return left <= right
        elif operator == 'eq':
            return left == right
        elif operator == 'neq':
            return left != right
        else:
            # shouldn't ever reach here!
            raise ValueError(
                'Not sure how to handle operator: {0}'.format(operator))

    # ** --------------------------------------------------------
    # ** AST Manipulation (Node Visitors)

    def insert_breakpoint(self, _):
        import ipdb; ipdb.set_trace()

    def visit_Assign_with_updates(self, node):
        """
        Given an assignment, attempt to infer a symbolic update from the
        target and value.
        """

        load_targets = copy.deepcopy(node.targets)
        value = node.value

        for t in load_targets:
            load_transformer.generic_visit(t)

        node_with_updates = copy.deepcopy(node)

        node_with_updates.value = self.ast_wrap(
            'handle_assign_updates', List(
                ctx=Load(),
                elts=load_targets + [value]))
        body=[node_with_updates]

        # wrap this in a try because if this is the first time a variable
        # is being assigned, then load_targets will try to reference
        # a nonexistant variable!
        return Try(
            body=body,
            handlers=[ExceptHandler(body=[node])],
            finalbody=[],
            orelse=[])

    def visit_Assign(self, node):
        """
        Applies the following transformations:

        - Transform subscripts. Tensor variables do not support inplace
          assignment, so subscript assigns must be changed to call the
          `set_subtensor` function.

            Statements of the form:
                x[a:b][c] = y
            Become:
                if utils.isvar(x):
                    x = T.set_subtensor(x[a:b], T.set_subtensor(x[a:b][c], y))
                else:
                    x[a:b][c] = y

        """
        # TODO
        # AugAssigns with unbounded subscripts decompile strangely and can't
        # be recompiled. Specifically, they decompile as an Assign to a target
        # with a value that is an AugAssign of the same target and the true
        # value. To get around this, we just take the AugAssign (which appears
        # to be correct) and replace the Assign with it.
        # This is the syntax that creates the weird AST:
        #    a[:b] += c

        # if isinstance(node.value, AugAssign):
            # return self.visit_AugAssign(node.value)

        # handle subscripted assignment for tensor variables
        if isinstance(node.targets[0], Subscript):
            # helper function to transform subscript into (possibly nested)
            # T.set_subtensor statements
            def build_subt(subscript, value):
                subscript_load = Subscript(ctx=Load(),
                                           slice=subscript.slice,
                                           value=subscript.value)
                set_subtensor = simple_Call(
                    args=[subscript_load, value],
                    func=Attribute(attr='set_subtensor',
                                   ctx=Load(),
                                   value=Name(ctx=Load(), id='_T__')))
                if isinstance(subscript.value, Subscript):
                    set_subtensor = build_subt(subscript.value, set_subtensor)
                return set_subtensor

            # get root tensor; check for nested subscripts
            tensor = node.targets[0]
            while not isinstance(tensor, Name):
                try:
                    tensor = tensor.value
                except:
                    break

            if isinstance(tensor, Name):
                # transform subscript into set_subtensor
                if isinstance(node.value, AugAssign):
                    value = BinOp(op=node.value.op,
                                  left=node.targets[0],
                                  right=node.value.value)
                else:
                    value = node.value
                set_subt = build_subt(subscript=node.targets[0], value=value)

                # wrap set_subtensor statements in Assign to root tensor
                assign_subtensor = Assign(targets=[Name(ctx=Store(),
                                                        id=tensor.id)],
                                          value=set_subt)

                # wrap assign_subtensor in If to ensure that the modification
                # is only applied to tensor args
                self.generic_visit(node.value)
                if self.context.infer_updates:
                    node = self.visit_Assign_with_updates(node)
                return If(test=isvar_ast(tensor),
                          body=[assign_subtensor],
                          orelse=[node])
            else:
                self.generic_visit(node)
        else:
            self.generic_visit(node)

        if self.context.infer_updates:
            return self.visit_Assign_with_updates(node)
        else:
            return node

    # ==================================================
    # ==================================================
    #
    # AST Modifications
    #
    # ==================================================
    # ==================================================

    def visit_Attribute(self, node):
        """
        When dealing with an attribute, first see if the object has that
        attribute and return it. If not, call the handle_methods method.
        """
        self.generic_visit(node)
        if isinstance(node.ctx, Store):
            return node
        else:
            new_node = simple_Call(
                args=[node.value,
                      Str(s=node.attr),
                      self.ast_wrap('handle_methods',
                                    [node.value, Str(s=node.attr)])],
                func=Name(ctx=Load(), id='getattr'))
            return self.ast_wrap('shadow', new_node)

    def visit_AugAssign(self, node):
        """
        See documentation for self.visit_Assign() for information on
        transformations applied here.
        """
        #transform into assign
        load_target = load_transformer.generic_visit(copy.deepcopy(node.target))
        value = BinOp(op=node.op,
                      left=self.ast_wrap('shadow', load_target),
                      right=node.value)
        new_node = Assign(targets=[node.target],
                          value=value)
        return self.visit_Assign(new_node)

    def visit_Call(self, node):
        """
        Whenever a function is called, first pass it to the 'handle_functions'
        method. This method examines the function and modifies it prior to
        calling it. For example, it might replace `numpy.ones` with
        `theano.ones`.
        """
        self.generic_visit(node)
        node.func = self.ast_wrap('handle_functions', node.func)

        # the * and ** syntax won't work if an object has been shadowed...
        # if node.starargs:
            # node.starargs = self.ast_wrap('handle_shadow_class', node.starargs)
        # if node.kwargs:
            # node.kwargs = self.ast_wrap('handle_shadow_class', node.kwargs)

        return node

    def visit_ClassDef(self, node):
        return node

    def visit_Compare(self, node):
        """
        Replaces comparison operators with Theano functions, if either argument
        is a tensor variable. Prior to NumPy 1.8, this is required for all
        comparisons where the NumPy array is on the left; thereafter it is
        required only for == and !=.


        Given:

            x == y

        Becomes:

            _ctx__.handle_comparison('eq', x, y)

        Which internally performs:

            if utils.isvar(x) or utils.isvar(y):
                T.eq(x, y)
            else:
                x == y

        This could be done by directly replacing the literal comparison with
        the `if` clause, but this wouldn't be compatible with all code. For
        example, if the comparison takes place in an `if` clause, the new
        (and nested) `if` clause would be illegal syntax. Wrapping the `isvar`
        check in a function call means the syntax remains compatible.
        """
        self.generic_visit(node)

        if isinstance(node.ops[0], Eq):
            theano_op = Str(s='eq')
        elif isinstance(node.ops[0], NotEq):
            theano_op = Str(s='neq')
        elif isinstance(node.ops[0], Gt):
            theano_op = Str(s='gt')
        elif isinstance(node.ops[0], GtE):
            theano_op = Str(s='ge')
        elif isinstance(node.ops[0], Lt):
            theano_op = Str(s='lt')
        elif isinstance(node.ops[0], LtE):
            theano_op = Str(s='le')
        else:
            # Is, IsNot, In, NotIn
            return node

        if len(node.comparators) == 1:
            return self.ast_wrap('handle_comparison',
                                 [theano_op, node.left, node.comparators[0]])
        else:
            return node

    def visit_FunctionDef(self, node):
        """
        When a function is defined, shadow each of its arguments immediately.

        The AST is modified so that a function defined as:

            def f(a, b=None, *c, **d):
                ...

        is changed via this method to:

            def f(a, b=None, *c, **d):
                a = self.shadow(a)
                b = self.shadow(b)
                c = self.shadow(c)
                d = self.shadow(d)
                tag(a, 'a')
                tag(b, 'b')
                for k, v in d.items():
                    tag(v, k)
                ...

        This way, any future references to these variables will access their
        shadowed values. This is important because inplace modifications do
        not always force the `shadow` method to get called, and so the inplace
        changes might not be reflected the next (and first!) time the variable
        is loaded.
        """
        self.generic_visit(node)
        assigns = []
        tags = []

        # shadow and tag args
        for param in node.args.args:
            assigns.append(Assign(
                targets=[Name(ctx=Store(), id=param.arg)],
                value=self.ast_wrap('shadow', Name(ctx=Load(), id=param.arg))))

            tags.append(Expr(value=self.ast_wrap(
                method_name='handle_tag_function_arg',
                args=[Name(ctx=Load(), id=param.arg), Str(s=param.arg)])))

        # shadow the varargs
        if node.args.vararg:
            if isinstance(node.args.vararg, str):
                node.args.vararg = arg(annotation=None, arg=node.args.vararg)

            assigns.append(Assign(
                targets=[Name(ctx=Store(), id=node.args.vararg.arg)],
                value=self.ast_wrap('shadow', Name(ctx=Load(),
                                                   id=node.args.vararg.arg))))

        # shadow and tag the kwargs
        if node.args.kwarg:
            if isinstance(node.args.kwarg, str):
                node.args.kwarg = arg(annotation=None, arg=node.args.kwarg)

            assigns.append(Assign(
                targets=[Name(ctx=Store(), id=node.args.kwarg.arg)],
                value=self.ast_wrap('shadow', Name(ctx=Load(),
                                                   id=node.args.kwarg.arg))))

            tags.append(For(
                body=[Expr(value=self.ast_wrap(
                    method_name='handle_tag_function_arg',
                    args=[Name(ctx=Load(), id='v'),
                          Name(ctx=Load(), id='k')]))],
                iter=simple_Call(
                    func=Attribute(attr='items',
                                   ctx=Load(),
                                   value=Name(ctx=Load(),
                                              id=node.args.kwarg.arg))),
                orelse=[],
                target=Tuple(ctx=Store(), elts=[Name(ctx=Store(), id='k'),
                                                Name(ctx=Store(), id='v')])))

        if node is self.context._top_def:
            node.body = assigns + tags + node.body
            self.context._top_def = None
        else:
            node.body = assigns + node.body

        return node

    def visit_If(self, node):
        """
        Transform this:

            if <statement>:
                ...
            else:
                ...

        to this:

            if escape(<statement>):
                ...
            else:
                ...

        This means that the if statement's test clause will be evaluated at
        runtime. Note that this does NOT carry over to the compiled Theano
        code. It just protects against the following case:

            if x:
                <do something>

        If x is a shadowed variable, then it always resolves to True. However,
        x could have a value of 0, in which case this shouldn't pass. Escaping
        x resolves it when the function is called.
        """
        self.generic_visit(node)
        node.test = self.ast_wrap('handle_escape', node.test)
        return node

    def visit_Subscript(self, node):
        """
        Theano does not have a bool dtype, and therefore does not support
        Numpy's advanced indexing with boolean masks. For example, the
        following is interpreted as requested many items at the indices 1 and
        0, not as a boolean mask:

            x[x > 0.5]

        It is possible to replicate the boolean mask behavior in Theano with
        the following construction:

            x[(x > 0.5).nonzero()]

        tensor.nonzero() returns a tuple of indices corresponding to the
        nonzero elements. Thus, this properly selects the desired elements but
        is not compatible with Numpy comparisons anywhere else.

        To resolve this, if a Theano 'int8' subscript or index is requested,
        it is treated as a boolean mask and wrapped in a nonzero() call.

        NOTE THIS DOESN'T HANDLE ALL CASES
        """
        self.generic_visit(node)
        if isinstance(node.slice, Index):
            node.slice = Index(value=self.ast_wrap('handle_subscript',
                                                   node.slice.value))
        return node

    def visit_Name(self, node):
        """
        Whenever a literal variable name is loaded, call the
        'shadow' method on its value.
        """
        # self.generic_visit(node)
        if isinstance(node.ctx, Load):
            node = self.ast_wrap('shadow', node)
        return node


class LoadTransformer(NodeTransformer):
    def generic_visit(self, node):
        node = super(LoadTransformer, self).generic_visit(node)
        if hasattr(node, 'ctx'):
            if isinstance(node.ctx, Store):
                node.ctx = Load()

        return node

load_transformer = LoadTransformer()