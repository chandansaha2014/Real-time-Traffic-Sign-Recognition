def tag(obj, tag):
    """
    NOTE: this function simply returns obj. When encountered by an Autodiff
    context object, it is transformed into a function that acts as described
    below.

    Tags an object with a certain keyword. By default, all symbolic objects are
    associated with the id of the Python object they shadow in a Context's
    svars dict.  By calling this function on an object and providing a
    (hashable) tag, users can more easily access the symbolic representation of
    any objects that might only be created during function execution.

    Example:

        @function
        def fn(x):
            y = tag(x + 2, 'y')
            z = y * 3
            return z

        fn.s_vars['y'] # returns the symbolic version of y

    """
    return obj

def escape(obj):
    """
    NOTE: this function simply returns obj. When encountered by an Autodiff
    context object, it is transformed into a function that acts as described
    below.

    Allows the use of an objects "raw" state, not one affected by any Autodiff
    or Theano transformations. This is frequently used by Autodiff for
    compatibility with functions that do not accept Tensor arguments, or to cut
    through some of the shadow mechanism employed by the backend.

    If escape is called on a container, it is applied to every item in the
    container, including any nested containers. For dicts, it applies to all
    values also in a recursive manner (but not keys).

    Example (read below for why escape isn't actually in this example...):

        @function
        def fn(x):
            for i in range(x):
                ...

    Autodiff might* convert x to a Tensor that can't be passed to range as an
    argument. Users could call `range(escape(x))` instead to access the value
    of x directly. Note that the range(x) construct is so common that Autodiff
    calls escape(x) automatically!
    """
    return obj

def escaped_call(fn, *args, **kwargs):
    """
    NOTE: In its raw form, this function simply calls fn on the supplied
    arguments. Only when encountered by an Autodiff context object does that
    change as described below.

    Allows users to call functions which can not be recompiled by Autodiff.

    When an Autodiff context encounters an `escaped_call` function, it does not
    attempt to modify the called function `fn`. Instead, it calls escape on all
    the arguments and passes them to the unmodified function. This should be
    used only when necessary, as tracing will not be possible on the result (it
    will be treated as a constant value).
    """
    return fn(*args, **kwargs)

def shadow(obj):
    """
    NOTE: In its raw form, this function simply calls fn on the supplied
    arguments. Only when encountered by an Autodiff context object does that
    change as described below.

    Allows users to force autodiff to shadow an array, for example one returned
    by escape().
    """
    return obj