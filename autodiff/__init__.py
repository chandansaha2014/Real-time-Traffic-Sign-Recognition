import logging

logger = logging.getLogger('autodiff')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

import autodiff.utils
import autodiff.optimize

from autodiff.symbolic import Symbolic, Tracer, Function, Gradient, HessianVector
from autodiff.decorators import (function, gradient, hessian_vector,
    as_symbolic, theanify)
from autodiff.functions import escape, tag, escaped_call, shadow
from autodiff.context import get_ast, print_ast, print_source

