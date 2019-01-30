# copied from PonyGE2

import numpy as np

from numpy import add, subtract, multiply, divide, sin, cos, exp, log, power, square, mod, sqrt, floor_divide, reciprocal
from numpy import bitwise_and, bitwise_or, bitwise_xor, invert, left_shift, right_shift
from numpy import logical_and, logical_or, logical_not, logical_xor




def aq(a, b):
    """aq is the analytic quotient, intended as a "better protected
    division", from: Ji Ni and Russ H. Drieberg and Peter I. Rockett,
    "The Use of an Analytic Quotient Operator in Genetic Programming",
    IEEE Transactions on Evolutionary Computation.

    :param a: np.array numerator
    :param b: np.array denominator
    :return: np.array analytic quotient, analogous to a / b.

    """
    try:
        return a / np.sqrt(1.0 + b**2.0)
    except FloatingPointError:
        # this still occurs! "FloatingPointError: underflow encountered in square"
        # that means that b is so close to zero that squaring it makes it zero
        # so, let's treat it as zero.
        return a # = a / np.sqrt(1.0 + 0.0**2.0)


def pdiv(x, y):
    """
    Koza's protected division is:

    if y == 0:
      return 1
    else:
      return x / y

    but we want an eval-able expression. The following is eval-able:

    return 1 if y == 0 else x / y

    but if x and y are Numpy arrays, this creates a new Boolean
    array with value (y == 0). if doesn't work on a Boolean array.

    The equivalent for Numpy is a where statement, as below. However
    this always evaluates x / y before running np.where, so that
    will raise a 'divide' error (in Numpy's terminology), which we
    ignore using a context manager.

    In some instances, Numpy can raise a FloatingPointError. These are
    ignored with 'invalid = ignore'.

    :param x: numerator np.array
    :param y: denominator np.array
    :return: np.array of x / y, or 1 where y is 0.
    """
    try:
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(y == 0, np.ones_like(x), x / y)
    except ZeroDivisionError:
        # In this case we are trying to divide two constants, one of which is 0
        # Return a constant.
        return 1.0


def rlog(x):
    """
    Koza's protected log:
    if x == 0:
      return 1
    else:
      return log(abs(x))

    See pdiv above for explanation of this type of code.

    :param x: argument to log, np.array
    :return: np.array of log(x), or 1 where x is 0.
    """
    with np.errstate(divide='ignore'):
        return np.where(x == 0, np.ones_like(x), np.log(np.abs(x)))


def ppow(x, y):
    """pow(x, y) is undefined in the case where x negative and y
    non-integer. This takes abs(x) to avoid it.

    :param x: np.array, base
    :param y: np.array, exponent
    :return: np.array x**y, but protected

    """
    return np.abs(x)**y


def ppow2(x, y):
    """pow(x, y) is undefined in the case where x negative and y
    non-integer. This takes abs(x) to avoid it. But it preserves
    sign using sign(x).

    :param x: np.array, base
    :param y: np.array, exponent
    :return: np.array, x**y, but protected
    """
    return np.sign(x) * (np.abs(x) ** y)


def psqrt(x):
    """
    Protected square root operator

    :param x: np.array, argument to sqrt
    :return: np.array, sqrt(x) but protected.
    """
    return np.sqrt(np.abs(x))


def psqrt2(x):
    """
    Protected square root operator that preserves the sign of the original
    argument.

    :param x: np.array, argument to sqrt
    :return: np.array, sqrt(x) but protected, preserving sign.
    """
    return np.sign(x) * (np.sqrt(np.abs(x)))


def plog(x):
    """
    Protected log operator. Protects against the log of 0.

    :param x: np.array, argument to log
    :return: np.array of log(x), but protected
    """
    return np.log(1.0 + np.abs(x))



def pmul(x, y):
    try:
        return x * y
    except FloatingPointError:
        # FIXME bad logic here this will return 0 for all fit-cases
        # not just the error one
        return 0.0 # I think it will occur with an underflow, so effectively 0
    
operators = {
    '+':  (add, 2),
    '-':  (subtract, 2),
    '*':  (pmul, 2),
    # '/':  (divide, 2),
    'aq':  (aq, 2),
    # '%':  (mod, 2),
    # '**': (power, 2),
    # '//': (floor_divide, 2),
    # 'exp': (exp, 1),
    'plog': (plog, 1),
    # 'sqrt': (sqrt, 1),
    # 'sq': (square, 1),
    # 'recip': (reciprocal, 1),
    'sin': (sin, 1),
    # 'cos': (cos, 1),
    # "&": (bitwise_and, 2),
    # "|": (bitwise_or, 2),
    # "^": (bitwise_xor, 2),
    # "~": (invert, 1),
    # "<<": (left_shift, 2),
    # ">>": (right_shift, 2),
    # "and": (logical_and, 2),
    # "or": (logical_or, 2),
    # "not": (logical_not, 1),
    # "xor": (logical_xor, 2),
}
operators_keys = list(operators.keys())
