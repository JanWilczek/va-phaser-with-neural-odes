from functools import partial
from torchdiffeq import odeint, odeint_adjoint
from solvers import ForwardEuler, trapezoid_rule


def get_method(args):
    if args.method.startswith('odeint') or args.method.startswith('ScaledODENet'):
        odeint_method = odeint_adjoint if args.adjoint else odeint
        method_name = args.method[(args.method.index('_') + 1):]
        return partial(odeint_method, method=method_name)

    method_dict = {"forward_euler": ForwardEuler(),
                   "trapezoid_rule": trapezoid_rule}
    return method_dict[args.method]
