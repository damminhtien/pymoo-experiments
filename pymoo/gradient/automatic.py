import gradient.toolbox as anp
import numpy as np

from core.individual import calc_cv
from core.meta import Meta
from core.problem import Problem
from util.misc import from_dict

from pymoo.core.meta import Meta
from pymoo.core.problem import Problem, ElementwiseEvaluationFunction


class ElementwiseEvaluationFunctionWithGradient(ElementwiseEvaluationFunction):

    def __call__(self, x):
        f = super().__call__

        from pymoo.gradient import TOOLBOX

        if TOOLBOX == "jax.numpy":
            from pymoo.gradient.grad_jax import jax_elementwise_value_and_grad
            out, grad = jax_elementwise_value_and_grad(f, x)

        elif TOOLBOX == "autograd.numpy":
            from pymoo.gradient.grad_autograd import autograd_elementwise_value_and_grad
            out, grad = autograd_elementwise_value_and_grad(f, x)

        for k, v in grad.items():
            out["d" + k] = np.array(v)

        return out


class ElementwiseAutomaticDifferentiation(Meta, Problem):

    def __init__(self, problem, copy=True):
        if not problem.elementwise:
            raise Exception(
                "Elementwise automatic differentiation can only be applied to elementwise problems.")

        super().__init__(problem, copy)
        self.elementwise_func = ElementwiseEvaluationFunctionWithGradient


class AutomaticDifferentiation(Meta, Problem):

    def do(self, x, return_values_of, *args, **kwargs):
        from pymoo.gradient import TOOLBOX

        vals_not_grad = [v for v in return_values_of if not v.startswith("d")]
        def f(xp): return self.__object__.do(
            xp, vals_not_grad, *args, **kwargs)

        if TOOLBOX == "jax.numpy":
            from pymoo.gradient.grad_jax import jax_vectorized_value_and_grad
            out, grad = jax_vectorized_value_and_grad(f, x)

        elif TOOLBOX == "autograd.numpy":
            from pymoo.gradient.grad_autograd import autograd_vectorized_value_and_grad
            out, grad = autograd_vectorized_value_and_grad(f, x)

        for k, v in grad.items():
            out["d" + k] = v

        return out


class MyAutomaticDifferentiation(AutomaticDifferentiation):
    def __init__(self,
                 problem,
                 config=None,
                 append=True):

        super().__init__(problem)
        self.config = config
        self.append = append

        if append:
            self.n_obj = problem.n_obj + 1
        else:
            self.n_obj = problem.n_obj

    def do(self, X, return_values_of, *args, **kwargs):
        out = self.__object__.do(X, return_values_of, *args, **kwargs)

         # get at the values from the output
        F, G, H, dG, dH = from_dict(out, "F", "G", "H", "dG", "dH")

        # store a backup of the values in out
        out["__F__"], out["__G__"], out["__H__"], out["__dG__"], out["__dH__"] = F, G, H, dG, dH

        # calculate the total constraint violation (here normalization shall be already included)
        CV = calc_cv(G=G, H=H, config=self.config)

        # do something here
        # hybird method
        # separate the infeasible vs the feasible

        # append the constraint violation as objective
        if self.append:
            out["F"] = anp.column_stack([CV, F])
        else:
            out["F"] = CV

        # del out["G"]
        # del out["H"]
        
        return out

    def pareto_front(self, *args, **kwargs):
        pf = super().pareto_front(*args, **kwargs)
        if pf is not None:
            pf = np.column_stack([np.zeros(len(pf)), pf])
        return pf
