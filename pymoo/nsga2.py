from core.problem import ElementwiseProblem
from algorithms.moo.nsga2 import NSGA2
from constraints.as_obj import ConstraintsAsObjective
from optimize import minimize
from visualization.scatter import Scatter


class ConstrainedProblem(ElementwiseProblem):

    def __init__(self, **kwargs):
        super().__init__(n_var=2, n_obj=1, n_ieq_constr=1,
                         n_eq_constr=0, xl=0, xu=2, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = x[0] ** 2 + x[1] ** 2
        out["G"] = 1.0 - (x[0] + x[1])


problem = ConstrainedProblem()

algorithm = NSGA2(pop_size=100)

res = minimize(ConstraintsAsObjective(problem),
               algorithm,
               ('n_gen', 300),
               seed=1,
               verbose=False)
