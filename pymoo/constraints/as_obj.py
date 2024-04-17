import gradient.toolbox as anp
import numpy as np

from constraints.as_penalty import ConstraintsAsPenalty
from core.individual import calc_cv
from core.meta import Meta
from core.problem import Problem
from util.misc import from_dict


class ModifiedConstraintsAsObjective(Meta, Problem):

    def __init__(self,
                 problem,
                 config=None,
                 append=True,
                 penalty: float = 0.1):

        super().__init__(problem)
        self.config = config
        self.append = append

        # the amount of penalty to add for this type
        self.penalty = penalty

        if append:
            self.n_obj = problem.n_obj + 2  # add constraints and penaltized F as objectives
        else:
            self.n_obj = problem.n_obj

        self.n_ieq_constr = 0
        self.n_eq_constr = 0

    def do(self, X, return_values_of, *args, **kwargs):
        out = self.__object__.do(X, return_values_of, *args, **kwargs)

        # get at the values from the output
        F, G, H = from_dict(out, "F", "G", "H")

        # store a backup of the values in out
        out["__F__"], out["__G__"], out["__H__"] = F, G, H

        # calculate the total constraint violation (here normalization shall be already included)
        CV = calc_cv(G=G, H=H, config=self.config)

        # do something here
        # hybird method
        # separate the infeasible vs the feasible
        pF = F + self.penalty * CV[:, np.newaxis]
        # try:
        #     pF = F + self.penalty * np.reshape(CV, F.shape)
        # except Exception:
        #     pF = F + self.penalty * CV

        # append the constraint violation as objective
        if self.append:
            out["F"] = anp.column_stack([CV, pF, F])
        else:
            out["F"] = CV

        del out["G"]
        del out["H"]

        return out

    def pareto_front(self, *args, **kwargs):
        pf = super().pareto_front(*args, **kwargs)
        if pf is not None:
            pf = np.column_stack([np.zeros(len(pf)), pf])
        return pf


class ConstraintsAsObjective(Meta, Problem):

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
            self.n_obj = 1

        self.n_ieq_constr = 0
        self.n_eq_constr = 0

    def do(self, X, return_values_of, *args, **kwargs):
        out = self.__object__.do(X, return_values_of, *args, **kwargs)

        # get at the values from the output
        F, G, H = from_dict(out, "F", "G", "H")

        # store a backup of the values in out
        out["__F__"], out["__G__"], out["__H__"] = F, G, H

        # calculate the total constraint violation (here normalization shall be already included)
        CV = calc_cv(G=G, H=H, config=self.config)

        # append the constraint violation as objective
        if self.append:
            out["F"] = anp.column_stack([CV, F])
        else:
            out["F"] = CV

        del out["G"]
        del out["H"]

        return out

    def pareto_front(self, *args, **kwargs):
        pf = super().pareto_front(*args, **kwargs)
        if pf is not None:
            pf = np.column_stack([np.zeros(len(pf)), pf])
        return pf
