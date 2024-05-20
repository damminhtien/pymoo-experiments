import gradient.toolbox as anp
import numpy as np

from constraints.as_penalty import ConstraintsAsPenalty
from core.individual import calc_cv
from core.meta import Meta
from core.problem import Problem
from util.misc import from_dict


class CDFAsObjective(Meta, Problem):

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

        self.n_ieq_constr = 0
        self.n_eq_constr = 0

    def do(self, X, return_values_of, *args, **kwargs):
        out = self.__object__.do(X, return_values_of, *args, **kwargs)

        # get at the values from the output
        F, G, H = from_dict(out, "F", "G", "H")

        # store a backup of the values in out
        out["__F__"], out["__G__"], out["__H__"] = F, G, H

        # G is matrix (m,n), H is matrix (m,k)
        m, n, k = G.shape[0], G.shape[1], H.shape[1]

        # Define the total number of constraints
        n_constraints = n + k

        # Initialize weights for the constraints (assuming it should be of length n_constraints)
        weighted_constraints = np.ones(n_constraints)

        # Initialize a count for violation frequencies
        constraint_violation_frequencies = np.zeros(n_constraints, dtype=int)

        # Count violations in G if n > 0
        if n > 0:
            constraint_violation_frequencies[:n] = np.sum(G > 0, axis=0)

        # Count violations in H if k > 0
        if k > 0:
            constraint_violation_frequencies[n:] = np.sum(H > 0, axis=0)

        # Compute masks for constraint violations in G and H, handling empty cases
        constraint_violation_mask_G = (G > 0).astype(
            int) if n > 0 else np.zeros((m, 0), dtype=int)
        constraint_violation_mask_H = (H > 0).astype(
            int) if k > 0 else np.zeros((m, 0), dtype=int)

        # Handle the combination and weighted calculation of violation masks based on G and H presence
        if n > 0 and k > 0:
            constraint_violation_mask = np.hstack(
                [constraint_violation_mask_G, constraint_violation_mask_H])
        elif n > 0:
            constraint_violation_mask = constraint_violation_mask_G
        elif k > 0:
            constraint_violation_mask = constraint_violation_mask_H
        else:
            constraint_violation_mask = np.zeros(
                (m, 0), dtype=int)  # No constraints to process

        # Calculation for wcvf adjusted for actual use of weights and frequencies
        wcvf = constraint_violation_mask * \
            weighted_constraints if n_constraints > 0 else np.zeros((m, 0))
        if n_constraints > 0:
            wcvf *= (1 - constraint_violation_frequencies /
                     n_constraints / m).reshape(1, -1)

        # Summing up the weighted constraint violation frequencies per sample
        sum_wcvf = np.sum(wcvf, axis=1).reshape(-1,
                                                1) if n_constraints > 0 else np.zeros((m, 1))

        # Print sum_wcvf to verify (Comment out in final version)
        # print(sum_wcvf)

        # CV = calc_cv(G=G, H=H, config=self.config)

        # do something here
        # hybird method
        # separate the infeasible vs the feasible
        # pF = F + self.penalty * CV[:, np.newaxis]
        # if CV.ndim > 1:
        #     CV = np.amax(CV, axis=1)
        # CV_mean = np.array([CV_i.max() for CV_i in CV])
        # CV_median = np.array([CV_i.max() for CV_i in CV])
        # try:
        #     pF = F + self.penalty * np.reshape(CV, F.shape)
        # except Exception:
        #     pF = F + self.penalty * CV

        # append the constraint violation as objective
        if self.append:
            out["F"] = anp.column_stack(
                [F, sum_wcvf])
        else:
            out["F"] = sum_wcvf

        del out["G"]
        del out["H"]

        return out

    def pareto_front(self, *args, **kwargs):
        pf = super().pareto_front(*args, **kwargs)
        if pf is not None:
            pf = np.column_stack([np.zeros(len(pf)), pf])
        return pf


class CVRAsObjective(Meta, Problem):

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

        self.n_ieq_constr = 0
        self.n_eq_constr = 0

    def do(self, X, return_values_of, *args, **kwargs):
        out = self.__object__.do(X, return_values_of, *args, **kwargs)

        # get at the values from the output
        F, G, H = from_dict(out, "F", "G", "H")

        # store a backup of the values in out
        out["__F__"], out["__G__"], out["__H__"] = F, G, H

        # calculate the total constraint violation (here normalization shall be already included)

        # Assuming G and H are numpy arrays where each column represents a constraint.
        # n_constraints = G.shape[1] + H.shape[1]
        # weighted_constraints = np.ones(n_constraints)
        # constraint_violation_frequencies = np.zeros(n_constraints, dtype=int)

        # Count violations in G (where constraints are greater than zero)
        # constraint_violation_frequencies[:G.shape[1]] = np.sum(G > 0, axis=0)

        # Count violations in H, offset by the number of constraints in G
        # constraint_violation_frequencies[G.shape[1]:] = np.sum(H > 0, axis=0)

        # print(constraint_violation_frequencies)
        if G.shape[1] == 0 and H.shape[1] == 0:
            contraint_violation_ratio = np.zeros((100, 1))
        else:
            # Calculate the positive counts for each matrix only if they are not empty
            G_sum = np.sum(G > 0, axis=1).reshape(-1,
                                                  1) if G.shape[1] > 0 else 0
            H_sum = np.sum(H > 0, axis=1).reshape(-1,
                                                  1) if H.shape[1] > 0 else 0
            # Sum the positive counts, utilizing broadcasting if one is zero
            contraint_violation_ratio = G_sum + H_sum
        # constraint_violation_mask = (G > 0).astype(int)
        # print(contraint_violation_ratio)

        # wcvf = constraint_violation_mask * weighted_constraints * \
        #     (1-constraint_violation_frequencies /
        #      n_constraints/G.shape[0])

        # sum_wcvf = np.sum(wcvf, axis=1).reshape(-1, 1)
        # print(wcvf)
        # print(np.sum(wcvf, axis=1).reshape(-1, 1))

        # CV = calc_cv(G=G, H=H, config=self.config)

        # do something here
        # hybird method
        # separate the infeasible vs the feasible
        # pF = F + self.penalty * CV[:, np.newaxis]
        # if CV.ndim > 1:
        #     CV = np.amax(CV, axis=1)
        # CV_mean = np.array([CV_i.max() for CV_i in CV])
        # CV_median = np.array([CV_i.max() for CV_i in CV])
        # try:
        #     pF = F + self.penalty * np.reshape(CV, F.shape)
        # except Exception:
        #     pF = F + self.penalty * CV

        # append the constraint violation as objective
        if self.append:
            out["F"] = anp.column_stack(
                [F, contraint_violation_ratio])
        else:
            out["F"] = contraint_violation_ratio

        del out["G"]
        del out["H"]

        return out

    def pareto_front(self, *args, **kwargs):
        pf = super().pareto_front(*args, **kwargs)
        if pf is not None:
            pf = np.column_stack([np.zeros(len(pf)), pf])
        return pf


class CDFAsObjective2(Meta, Problem):

    def __init__(self,
                 problem,
                 config=None,
                 append=True):

        super().__init__(problem)
        self.config = config
        self.append = append

        if append:
            self.n_obj = problem.n_obj + 2
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

        # G is matrix (m,n), H is matrix (m,k)
        m, n, k = G.shape[0], G.shape[1], H.shape[1]

        # Define the total number of constraints
        n_constraints = n + k

        # Initialize weights for the constraints (assuming it should be of length n_constraints)
        weighted_constraints = np.ones(n_constraints)

        # Initialize a count for violation frequencies
        constraint_violation_frequencies = np.zeros(n_constraints, dtype=int)

        # Count violations in G if n > 0
        if n > 0:
            constraint_violation_frequencies[:n] = np.sum(G > 0, axis=0)

        # Count violations in H if k > 0
        if k > 0:
            constraint_violation_frequencies[n:] = np.sum(H > 0, axis=0)

        # Compute masks for constraint violations in G and H, handling empty cases
        constraint_violation_mask_G = (G > 0).astype(
            int) if n > 0 else np.zeros((m, 0), dtype=int)
        constraint_violation_mask_H = (H > 0).astype(
            int) if k > 0 else np.zeros((m, 0), dtype=int)

        # Handle the combination and weighted calculation of violation masks based on G and H presence
        if n > 0 and k > 0:
            constraint_violation_mask = np.hstack(
                [constraint_violation_mask_G, constraint_violation_mask_H])
        elif n > 0:
            constraint_violation_mask = constraint_violation_mask_G
        elif k > 0:
            constraint_violation_mask = constraint_violation_mask_H
        else:
            constraint_violation_mask = np.zeros(
                (m, 0), dtype=int)  # No constraints to process

        # Calculation for wcvf adjusted for actual use of weights and frequencies
        wcvf = constraint_violation_mask * \
            weighted_constraints if n_constraints > 0 else np.zeros((m, 0))
        sum_f = sum(constraint_violation_frequencies)
        if n_constraints > 0 and sum_f > 0:
            wcvf *= (1 - 0.05*constraint_violation_frequencies /
                     sum_f).reshape(1, -1)

        # Summing up the weighted constraint violation frequencies per sample
        sum_wcvf = np.sum(wcvf, axis=1).reshape(-1,
                                                1) if n_constraints > 0 else np.zeros((m, 1))

        CV = calc_cv(G=G, H=H, config=self.config)

        # append the constraint violation as objective
        if self.append:
            out["F"] = anp.column_stack(
                [F, sum_wcvf, CV])
        else:
            out["F"] = anp.column_stack(
                [sum_wcvf, CV])

        del out["G"]
        del out["H"]

        return out

    def pareto_front(self, *args, **kwargs):
        pf = super().pareto_front(*args, **kwargs)
        if pf is not None:
            pf = np.column_stack([np.zeros(len(pf)), pf])
        return pf


class CVRAsObjective2(Meta, Problem):
    def __init__(self,
                 problem,
                 config=None,
                 append=True):

        super().__init__(problem)
        self.config = config
        self.append = append

        if append:
            self.n_obj = problem.n_obj + 2
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

        # print(constraint_violation_frequencies)
        if G.shape[1] == 0 and H.shape[1] == 0:
            contraint_violation_ratio = np.zeros((100, 1))
        else:
            # Calculate the positive counts for each matrix only if they are not empty
            G_sum = np.sum(G > 0, axis=1).reshape(-1,
                                                  1) if G.shape[1] > 0 else 0
            H_sum = np.sum(H > 0, axis=1).reshape(-1,
                                                  1) if H.shape[1] > 0 else 0
            # Sum the positive counts, utilizing broadcasting if one is zero
            contraint_violation_ratio = G_sum + H_sum

        CV = calc_cv(G=G, H=H, config=self.config)

        # append the constraint violation as objective
        if self.append:
            out["F"] = anp.column_stack(
                [F, contraint_violation_ratio, CV])
        else:
            out["F"] = [contraint_violation_ratio, CV]

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
            out["F"] = anp.column_stack([F, CV])
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
