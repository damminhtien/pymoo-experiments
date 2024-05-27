import gradient.toolbox as anp
import numpy as np

from constraints.as_penalty import ConstraintsAsPenalty
from core.individual import calc_cv
from core.meta import Meta
from core.problem import Problem
from util.misc import from_dict


def calc_cvr(G, H, weighted_constraint_vector):
    # Compute masks for constraint violations in G and H
    constraint_violation_mask_G = (G > 0.00001).astype(int)
    constraint_violation_mask_H = (H > 0.00001).astype(int)

    # Combine masks
    constraint_violation_mask = np.hstack(
        [constraint_violation_mask_G, constraint_violation_mask_H])

    # Calculate the weighted constraint violation vector
    cv_vector = constraint_violation_mask * weighted_constraint_vector

    # Summing up the weighted constraint violation frequencies per sample
    cvr = np.sum(cv_vector, axis=1).reshape(-1, 1)

    # Calculate and print the feasible rate
    # infeasible_rate = np.sum(
    #     np.sum(constraint_violation_mask, axis=1) > 0.0001) / m
    # print(f'Infeasible rate: {infeasible_rate}')

    # Debug prints
    # print(np.sum(constraint_violation_mask, axis=0))
    # print(f'cvr mean: {np.mean(cvr)}')
    # print(cvr)
    return cvr


def calc_cdf(G, H, alpha=0.5):
    # G is matrix (m, n), H is matrix (m, k)
    m, n = G.shape[0], G.shape[1]
    k = H.shape[1]

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

    # Compute masks for constraint violations in G and H
    constraint_violation_mask_G = (G > 0).astype(int)
    constraint_violation_mask_H = (H > 0).astype(int)

    # Combine masks
    constraint_violation_mask = np.hstack(
        [constraint_violation_mask_G, constraint_violation_mask_H])

    # Calculation for WCVF adjusted for actual use of weights and frequencies
    cvr = constraint_violation_mask * weighted_constraints

    f_sum = np.sum(constraint_violation_frequencies)
    if f_sum == 0:
        epsilon = 1
    else:
        epsilon = (1 + alpha*constraint_violation_frequencies /
                   f_sum).reshape(1, -1)

    cdf = cvr*epsilon

    # Summing up the weighted constraint violation frequencies per sample
    cdf = np.sum(cdf, axis=1).reshape(-1, 1)

    # Calculate and print the feasible rate
    # infeasible_rate = np.sum(
    #     np.sum(constraint_violation_mask, axis=1) > 0.0001) / m
    # print(f'Infeasible rate: {infeasible_rate}')
    # print(epsilon)

    # Debug prints
    # print(np.sum(constraint_violation_mask, axis=0))
    # print(f'CDF mean: {np.mean(cdf)}')

    return cdf


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

        cdf = calc_cdf(G, H)

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
                [F, cdf])
        else:
            out["F"] = cdf

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
                 append=True,
                 weighted_constraint_vector=None):

        super().__init__(problem)
        self.config = config
        self.append = append

        if weighted_constraint_vector is None:
            # Initialize weights for the constraints (assuming it should be of length n_constraints)
            weighted_constraint_vector = np.ones(
                self.n_ieq_constr+self.n_eq_constr)
        elif isinstance(weighted_constraint_vector, int) or isinstance(weighted_constraint_vector, float):
            weighted_constraint_vector = np.random.uniform(
                low=1.0, high=weighted_constraint_vector, size=(self.n_ieq_constr+self.n_eq_constr))
        print(
            f'CVRAsObjective: weighted_constraint_vector={weighted_constraint_vector}')
        self.weighted_constraint_vector = weighted_constraint_vector

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

        cvr = calc_cvr(G, H, self.weighted_constraint_vector)

        # append the constraint violation as objective
        if self.append:
            out["F"] = anp.column_stack(
                [F, cvr])
        else:
            out["F"] = cvr

        del out["G"]
        del out["H"]

        return out

    def pareto_front(self, *args, **kwargs):
        pf = super().pareto_front(*args, **kwargs)
        if pf is not None:
            pf = np.column_stack([np.zeros(len(pf)), pf])
        return pf


class CDFAsObjective2(CDFAsObjective):
    def __init__(self,
                 problem,
                 config=None,
                 append=True,
                 alpha=1):

        super().__init__(problem)
        self.config = config
        self.append = append
        self.alpha = alpha

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

        cdf = calc_cdf(G, H)

        CV = calc_cv(G=G, H=H, config=self.config)

        # append the constraint violation as objective
        if self.append:
            out["F"] = anp.column_stack(
                [F, cdf, CV])
        else:
            out["F"] = anp.column_stack(
                [cdf, CV])

        del out["G"]
        del out["H"]

        return out

    def pareto_front(self, *args, **kwargs):
        pf = super().pareto_front(*args, **kwargs)
        if pf is not None:
            pf = np.column_stack([np.zeros(len(pf)), pf])
        return pf


class CDFAsObjective3(Meta, Problem):

    def __init__(self,
                 problem,
                 config=None,
                 append=True,
                 alpha=1):

        super().__init__(problem)
        self.config = config
        self.append = append
        self.alpha = alpha

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

        cdf = calc_cdf(G, H, alpha=self.alpha)

        CV = calc_cv(G=G, H=H, config=self.config)

        # append the constraint violation as objective
        if self.append:
            out["F"] = anp.column_stack(
                [F, cdf, CV])
        else:
            out["F"] = anp.column_stack(
                [cdf, CV])

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
                 append=True,
                 weighted_constraint_vector=None):

        super().__init__(problem)
        self.config = config
        self.append = append

        if weighted_constraint_vector is None:
            # Initialize weights for the constraints (assuming it should be of length n_constraints)
            weighted_constraint_vector = np.ones(
                self.n_ieq_constr+self.n_eq_constr)
        elif isinstance(weighted_constraint_vector, int) or isinstance(weighted_constraint_vector, float):
            weighted_constraint_vector = np.random.uniform(
                low=1.0, high=weighted_constraint_vector, size=(self.n_ieq_constr+self.n_eq_constr))
        print(
            f'CVRAsObjective2: weighted_constraint_vector={weighted_constraint_vector}')
        self.weighted_constraint_vector = weighted_constraint_vector

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

        cvr = calc_cvr(G, H, self.weighted_constraint_vector)

        ocv = calc_cv(G=G, H=H, config=self.config)

        # append the constraint violation as objective
        if self.append:
            out["F"] = anp.column_stack(
                [F, cvr, ocv])
        else:
            out["F"] = [cvr, ocv]

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
