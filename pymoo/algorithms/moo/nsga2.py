import numpy as np
import warnings

from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.docs import parse_doc_string
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.selection.tournament import compare, TournamentSelection
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.util.dominator import Dominator
from pymoo.util.misc import has_feasible


# ---------------------------------------------------------------------------------------------------------
# Binary Tournament Selection Function
# ---------------------------------------------------------------------------------------------------------


def compare(a, a_val, b, b_val, method, return_random_if_equal=False):
    """Utility function to compare two solutions based on their attribute values."""
    if method == 'smaller_is_better':
        if a_val < b_val:
            return a
        elif a_val > b_val:
            return b
        else:
            return a if not return_random_if_equal else np.random.choice([a, b])
    elif method == 'larger_is_better':
        if a_val > b_val:
            return a
        elif a_val < b_val:
            return b
        else:
            return a if not return_random_if_equal else np.random.choice([a, b])
    else:
        raise ValueError("Unknown comparison method.")


def binary_tournament(pop, P, algorithm, **kwargs):
    n_tournaments, n_parents = P.shape
    if n_parents != 2:
        raise ValueError("Only implemented for binary tournament!")

    tournament_type = getattr(
        algorithm, 'tournament_type', 'comp_by_rank_and_crowding')
    S = np.full(n_tournaments, np.nan)

    for i in range(n_tournaments):
        a, b = P[i, 0], P[i, 1]
        a_cv, b_cv = getattr(pop[a], 'CV', np.inf), getattr(
            pop[b], 'CV', np.inf)
        a_rank, b_rank = getattr(pop[a], 'rank', np.inf), getattr(
            pop[b], 'rank', np.inf)
        a_cd, b_cd = getattr(pop[a], 'crowding', np.inf), getattr(
            pop[b], 'crowding', np.inf)

        # Compare first by feasibility
        if a_cv > 0.0 or b_cv > 0.0:
            S[i] = compare(
                a, a_cv, b, b_cv, method='smaller_is_better', return_random_if_equal=True)
        else:
            # If both are feasible, use the specified tournament type
            if tournament_type == 'comp_by_dom_and_crowding':
                rel = Dominator.get_relation(pop[a].F, pop[b].F)
                if rel == 1:
                    S[i] = a
                elif rel == -1:
                    S[i] = b
                else:
                    S[i] = compare(
                        a, a_cd, b, b_cd, method='larger_is_better', return_random_if_equal=True)
            elif tournament_type == 'comp_by_rank_and_crowding':
                S[i] = compare(a, a_rank, b, b_rank,
                               method='smaller_is_better')
                if np.isnan(S[i]):
                    S[i] = compare(
                        a, a_cd, b, b_cd, method='larger_is_better', return_random_if_equal=True)
            else:
                raise Exception("Unknown tournament type.")

    return S[:, None].astype(int, copy=False)


def compare(a, a_val, b, b_val, method='smaller_is_better', return_random_if_equal=True):
    if method == 'smaller_is_better':
        if a_val < b_val:
            return a
        elif a_val > b_val:
            return b
        elif return_random_if_equal:
            return np.random.choice([a, b])
    elif method == 'larger_is_better':
        if a_val > b_val:
            return a
        elif a_val < b_val:
            return b
        elif return_random_if_equal:
            return np.random.choice([a, b])
    else:
        raise ValueError("Unknown comparison method.")


def constr_binary_tournament(pop, P, algorithm, **kwargs):
    # TODO: add n_additional_obj to kwargs
    n_additional_obj = kwargs.get('n_additional_obj', 1)
    n_tournaments, n_parents = P.shape
    if n_parents != 2:
        raise ValueError("Only implemented for binary tournament!")

    S = np.full(n_tournaments, np.nan)

    for i in range(n_tournaments):
        a, b = P[i, 0], P[i, 1]

        if n_additional_obj == 1:
            a_cv = pop[a].F[-1]
            b_cv = pop[b].F[-1]
            a_cvr = 0
            b_cvr = 0
            original_a_F = pop[a].F[:-1]
            original_b_F = pop[b].F[:-1]
        elif n_additional_obj == 2:
            a_cvr, a_cv = pop[a].F[-2], pop[a].F[-1]
            b_cvr, b_cv = pop[b].F[-2], pop[b].F[-1]
            original_a_F = pop[a].F[:-2]
            original_b_F = pop[b].F[:-2]
        else:
            raise ValueError("Unsupported number of additional objectives")

        # Compare first by CVR
        if a_cvr < b_cvr:
            S[i] = a
        elif a_cvr > b_cvr:
            S[i] = b
        else:
            # Compare by CV if CVR is the same
            if a_cv < b_cv:
                S[i] = a
            elif a_cv > b_cv:
                S[i] = b
            else:
                # If both CVR and CV are equal, use the specified tournament type
                tournament_type = getattr(
                    algorithm, 'tournament_type', 'comp_by_rank_and_crowding')
                a_rank, b_rank = getattr(pop[a], 'rank', np.inf), getattr(
                    pop[b], 'rank', np.inf)
                a_cd, b_cd = getattr(pop[a], 'crowding', np.inf), getattr(
                    pop[b], 'crowding', np.inf)

                if tournament_type == 'comp_by_dom_and_crowding':
                    rel = Dominator.get_relation(original_a_F, original_b_F)
                    if rel == 1:
                        S[i] = a
                    elif rel == -1:
                        S[i] = b
                    else:
                        S[i] = compare(
                            a, a_cd, b, b_cd, method='larger_is_better', return_random_if_equal=True)
                elif tournament_type == 'comp_by_rank_and_crowding':
                    S[i] = compare(a, a_rank, b, b_rank,
                                   method='smaller_is_better')
                    if np.isnan(S[i]):
                        S[i] = compare(
                            a, a_cd, b, b_cd, method='larger_is_better', return_random_if_equal=True)
                else:
                    raise Exception("Unknown tournament type.")

    return S[:, None].astype(int, copy=False)


# ---------------------------------------------------------------------------------------------------------
# Survival Selection
# ---------------------------------------------------------------------------------------------------------


class RankAndCrowdingSurvival(RankAndCrowding):

    def __init__(self, nds=None, crowding_func="cd"):
        warnings.warn(
            "RankAndCrowdingSurvival is deprecated and will be removed in version 0.8.*; use RankAndCrowding operator instead, which supports several and custom crowding diversity metrics.",
            DeprecationWarning, 2
        )
        super().__init__(nds, crowding_func)

# =========================================================================================================
# Implementation
# =========================================================================================================


class NSGA2(GeneticAlgorithm):

    def __init__(self,
                 pop_size=100,
                 sampling=FloatRandomSampling(),
                 selection=TournamentSelection(func_comp=binary_tournament),
                 crossover=SBX(eta=15, prob=0.9),
                 mutation=PM(eta=20),
                 survival=RankAndCrowding(),
                 output=MultiObjectiveOutput(),
                 **kwargs):

        super().__init__(
            pop_size=pop_size,
            sampling=sampling,
            selection=selection,
            crossover=crossover,
            mutation=mutation,
            survival=survival,
            output=output,
            advance_after_initial_infill=True,
            **kwargs)

        self.termination = DefaultMultiObjectiveTermination()
        self.tournament_type = 'comp_by_dom_and_crowding'

    def _set_optimum(self, **kwargs):
        if not has_feasible(self.pop):
            self.opt = self.pop[[np.argmin(self.pop.get("CV"))]]
        else:
            self.opt = self.pop[self.pop.get("rank") == 0]


parse_doc_string(NSGA2.__init__)
