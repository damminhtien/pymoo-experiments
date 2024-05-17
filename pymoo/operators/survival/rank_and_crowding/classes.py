from concurrent.futures import ThreadPoolExecutor
import numpy as np
from pymoo.util.randomized_argsort import randomized_argsort
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.core.survival import Survival, split_by_feasibility
from pymoo.core.population import Population
from pymoo.operators.survival.rank_and_crowding.metrics import get_crowding_function


class RankAndCrowding(Survival):

    def __init__(self, nds=None, crowding_func="cd"):
        """
        A generalization of the NSGA-II survival operator that ranks individuals by dominance criteria
        and sorts the last front by some user-specified crowding metric. The default is NSGA-II's crowding distances
        although others might be more effective.

        For many-objective problems, try using 'mnn' or '2nn'.

        For Bi-objective problems, 'pcd' is very effective.

        Parameters
        ----------
        nds : str or None, optional
            Pymoo type of non-dominated sorting. Defaults to None.

        crowding_func : str or callable, optional
            Crowding metric. Options are:

                - 'cd': crowding distances
                - 'pcd' or 'pruning-cd': improved pruning based on crowding distances
                - 'ce': crowding entropy
                - 'mnn': M-Nearest Neighbors
                - '2nn': 2-Nearest Neighbors

            If callable, it has the form ``fun(F, filter_out_duplicates=None, n_remove=None, **kwargs)``
            in which F (n, m) and must return metrics in a (n,) array.

            The options 'pcd', 'cd', and 'ce' are recommended for two-objective problems, whereas 'mnn' and '2nn' for many objective.
            When using 'pcd', 'mnn', or '2nn', individuals are already eliminated in a 'single' manner. 
            Due to Cython implementation, they are as fast as the corresponding 'cd', 'mnn-fast', or '2nn-fast', 
            although they can singnificantly improve diversity of solutions.
            Defaults to 'cd'.
        """

        crowding_func_ = get_crowding_function(crowding_func)

        super().__init__(filter_infeasible=True)
        self.nds = nds if nds is not None else NonDominatedSorting()
        self.crowding_func = crowding_func_

    def _do(self,
            problem,
            pop,
            *args,
            n_survive=None,
            **kwargs):

        # get the objective space values and objects
        F = pop.get("F").astype(float, copy=False)

        # the final indices of surviving individuals
        survivors = []

        # do the non-dominated sorting until splitting front
        fronts = self.nds.do(F, n_stop_if_ranked=n_survive)

        for k, front in enumerate(fronts):

            I = np.arange(len(front))

            # current front sorted by crowding distance if splitting
            if len(survivors) + len(I) > n_survive:

                # Define how many will be removed
                n_remove = len(survivors) + len(front) - n_survive

                # re-calculate the crowding distance of the front
                crowding_of_front = \
                    self.crowding_func.do(
                        F[front, :],
                        n_remove=n_remove
                    )

                I = randomized_argsort(
                    crowding_of_front, order='descending', method='numpy')
                I = I[:-n_remove]

            # otherwise take the whole front unsorted
            else:
                # calculate the crowding distance of the front
                crowding_of_front = \
                    self.crowding_func.do(
                        F[front, :],
                        n_remove=0
                    )

            # save rank and crowding in the individual class
            for j, i in enumerate(front):
                pop[i].set("rank", k)
                pop[i].set("crowding", crowding_of_front[j])

            # extend the survivors by all or selected individuals
            survivors.extend(front[I])

        return pop[survivors]


class MyConstrRankAndCrowding(Survival):
    def __init__(self, nds=None, crowding_func="cd"):
        super().__init__(filter_infeasible=False)
        self.nds = nds if nds is not None else NonDominatedSorting()
        self.ranking = RankAndCrowding(nds=nds, crowding_func=crowding_func)

    def _do(self, problem, pop, *args, n_survive=None, **kwargs):
        if n_survive is None:
            n_survive = len(pop)

        F = pop.get("F")
        CV = F[:, -1]

        # Boolean masks for feasible and infeasible individuals
        feasible_mask = CV <= 0
        infeasible_mask = CV > 0

        # Use boolean masks directly to create new populations
        feas_pop = pop[feasible_mask]
        infeas_pop = pop[infeasible_mask]

        # Rank and select feasible individuals
        if feas_pop.size > 0:
            feas_survivors = self.ranking.do(
                problem, feas_pop, *args, n_survive=min(len(feas_pop), n_survive), **kwargs)
        else:
            feas_survivors = Population()

        n_remaining = n_survive - len(feas_survivors)

        # Process infeasible individuals if needed
        if n_remaining > 0 and infeas_pop.size > 0:
            # Sort infeasible population by CV and select the best
            sorted_indices = np.argsort(infeas_pop.get("F")[:, -1])
            sorted_infeas_pop = infeas_pop[sorted_indices[:n_remaining]]

            # Merge feasible and the best infeasible solutions
            survivors = Population.merge(feas_survivors, sorted_infeas_pop)
        else:
            survivors = feas_survivors

        return survivors


class MyConstrRankAndCrowding2(Survival):
    def __init__(self, nds=None, crowding_func="cd"):
        super().__init__(filter_infeasible=False)
        self.nds = nds if nds is not None else NonDominatedSorting()
        self.ranking = RankAndCrowding(nds=nds, crowding_func=crowding_func)

    def _do(self, problem, pop, *args, n_survive=None, **kwargs):
        if n_survive is None:
            n_survive = len(pop)

        F = pop.get("F")
        CVR = F[:, -2]  # Constraint Violation Ratio
        CV = F[:, -1]   # Overall Constraint Violation

        # Boolean masks for feasible and infeasible individuals
        feasible_mask = CV <= 0
        infeasible_mask = CV > 0

        # Use boolean masks directly to create new populations
        feas_pop = pop[feasible_mask]
        infeas_pop = pop[infeasible_mask]

        # Rank and select feasible individuals
        if len(feas_pop) > 0:
            feas_survivors = self.ranking.do(
                problem, feas_pop, *args, n_survive=min(len(feas_pop), n_survive), **kwargs)
        else:
            feas_survivors = Population()

        n_remaining = n_survive - len(feas_survivors)

        # Process infeasible individuals if needed
        if n_remaining > 0 and len(infeas_pop) > 0:
            # Filter CVR and CV for infeasible individuals
            infeas_CVR = CVR[infeasible_mask]
            infeas_CV = CV[infeasible_mask]

            # Sort infeasible population by CVR first, then by CV if CVR values are equal
            sorted_indices = np.lexsort((infeas_CV, infeas_CVR))

            # Ensure n_remaining does not exceed the size of infeas_pop
            n_to_select = min(n_remaining, len(infeas_pop))

            if n_to_select > 0:
                sorted_infeas_pop = infeas_pop[sorted_indices[:n_to_select]]
            else:
                sorted_infeas_pop = Population()

            # Merge feasible and the best infeasible solutions
            survivors = Population.merge(feas_survivors, sorted_infeas_pop)
        else:
            survivors = feas_survivors

        return survivors


class ParallelConstrRankAndCrowding(Survival):
    def __init__(self, nds=None, crowding_func="cd"):
        super().__init__(filter_infeasible=False)
        self.nds = nds if nds is not None else NonDominatedSorting()
        self.ranking = RankAndCrowding(nds=nds, crowding_func=crowding_func)

    def _do(self, problem, pop, *args, n_survive=None, **kwargs):
        if n_survive is None:
            n_survive = len(pop)

        F = pop.get("F")
        CV = F[:, -1]
        feasible_mask = CV <= 0
        infeasible_mask = CV > 0
        feas_pop = pop[feasible_mask]
        infeas_pop = pop[infeasible_mask]

        # Use ThreadPoolExecutor to parallelize the ranking and sorting operations
        with ThreadPoolExecutor() as executor:
            # Future for ranking feasible individuals
            feas_future = executor.submit(self.ranking.do, problem, feas_pop, *args,
                                          n_survive=min(len(feas_pop), n_survive), **kwargs) if feas_pop.size > 0 else None

            # Future for sorting infeasible individuals by CV
            if n_survive > len(feas_pop) and infeas_pop.size > 0:
                n_remaining = n_survive - len(feas_pop)
                infeas_future = executor.submit(
                    self.sort_infeasibles, infeas_pop, n_remaining)
            else:
                infeas_future = None

            feas_survivors = feas_future.result() if feas_future else Population()
            sorted_infeas_pop = infeas_future.result() if infeas_future else Population()

        # Merge feasible and the best infeasible solutions
        survivors = Population.merge(
            feas_survivors, sorted_infeas_pop) if infeas_future else feas_survivors

        return survivors

    def sort_infeasibles(self, infeas_pop, n_remaining):
        sorted_indices = np.argsort(infeas_pop.get("F")[:, -1])
        return infeas_pop[sorted_indices[:n_remaining]]


class ConstrRankAndCrowding(Survival):

    def __init__(self, nds=None, crowding_func="cd"):
        """
        The Rank and Crowding survival approach for handling constraints proposed on
        GDE3 by Kukkonen, S. & Lampinen, J. (2005).

        Parameters
        ----------
        nds : str or None, optional
            Pymoo type of non-dominated sorting. Defaults to None.

        crowding_func : str or callable, optional
            Crowding metric. Options are:

                - 'cd': crowding distances
                - 'pcd' or 'pruning-cd': improved pruning based on crowding distances
                - 'ce': crowding entropy
                - 'mnn': M-Nearest Neighbors
                - '2nn': 2-Nearest Neighbors

            If callable, it has the form ``fun(F, filter_out_duplicates=None, n_remove=None, **kwargs)``
            in which F (n, m) and must return metrics in a (n,) array.

            The options 'pcd', 'cd', and 'ce' are recommended for two-objective problems, whereas 'mnn' and '2nn' for many objective.
            When using 'pcd', 'mnn', or '2nn', individuals are already eliminated in a 'single' manner. 
            Due to Cython implementation, they are as fast as the corresponding 'cd', 'mnn-fast', or '2nn-fast', 
            although they can singnificantly improve diversity of solutions.
            Defaults to 'cd'.
        """

        super().__init__(filter_infeasible=False)
        self.nds = nds if nds is not None else NonDominatedSorting()
        self.ranking = RankAndCrowding(nds=nds, crowding_func=crowding_func)

    def _do(self,
            problem,
            pop,
            *args,
            n_survive=None,
            **kwargs):

        if n_survive is None:
            n_survive = len(pop)

        n_survive = min(n_survive, len(pop))

        # If the split should be done beforehand
        if problem.n_constr > 0:

            # Split by feasibility
            feas, infeas = split_by_feasibility(
                pop, sort_infeas_by_cv=True, sort_feas_by_obj=False, return_pop=False)

            # Obtain len of feasible
            n_feas = len(feas)

            # Assure there is at least_one survivor
            if n_feas == 0:
                survivors = Population()
            else:
                survivors = self.ranking.do(
                    problem, pop[feas], *args, n_survive=min(len(feas), n_survive), **kwargs)

            # Calculate how many individuals are still remaining to be filled up with infeasible ones
            n_remaining = n_survive - len(survivors)

            # If infeasible solutions need to be added
            if n_remaining > 0:

                # Constraints to new ranking
                G = pop[infeas].get("G")
                G = np.maximum(G, 0)
                H = pop[infeas].get("H")
                H = np.absolute(H)
                C = np.column_stack((G, H))

                # Fronts in infeasible population
                infeas_fronts = self.nds.do(C, n_stop_if_ranked=n_remaining)

                # Iterate over fronts
                for k, front in enumerate(infeas_fronts):

                    # Save ranks
                    pop[infeas][front].set("cv_rank", k)

                    # Current front sorted by CV
                    if len(survivors) + len(front) > n_survive:

                        # Obtain CV of front
                        CV = pop[infeas][front].get("CV").flatten()
                        I = randomized_argsort(
                            CV, order='ascending', method='numpy')
                        I = I[:(n_survive - len(survivors))]

                    # Otherwise take the whole front unsorted
                    else:
                        I = np.arange(len(front))

                    # extend the survivors by all or selected individuals
                    survivors = Population.merge(
                        survivors, pop[infeas][front[I]])

        else:
            survivors = self.ranking.do(
                problem, pop, *args, n_survive=n_survive, **kwargs)

        return survivors
