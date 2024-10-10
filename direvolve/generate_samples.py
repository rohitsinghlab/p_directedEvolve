import numpy as np
import pandas as pd
from functools import reduce
import numpy as np
import pandas as pd
from functools import reduce
from scipy.special import softmax 
def get_candidates(locationsdf, no_mutations, no_samples = 1000, mode = "greedy", softmaxstrength=2):
    """
    Input
    -----
    Takes in ``locationsdf``, a pandas Dataframe with columns, 
                    [`location`, `residue`, `phenotype`]
    `location`, `residue` represent the location of missense mutation, with original aa represented by `residue`
    `phenotype` represents the magnitude of change observed due to the missense mutation ---- HIGHER IS BETTER, used for greedy search
    
    ``no_mutations`` : how many combinatorial mutations do you want to perform greedily using the missense mutations data from ``locationsdf``
    
    Output
    ------
    returns a pandas dataframe of greedy combinatorial mutations
    """
    locationsdf = locationsdf.loc[:, ["location", "residue", "phenotype"]]
    locationsdf = locationsdf.sort_values(by = ["phenotype"], ascending = False,)
    
    if mode == "greedy":
        """
        Use only for `no_mutations` <= 3, and `len(locationsdf)` <= 100. Otherwise, will take forever.
        """
        combinedf = locationsdf.copy()
        for i in range(no_mutations-1):
            combinedf = combinedf.merge(locationsdf, how = "cross")
            combinedf.columns = reduce(lambda x, y: x + y, [
                [f"location_{j}", f"residue_{j}", f"phenotype_{j}"] for j in range(i+2)
            ])
            allowed_combinations = combinedf.apply(lambda x: len({x[f"location_{k}"] for k in range(i+2)}) == i+2, axis = 1)
            combinedf = combinedf.loc[allowed_combinations, :]
        combinedf["combined_phenotypes"] = combinedf.apply(lambda x : np.sum([x[f"phenotype_{l}"] for l in range(no_mutations)]) , axis = 1)
        combinedf["combined_mutations"]  = combinedf.apply(lambda x : ":".join([f"{loc}-{res}" 
                                                                                for loc, res in
                                                                                sorted([(x[f"location_{m}"], x[f"residue_{m}"]) for m in range(i+2)],
                                                                                       key = lambda y : y[0]
                                                                                      )]
                                                                              ), axis = 1)
        combinedf = combinedf.drop_duplicates(subset = "combined_mutations")
        combinedf = combinedf.sort_values(by = "combined_phenotypes", ascending = False).reset_index(drop = True)
        return combinedf if len(combinedf) < no_samples else combinedf.loc[:no_samples, :]
    
    else:
        """
        Semi-greedy mode. 
        """
        mutation_probs = softmax(softmaxstrength * 
                                 locationsdf["phenotype"].values)
        allchoices     = locationsdf.values 
        allchoiceidx   = np.arange(len(allchoices), dtype = int)
        mutations      = set()
        remaining      = no_samples
        selected       = []
        minrankdms     = [] # of all mutations in a candidate, what is the rank of the mutation with highest positive phenotype? Stores this for all 
        maxrankdms     = []
        columns        = reduce(lambda x, y: x + y, [[f"location_{i}", 
                                                      f"residue_{i}", 
                                                      f"phenotype_{i}"] 
                                                     for i in range(no_mutations)])
        while(True):
            choiceids = np.random.choice(allchoiceidx, no_mutations, 
                                      p = mutation_probs)
            choice    = [allchoices[c].tolist() for c in choiceids]
            
            if len({ch[0] for ch in choice}) < no_mutations:
                continue
            index = ":".join([f"{loc}-{aa}" for loc, aa, _ 
                              in sorted(choice, key = lambda x : x[0])
                             ])
            if index in mutations:
                continue
            else:
                minrankdms.append(np.min(choiceids))
                maxrankdms.append(np.max(choiceids))
                mutations.add(index)
                selected.append(reduce(lambda x, y : x + y, choice))
                remaining -= 1
                if remaining == 0: 
                    break
        combinedf = pd.DataFrame(selected, columns = columns)
        combinedf["combined_phenotypes"] = combinedf.apply(lambda x : np.sum([x[f"phenotype_{l}"] 
                                                                              for l in range(no_mutations)]), 
                                                           axis = 1)
        combinedf["rank_of_best_performing_mutation"]  = minrankdms
        combinedf["rank_of_worst_performing_mutation"] = maxrankdms
        return combinedf
