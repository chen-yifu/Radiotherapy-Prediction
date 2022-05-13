# Showing the side-by-side comparison of RF and KNN imputation
import collections
from pprint import pprint
import readline  # avoid input length limit

if __name__ == '__main__':
    # while True:
    KNN_compares = input("Copy-paste the KNN imputation array:")
    RF_compares = input("Copy-paste the RF imputation array:")
    try:
        KNN_compares = eval(KNN_compares)
        RF_compares = eval(RF_compares)
        assert type(KNN_compares) == list and type(RF_compares) == list
        print(f"There are {len(KNN_compares)} KNN imputations"
              f" and {len(RF_compares)} RF imputations.")
        assert len(KNN_compares) == len(RF_compares)
        mapping = collections.defaultdict(dict)
        for compares, model in zip([KNN_compares, RF_compares], ["KNN", "RF"]):
            for gt, imp, pid in compares:
                if type(imp) == float:
                    imp = round(imp, 4)
                mapping[pid][model] = imp
                mapping[pid]["gt"] = gt
        # Order the dictionary by pid
        all_compares = collections.OrderedDict(sorted(mapping.items()))
        for pid, d in all_compares.items():
            print(f"{pid}: GT {d['gt']} | KNN {d['KNN']} | RF {d['RF']}")
    except Exception as e:
        print(f"There was an error parsing the input: {e}.")
        print("Please try again.")
