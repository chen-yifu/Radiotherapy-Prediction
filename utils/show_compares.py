# Showing the side-by-side comparison of RF and KNN imputation
import collections
import readline  # avoids input length limit


class bcolors:
    # Helper class to print in terminal with colors
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    NORMAL = '\033[0m'


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
        # Calculate metrics
        for compares, model in zip([KNN_compares, RF_compares], ["KNN", "RF"]):
            total_sq_error, total_count = 0, 0
            for gt, imp, _ in compares:
                total_sq_error += (gt - imp) ** 2
                total_count += 1
            rmse = (total_sq_error/total_count) ** 0.5
            rmse = round(rmse, 4)
            print(f"RMSE of {model} imputation: {rmse}")
        # Print the side-by-side comparison
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
            print(f"ID {pid} | GT {d['gt']} | ", end="")
            correct = bcolors.OKGREEN
            if d["gt"] == d["KNN"]:
                print(f"{correct}KNN {d['KNN']}{bcolors.ENDC} |", end=" ")
            else:
                print(f"KNN {d['KNN']} |", end=" ")
            if d["gt"] == d["RF"]:
                print(f"{correct}RF {d['RF']}{bcolors.ENDC}")
            else:
                print(f"RF {d['RF']}")

    except Exception as e:
        print(f"There was an error parsing the input: {e}.")
        print("Please try again.")
