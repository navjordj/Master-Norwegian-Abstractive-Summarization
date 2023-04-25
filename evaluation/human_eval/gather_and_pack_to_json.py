import pandas as pd
import copy
import json

DIMENSIONS = ["Informativeness", "Relevance",
              "Fluency", "Coherence", "Factuality"]

DATASETS_RES = {
    "Aftenposten": {},
    "Wikipedia": {},
    "Kvinneguiden": {},
    "Komplett": {},

    "SNL_Bottom": {},
    "SNL_Top": {},
    "SNL_Random": {},


    "CNN_Bottom": {},
    "CNN_Top": {},
    "CNN_Random": {},
}
DATASET_INDICES = {
    "Aftenposten": [1, 2],
    "Wikipedia": [3, 4],
    "Kvinneguiden": [5],
    "Komplett": [6],
    "SNL_Bottom": [i for i in range(7, 12)],
    "SNL_Top": [i for i in range(12, 17)],
    "SNL_Random": [i for i in range(17, 22)],
    "CNN_Bottom": [i for i in range(22, 27)],
    "CNN_Top": [i for i in range(27, 32)],
    "CNN_Random": [i for i in range(32, 37)]
}

MODEL_RES = {
    "CNN Base": {},
    "CNN Large": {},
    "SNL Base": {},
    "SNL Large": {},
}


DIMENSIONS_RES = {
    "Informativeness": [],
    "Relevance": [],
    "Fluency": [],
    "Coherence": [],
    "Factuality": [],
    "Overall": [],
}


def pack_row_results(row, dataset_dict, dataset_key, dimension_key):
    # To be implemented, if we want less nested loops logic
    pass


def is_nan(value):
    if value is None or str(value) == "NaN" or str(value) == "nan":
        return True
    return False


def get_human_eval_results_from_csv(path: str) -> dict:
    if path is None:
        raise ValueError(
            "Path is None, must be a valid path to a csv file in human_eval_results/"
        )

    df = pd.read_csv(
        path, header=[0, 1], index_col=0)

    # DEEP COPY IS IMPORTANT for all the templates
    dataset_results = copy.deepcopy(DATASETS_RES)

    for dataset_key, value in DATASET_INDICES.items():
        dataset_results[dataset_key] = copy.deepcopy(DIMENSIONS_RES)

        for row_index in value:

            row = df.loc[str(row_index)]
            dimensions_index = 0
            dimension_key = None

            for col_key, value in row.items():
                # Skips missing values as they are not relevant
                # We can infer missing values later as the number of samples is fixed

                # Updates the dimension key as we iterate through the columns
                # and works through the list with indexing
                # both the data and the dimensions are in the same order
                if col_key[0] in DIMENSIONS[dimensions_index:]:
                    dimension_key = DIMENSIONS[dimensions_index]
                    dimensions_index += 1
                if dimension_key is None:
                    continue

                # print(value)
                # if value is None or str(value) == "NaN" or str(value) == "nan":
                # print(value)
                #    continue
                # The values are read correctly now, and only added if they are not NaN
                # TODO: evaluate if we want to add the NaN values as 0, None or something else
                # Makes sure we don't add the columns not in the dimensions list
                # as all of them have "Kors" and "Nav" secondary keys
                if col_key[1] == "Kors":
                    if not is_nan(value):
                        dataset_results[dataset_key][dimension_key].append(
                            int(value))
                elif col_key[1] == "Nav":
                    if not is_nan(value):
                        dataset_results[dataset_key][dimension_key].append(
                            int(value))
                else:
                    continue

    # print(dataset_results)
    return dataset_results


def main():
    # for each dimension
    # and for each model
    # and for each dataset
    # average out the scores between "Kors" and "Nav"
    # and average out the scores between the 10 samplings listed in DATASET_RES
    total_results = copy.deepcopy(MODEL_RES)

    total_results["CNN Base"] = get_human_eval_results_from_csv(
        "human_eval_results/EvaluationModels - Eval CNN Base.csv")
    print("Done with CNN Base")
    total_results["CNN Large"] = get_human_eval_results_from_csv(
        "human_eval_results/EvaluationModels - Eval CNN Large.csv")
    print("Done with CNN Large")
    total_results["SNL Base"] = get_human_eval_results_from_csv(
        "human_eval_results/EvaluationModels - Eval SNL Base.csv")
    print("Done with SNL Base")
    total_results["SNL Large"] = get_human_eval_results_from_csv(
        "human_eval_results/EvaluationModels - Eval SNL Large.csv")
    print("Done with SNL Large")

    with open("human_eval_results_gathered_and_packed.json", "w") as write_file:
        json.dump(total_results, write_file, indent=4)
    print("Done writing JSON data into file with indent=4")


if __name__ == "__main__":
    main()
