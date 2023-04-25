import pandas as pd

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


def get_human_eval_results_from_csv(path=None):
    if path is None:
        return None

    df = pd.read_csv(
        path, header=[0, 1], index_col=0)

    dataset_results = DATASETS_RES.copy()
    # aggregate the results for "Kors" and "Nav" for the five dimensions
    # "Informativeness", "Relevance", "Fluency", "Coherence", "Factuality"

    # average out the scores between "Kors" and "Nav" for each dimension

    # We have 36 samples that were evaluated by both "Kors" and "Nav"
    # valid_indices = [str(i) for i in range(36)]
    for dataset_key, value in DATASET_INDICES.items():
        # print(dataset_key, value)
        dataset_results[dataset_key] = DIMENSIONS_RES.copy()
        for row_index in value:
            row = df.loc[str(row_index)]

            dimensions_index = 0
            dimension_key = None
            for col_key, value in row.items():
                print(col_key, value)
                if col_key[0] in DIMENSIONS[dimensions_index:-1]:
                    dimensions_key = DIMENSIONS[dimensions_index]
                    print(dimensions_key)
                    dimensions_index += 1
                if dimensions_key is None:
                    continue

                if col_key[1] == "Kors":
                    print(dataset_key)
                    print(dimension_key)
                    dataset_results[dataset_key][dimension_key].append(value)

                elif col_key[1] == "Nav":
                    dataset_results[dataset_key][dimension_key].append(value)
                else:
                    continue

    print(dataset_results)


def main():
    # for each dimension
    # and for each model
    # and for each dataset
    # average out the scores between "Kors" and "Nav"

    total_results = MODEL_RES.copy()

    total_results["CNN Base"] = get_human_eval_results_from_csv(
        "human_eval_results/EvaluationModels - Eval CNN Base.csv"
    )
    total_results["CNN Large"] = get_human_eval_results_from_csv(
        "human_eval_results/EvaluationModels - Eval CNN Large.csv"
    )
    total_results["SNL Base"] = get_human_eval_results_from_csv(
        "human_eval_results/EvaluationModels - Eval SNL Base.csv"
    )
    total_results["SNL Large"] = get_human_eval_results_from_csv(
        "human_eval_results/EvaluationModels - Eval SNL Large.csv"
    )
    print(total_results)


if __name__ == "__main__":
    main()
