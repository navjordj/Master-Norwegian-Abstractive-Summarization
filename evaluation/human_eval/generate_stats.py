import json
import pandas as pd


def get_human_eval_results_from_json(path: str) -> dict:
    with open(path, "r") as f:
        data = json.load(f)
        # print(data)
    return data


def mean(lst):
    print(lst)
    if lst is None:
        return None
    if lst == []:
        return None
    return sum(lst) / len(lst)


def get_mean_scores_from_json(filepath: str) -> dict:
    # data = get_human_eval_results_from_json(filepath)
    # If orient="index", the columns will be the datasets.
    # df = pd.read_json(filepath, orient="index")
    # print(df)
    # for dataset in df.columns:
    #    print(dataset)

    # If orient="columns", the columns will be the models.
    model_scores = {}
    df = pd.read_json(filepath, orient="columns")
    # print(df)
    for dataset in df.iterrows():
        # print(dataset)
        for model in df.columns:
            # print(model)
            dimension_dict = df.loc[dataset[0], model]
            entry_name = f"{dataset[0]} {model}"
            # print(type(dimension_dict))
            scores_list = []
            for dimension, scores in dimension_dict.items():
                # print(dimension, scores, type(scores))
                mean_score = mean(scores)
                if dimension == "Overall":
                    # overwrite the overall score with the mean of the other scores
                    # print(scores_list)
                    model_scores[entry_name][dimension] = mean(scores_list)
                    continue

                if entry_name not in model_scores:
                    model_scores[entry_name] = {}
                if dimension not in model_scores[entry_name]:
                    model_scores[entry_name][dimension] = mean_score
                # model_scores[entry_name][dimension].append(scores)

                # Logic for getting the mean of the scores
                # print(entry_name, dimension, scores)
                if mean_score is not None:
                    scores_list.append(mean(scores))

    return model_scores


def main():
    filepath = "human_eval_results_gathered_and_packed.json"
    get_mean_scores_from_json(filepath)


if __name__ == "__main__":
    main()
