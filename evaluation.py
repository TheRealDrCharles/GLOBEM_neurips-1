import pickle
import pandas, numpy

with open("evaluation_output/evaluation_single_dataset_within_user/dep_weekly/ml_chikersal.pkl", "rb") as f:
    evaluation_results = pickle.load(f)
    df = pandas.DataFrame(evaluation_results["results_repo"]["dep_weekly"]).T
    print(df[["test_balanced_acc", "test_roc_auc"]])
