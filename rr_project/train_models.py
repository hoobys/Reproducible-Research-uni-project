from dataclasses import dataclass, field

import joblib
import pandas as pd
from loguru import logger
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from rr_project.config.const import SEED


def save_model(pipeline: Pipeline, model_name: str) -> None:
    """
    Save the model to a .pkl file.

    Parameters:
    pipeline (Pipeline): scikit-learn pipeline containing the model and scaler.
    model_name (str): Name of the model for saving the .pkl file.

    Returns:
    None
    """
    joblib.dump(pipeline, f"./models/{model_name}.pkl")
    print(f"Model {model_name} has been trained and saved successfully.")


def get_pipeline_for_model(model, model_params: dict = None):
    numerical_prep = make_pipeline(RobustScaler())
    categorical_prep = make_pipeline(
        OneHotEncoder(handle_unknown="ignore", sparse=False, drop="first"),
    )
    preprocess = ColumnTransformer(
        [
            (
                "numerical",
                numerical_prep,
                make_column_selector(dtype_include=["int64", "float64"]),
            ),
            (
                "categorical",
                categorical_prep,
                make_column_selector(dtype_include=object),
            ),
        ],
        remainder="passthrough",
    )
    preprocess.set_output(transform="pandas")
    return Pipeline(
        [
            ("preprocess", preprocess),
            ("model", model(**model_params if model_params else {})),
        ]
    )


@dataclass
class OneModelHyperoptResult:
    best_model: BaseEstimator
    best_score: float
    cv_results: pd.DataFrame

    def get_model_name(self) -> str:
        return self.best_model["model"].__class__.__name__


@dataclass
class HyperoptInput:
    model: BaseEstimator
    hyperopt_space: dict = field(default_factory=dict)


@dataclass
class HyperoptResults:
    results: list

    def __post_init__(self):
        self._sort_by_best_score()

    def _sort_by_best_score(self, reversed: bool = True):
        self.results.sort(key=lambda x: x.best_score, reverse=reversed)

    def get_best_model(self):
        return self.results[0].best_model

    def get_best_score(self):
        return self.results[0].best_score

    def get_merged_df(self):
        results = pd.DataFrame()
        for result in self.results:
            results = pd.concat(
                [
                    results,
                    result.cv_results.assign(model_name=result.get_model_name()),
                ],
                axis=0,
            )

        return results

    def get_all_dfs(self):
        return [(result.get_model_name(), result.cv_results) for result in self.results]

    def get_all_scores(self):
        return [(result.get_model_name(), result.best_score) for result in self.results]

    def get_all_models(self):
        return [(result.get_model_name(), result.best_model) for result in self.results]


def run_hyperopt_one_model(
    x: pd.DataFrame,
    y: pd.Series,
    model_input: HyperoptInput,
    n_iter: int = 10,
    cv: int = 5,
    random_state: int = SEED,
):
    pipeline = get_pipeline_for_model(model_input.model)
    search = RandomizedSearchCV(
        pipeline,
        model_input.hyperopt_space,
        n_iter=n_iter,
        scoring="average_precision",
        n_jobs=-1,
        cv=cv,
        random_state=random_state,
    )
    search.fit(x, y)
    return OneModelHyperoptResult(
        best_model=search.best_estimator_,
        best_score=search.best_score_,
        cv_results=pd.DataFrame(search.cv_results_),
    )


def run_hyperopt(
    hyperopt_inputs: list,
    x,
    y,
    n_iter: int = 10,
    cv: int = 5,
    random_state: int = SEED,
) -> HyperoptResults:
    results = []
    for model_input in hyperopt_inputs:
        logger.info(f"Running hyperopt for {model_input.model.__name__}")
        result = run_hyperopt_one_model(
            x=x,
            y=y,
            model_input=model_input,
            n_iter=n_iter,
            cv=cv,
            random_state=random_state,
        )
        results.append(result)
        logger.info(f"Best score: {result.best_score}")
    return HyperoptResults(results=results)
