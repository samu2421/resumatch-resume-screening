"""
Train and evaluate the resume classification model.

Run:  python train_model.py
The dataset path is read from DATASET_PATH (default data/resume_dataset.csv).
Three classifiers are compared with stratified k-fold CV on the deduplicated
resumes; the one with the best mean macro F1 that can give probabilities is
trained on all deduplicated resumes and saved to
model/resume_classifier_pipeline.pkl. Metrics go to results/metrics.json.
"""
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, make_scorer,
                             precision_recall_fscore_support, precision_score,
                             recall_score, f1_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_validate, train_test_split)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from preprocess import ResumePreprocessor

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'model' / 'resume_classifier_pipeline.pkl'
METRICS_PATH = BASE_DIR / 'results' / 'metrics.json'
DEFAULT_DATASET_PATH = BASE_DIR / 'data' / 'resume_dataset.csv'
RANDOM_STATE = 42
MAX_FOLDS = 5


# Candidate classifiers (no hyperparameter tuning beyond these settings).
MODELS = {
    'MultinomialNB': MultinomialNB,
    'LogisticRegression': lambda: LogisticRegression(
        max_iter=1000, class_weight='balanced'),
    'LinearSVC': lambda: LinearSVC(class_weight='balanced'),
}
ORIGINAL_MODEL = 'MultinomialNB'
TFIDF_SETTINGS = ('TF-IDF (1-2 grams, max_features=10000, min_df=2, '
                  'max_df=0.85)')


def build_pipeline(classifier=None) -> Pipeline:
    """TF-IDF pipeline (unchanged settings) with the given classifier."""
    return Pipeline([
        ('vectorizer', TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.85
        )),
        ('classifier', classifier if classifier is not None
         else MODELS[ORIGINAL_MODEL]())
    ])


class ResumeClassifier:
    """Predict job categories for resumes with a trained pipeline."""

    def __init__(self, pipeline=None):
        self.preprocessor = ResumePreprocessor()
        self.pipeline = pipeline

    @classmethod
    def load(cls, model_path=MODEL_PATH):
        """
        Load a trained pipeline from disk.

        Raises FileNotFoundError if the model file does not exist.
        """
        with open(model_path, 'rb') as f:
            return cls(pickle.load(f))

    def predict_top(self, resume_text, k=3):
        """
        Return the k most probable categories for raw resume text as a list
        of (category, probability) pairs, most probable first.

        The text is cleaned here with the same preprocessing used in
        training.
        """
        if self.pipeline is None:
            raise RuntimeError(
                "No trained model loaded. Run: python train_model.py")
        cleaned = self.preprocessor.clean_text(resume_text)
        probabilities = self.pipeline.predict_proba([cleaned])[0]
        order = np.argsort(probabilities)[::-1][:k]
        return [(str(self.pipeline.classes_[i]), float(probabilities[i]))
                for i in order]

    def predict_category(self, resume_text):
        """
        Predict the job category for raw resume text.

        Returns (category, confidence between 0 and 1).
        """
        return self.predict_top(resume_text, k=1)[0]


def load_dataset(dataset_path, preprocessor):
    """Load the CSV and add a cleaned_text column."""
    df = pd.read_csv(dataset_path)
    df['cleaned_text'] = df['Resume'].apply(preprocessor.clean_text)
    return df


def _scores(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0)
    return {
        'accuracy': round(accuracy_score(y_true, y_pred), 4),
        'precision_macro': round(float(precision), 4),
        'recall_macro': round(float(recall), 4),
        'f1_macro': round(float(f1), 4),
    }


def original_method(df):
    """
    The ORIGINAL evaluation, for comparison only: an 80/20 split of all
    962 rows without removing duplicates (same arguments as before).
    """
    x_train, x_test, y_train, y_test = train_test_split(
        df['cleaned_text'], df['Category'], test_size=0.2,
        random_state=RANDOM_STATE, stratify=df['Category'])
    pipeline = build_pipeline().fit(x_train, y_train)
    result = _scores(y_test, pipeline.predict(x_test))
    train_texts = set(df.loc[x_train.index, 'Resume'])
    leaked = int(df.loc[x_test.index, 'Resume'].isin(train_texts).sum())
    result.update({
        'test_size': int(len(x_test)),
        'test_resumes_with_identical_copy_in_train': leaked,
    })
    return result


def cross_validation(df, n_splits, classifier=None):
    """Stratified k-fold CV on the deduplicated data (fixed folds)."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True,
                         random_state=RANDOM_STATE)
    scoring = {
        'accuracy': 'accuracy',
        'precision_macro': make_scorer(precision_score, average='macro',
                                       zero_division=0),
        'recall_macro': make_scorer(recall_score, average='macro',
                                    zero_division=0),
        'f1_macro': make_scorer(f1_score, average='macro', zero_division=0),
    }
    scores = cross_validate(build_pipeline(classifier), df['cleaned_text'],
                            df['Category'], cv=cv, scoring=scoring,
                            error_score='raise')
    result = {'n_splits': n_splits}
    for name in scoring:
        values = scores[f'test_{name}']
        result[name] = {
            'mean': round(float(values.mean()), 4),
            'std': round(float(values.std()), 4),
            'per_fold': [round(float(v), 4) for v in values],
        }
    return result


def holdout(df, classifier=None):
    """Single stratified 80/20 hold-out on the deduplicated data."""
    x_train, x_test, y_train, y_test = train_test_split(
        df['cleaned_text'], df['Category'], test_size=0.2,
        random_state=RANDOM_STATE, stratify=df['Category'])
    pipeline = build_pipeline(classifier).fit(x_train, y_train)
    result = _scores(y_test, pipeline.predict(x_test))
    result.update({'train_size': int(len(x_train)),
                   'test_size': int(len(x_test))})
    return result


def select_model(comparison, df, n_splits):
    """
    Pick the model with the best mean macro F1 that can output
    probabilities (needed for the confidence score).

    LinearSVC has no predict_proba; it is only accepted if a
    CalibratedClassifierCV(cv=n_splits) wrapper can be evaluated on the same
    CV folds. Otherwise the next best model is used.
    Returns (name, classifier factory, notes).
    """
    ranking = sorted(comparison,
                     key=lambda name: comparison[name]['f1_macro']['mean'],
                     reverse=True)
    notes = []
    for name in ranking:
        if name != 'LinearSVC':
            return name, MODELS[name], notes

        def calibrated():
            return CalibratedClassifierCV(MODELS['LinearSVC'](), cv=n_splits)
        try:
            cross_validation(df, n_splits, calibrated())
        except ValueError as error:
            notes.append(
                f"LinearSVC had the best macro F1 but has no predict_proba. "
                f"CalibratedClassifierCV(cv={n_splits}) could not be "
                f"evaluated on the same {n_splits}-fold CV: {error} "
                f"(the smallest category has too few resumes per training "
                f"fold). The next best model was selected instead.")
            continue
        notes.append(f"LinearSVC wrapped in "
                     f"CalibratedClassifierCV(cv={n_splits}).")
        return 'LinearSVC (calibrated)', calibrated, notes
    raise RuntimeError('No usable model found')


# Hyperparameter grid for the tuned LogisticRegression pipeline.
# max_features=10000 and max_df=0.85 stay fixed.
TUNING_GRID = {
    'classifier__C': [0.1, 1, 10, 100],
    'classifier__class_weight': [None, 'balanced'],
    'vectorizer__ngram_range': [(1, 1), (1, 2)],
    'vectorizer__sublinear_tf': [False, True],
    'vectorizer__min_df': [1, 2],
}
INNER_FOLDS = 2  # smallest class has only 2 resumes per outer training fold


def _grid_search(n_folds):
    return GridSearchCV(
        build_pipeline(LogisticRegression(max_iter=1000)),
        TUNING_GRID,
        scoring='f1_macro',
        cv=StratifiedKFold(n_splits=n_folds, shuffle=True,
                           random_state=RANDOM_STATE),
        n_jobs=-1,
    )


def _jsonable_params(params):
    return {key: list(value) if isinstance(value, tuple) else value
            for key, value in params.items()}


def nested_cv_tuning(df, n_splits):
    """
    Nested CV for LogisticRegression: outer stratified n_splits-fold CV
    (same folds as the model comparison), inner GridSearchCV with
    stratified INNER_FOLDS-fold CV on each outer training fold.
    """
    outer = StratifiedKFold(n_splits=n_splits, shuffle=True,
                            random_state=RANDOM_STATE)
    x, y = df['cleaned_text'], df['Category']
    fold_scores, best_params = [], []
    for train_idx, test_idx in outer.split(x, y):
        search = _grid_search(INNER_FOLDS)
        search.fit(x.iloc[train_idx], y.iloc[train_idx])
        y_pred = search.predict(x.iloc[test_idx])
        fold_scores.append(_scores(y.iloc[test_idx], y_pred))
        best_params.append(_jsonable_params(search.best_params_))

    result = {}
    for metric in ('accuracy', 'precision_macro', 'recall_macro',
                   'f1_macro'):
        values = np.array([fold[metric] for fold in fold_scores])
        result[metric] = {
            'mean': round(float(values.mean()), 4),
            'std': round(float(values.std()), 4),
            'per_fold': [round(float(v), 4) for v in values],
        }
    result['best_params_per_outer_fold'] = best_params
    return result


def main():
    dataset_path = Path(os.getenv('DATASET_PATH', DEFAULT_DATASET_PATH))
    print(f"Dataset: {dataset_path}")
    df = load_dataset(dataset_path, ResumePreprocessor())

    rows_before = len(df)
    dedup = df.drop_duplicates(subset='Resume').reset_index(drop=True)
    rows_after = len(dedup)
    n_categories = int(dedup['Category'].nunique())
    print(f"Rows before deduplication: {rows_before}")
    print(f"Rows after deduplication:  {rows_after}")
    print(f"Categories: {n_categories}")

    class_counts = dedup['Category'].value_counts()
    n_splits = int(min(MAX_FOLDS, class_counts.min()))
    if n_splits < MAX_FOLDS:
        small = class_counts[class_counts < MAX_FOLDS]
        print(f"Note: some categories have fewer than {MAX_FOLDS} unique "
              f"resumes ({small.to_dict()}); using {n_splits}-fold CV.")

    print(f"\nOriginal method ({ORIGINAL_MODEL}, no deduplication, "
          "80/20 split)...")
    original = original_method(df)
    print(original)

    print(f"\nModel comparison: stratified {n_splits}-fold CV on "
          f"deduplicated data (same folds for all models)...")
    comparison = {}
    for name, factory in MODELS.items():
        comparison[name] = cross_validation(dedup, n_splits, factory())
        summary = ', '.join(
            f"{metric} {comparison[name][metric]['mean']:.4f}"
            f"±{comparison[name][metric]['std']:.4f}"
            for metric in ('accuracy', 'precision_macro', 'recall_macro',
                           'f1_macro'))
        print(f"  {name}: {summary}")

    selected, factory, notes = select_model(comparison, dedup, n_splits)
    for note in notes:
        print(f"Note: {note}")
    print(f"Selected model: {selected}")
    cv_result = (comparison[selected] if selected in comparison
                 else cross_validation(dedup, n_splits, factory()))

    print(f"\nStratified 80/20 hold-out on deduplicated data ({selected})...")
    holdout_result = holdout(dedup, factory())
    print(holdout_result)

    print(f"\nTuning LogisticRegression with nested CV (outer {n_splits}-fold, "
          f"inner {INNER_FOLDS}-fold GridSearchCV, "
          f"{len(sklearn.model_selection.ParameterGrid(TUNING_GRID))} "
          f"combinations)...")
    nested = nested_cv_tuning(dedup, n_splits)
    for metric in ('accuracy', 'precision_macro', 'recall_macro',
                   'f1_macro'):
        print(f"  {metric}: {nested[metric]['mean']:.4f} "
              f"± {nested[metric]['std']:.4f}")
    for i, params in enumerate(nested['best_params_per_outer_fold'], 1):
        print(f"  best params outer fold {i}: {params}")

    untuned_f1 = comparison['LogisticRegression']['f1_macro']['mean']
    tuned_f1 = nested['f1_macro']['mean']
    use_tuned = tuned_f1 > untuned_f1

    final_search = _grid_search(n_splits)
    final_search.fit(dedup['cleaned_text'], dedup['Category'])
    final_params = _jsonable_params(final_search.best_params_)
    print(f"Final GridSearchCV ({n_splits}-fold, all {rows_after} resumes) "
          f"best params: {final_params}")
    if use_tuned:
        decision = (f"Nested-CV macro F1 {tuned_f1:.4f} > untuned "
                    f"LogisticRegression {untuned_f1:.4f}: the tuned model "
                    f"is saved.")
        deployed = 'LogisticRegression (tuned)'
    else:
        decision = (f"Nested-CV macro F1 {tuned_f1:.4f} is not higher than "
                    f"untuned LogisticRegression {untuned_f1:.4f}: the "
                    f"untuned {selected} model is kept.")
        deployed = selected
    print(decision)

    metrics = {
        'dataset': {
            'rows_before_dedup': rows_before,
            'rows_after_dedup': rows_after,
            'duplicates_removed': rows_before - rows_after,
            'categories': n_categories,
            'min_unique_resumes_per_category': int(class_counts.min()),
        },
        'model_comparison': {
            'cv': f'stratified {n_splits}-fold, shuffle=True, '
                  f'random_state={RANDOM_STATE}, deduplicated data, '
                  'same folds for all models',
            'features': TFIDF_SETTINGS,
            'results': comparison,
        },
        'selected_model': {
            'name': selected,
            'criterion': 'best mean macro F1 in model_comparison '
                         '(model must provide probabilities)',
            'note': 'Selected on the same CV folds that are reported, so '
                    'its CV score is slightly optimistic.',
            'selection_notes': notes,
        },
        'cross_validation_dedup': {'model': selected, **cv_result},
        'holdout_80_20_dedup': {'model': selected, **holdout_result},
        'original_method_no_dedup_for_comparison': {
            'model': ORIGINAL_MODEL, **original},
        'tuned_model': {
            'model': 'TF-IDF + LogisticRegression(max_iter=1000)',
            'search_space': {key: [list(v) if isinstance(v, tuple) else v
                                   for v in values]
                             for key, values in TUNING_GRID.items()},
            'fixed': {'vectorizer__max_features': 10000,
                      'vectorizer__max_df': 0.85},
            'nested_cv': {
                'outer': f'stratified {n_splits}-fold, shuffle=True, '
                         f'random_state={RANDOM_STATE} (same folds as '
                         'model_comparison)',
                'inner': f'GridSearchCV, stratified {INNER_FOLDS}-fold, '
                         'scoring=f1_macro',
                **nested,
            },
            'note': 'These scores come from nested CV: the parameters were '
                    'chosen only on the outer training folds, so they are '
                    'an honest estimate for the tuning procedure.',
            'final_params': final_params,
            'final_grid_search': f'GridSearchCV, stratified {n_splits}-fold '
                                 f'on all {rows_after} resumes',
            'final_grid_search_best_cv_f1_macro_optimistic': round(
                float(final_search.best_score_), 4),
            'untuned_logistic_regression_f1_macro': untuned_f1,
            'saved': use_tuned,
            'decision': decision,
        },
        'deployed_model': deployed,
        'model': (f'TF-IDF + LogisticRegression {final_params}' if use_tuned
                  else f'{TFIDF_SETTINGS} + {selected}'),
        'random_state': RANDOM_STATE,
        'scikit_learn_version': sklearn.__version__,
    }
    METRICS_PATH.parent.mkdir(exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + '\n')
    print(f"\nMetrics written to {METRICS_PATH.relative_to(BASE_DIR)}")

    if use_tuned:
        # refit=True: best pipeline already fitted on all deduplicated data
        final_pipeline = final_search.best_estimator_
    else:
        final_pipeline = build_pipeline(factory()).fit(dedup['cleaned_text'],
                                                       dedup['Category'])
    MODEL_PATH.parent.mkdir(exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(final_pipeline, f)
    print(f"Final model ({deployed}) trained on {rows_after} resumes saved "
          f"to {MODEL_PATH.relative_to(BASE_DIR)}")


if __name__ == '__main__':
    main()
