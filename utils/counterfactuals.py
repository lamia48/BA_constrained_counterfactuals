import pandas as pd
from alibi.explainers import CEM, CounterfactualProto
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import numpy as np
from utils.model import *




def fix_features(feature_indices, X_train, instance):
    min_vals = X_train.min(axis=0).copy()
    max_vals = X_train.max(axis=0).copy()

    for i in feature_indices:
        min_vals[i] = 0.99 * instance[0, i]
        max_vals[i] = 1.01 * instance[0, i]
    return min_vals, max_vals


def perform_cem(datapoint_indice, constraints):
    model, scaler, data = load_model()
    X_train = data['X_train']
    X_test = data['X_test']
    feature_names = data['feature_names']

    instance = X_test[datapoint_indice].reshape(1, -1)
    preds = model.predict_proba(instance)

    original_pred = preds.argmax(axis=1)[0]
    original_prob = preds[0][original_pred]

    min_vals, max_vals = fix_features(constraints, X_train, instance)

    def predict(x: np.ndarray) -> np.ndarray:
        preds = model.predict_proba(x)
        return np.atleast_2d(preds)

    tf.keras.backend.clear_session()
    cem = CEM(
        predict,
        mode='PN',
        shape=instance.shape,
        kappa=0.2,
        beta=0.02,
        gamma=0.0,
        c_init=1.0,
        c_steps=10,
        max_iterations=1000,
        feature_range=(min_vals, max_vals),  # (X_train.min(axis=0), X_train.max(axis=0))
        clip=(min_vals, max_vals),  # (X_train.min(axis=0), X_train.max(axis=0))
        learning_rate_init=1e-2  # oder 1e-3 / 1e-4
    )
    cem.fit(instance)
    explanation = cem.explain(instance)
    counterfactual = explanation.PN if explanation.PN is not None else instance
    success = 1 if explanation.PN is not None else 0

    cf_predictions = model.predict_proba(counterfactual)
    cf_pred = cf_predictions.argmax(axis=1)[0]
    cf_prob = cf_predictions[0][cf_pred]

    row = {
        "success": success,
        "original_pred": original_pred,
        "original_prob": original_prob,
        "cf_pred": cf_pred,
        "cf_prob": cf_prob,
    }

    diff = (counterfactual - instance)[0]
    for j, name in enumerate(feature_names):
        row[f"og_{name}"] = X_test[datapoint_indice, j]
        row[f"cf_{name}"] = counterfactual[0, j]
        row[f"diff_{name}"] = diff[j]

    df = pd.DataFrame([row])
    return df


def perform_cfproto(datapoint_indice, constraints):
    model, scaler, data = load_model()
    X_train = data['X_train']
    X_test = data['X_test']
    feature_names = data['feature_names']

    instance = X_test[datapoint_indice].reshape(1, -1)
    preds = model.predict_proba(instance)

    original_pred = preds.argmax(axis=1)[0]
    original_prob = preds[0][original_pred]

    def predict(x: np.ndarray) -> np.ndarray:
        preds = model.predict_proba(x)
        return np.atleast_2d(preds)

    min_vals, max_vals = fix_features(constraints, X_train, instance)

    tf.keras.backend.clear_session()
    cf = CounterfactualProto(
        predict,
        shape=instance.shape,
        kappa=0.3,
        beta=0.02,
        gamma=0,
        theta=0,
        max_iterations=1000,
        ae_model=None,
        enc_model=None,
        feature_range=(min_vals, max_vals),
        clip=(min_vals, max_vals),
        c_init=1.,
        c_steps=10,  # vorher 5
        learning_rate_init=1e-2,
    )

    cf.fit(X_train, d_type='abdm', w=None, disc_perc=[25, 50, 75], standardize_cat_vars=False,
           smooth=1., center=True, update_feature_range=True)
    explanation = cf.explain(instance)

    data = explanation.data
    counterfactual = data["cf"]["X"] if data.get("cf") is not None else instance

    success = 1 if data.get("cf") is not None else 0

    cf_pred = predict(counterfactual).argmax(axis=1)[0]
    cf_prob = predict(counterfactual)[0][cf_pred]

    row = {
        "success": success,
        "original_pred": original_pred,
        "original_prob": original_prob,
        "cf_pred": cf_pred,
        "cf_prob": cf_prob,
    }

    diff = (counterfactual - instance)[0]
    for j, name in enumerate(feature_names):
        row[f"og_{name}"] = X_test[datapoint_indice, j]
        row[f"cf_{name}"] = counterfactual[0, j]
        row[f"diff_{name}"] = diff[j]

    df = pd.DataFrame([row])
    return df


def perform_cfproto_kdtrees(datapoint_indice, constraints):
    model, scaler, data = load_model()
    X_train = data['X_train']
    X_test = data['X_test']
    feature_names = data['feature_names']

    instance = X_test[datapoint_indice].reshape(1, -1)
    preds = model.predict_proba(instance)

    original_pred = preds.argmax(axis=1)[0]
    original_prob = preds[0][original_pred]

    def predict(x: np.ndarray) -> np.ndarray:
        preds = model.predict_proba(x)
        return np.atleast_2d(preds)

    min_vals, max_vals = fix_features(constraints, X_train, instance)

    tf.keras.backend.clear_session()
    cf = CounterfactualProto(
        predict,
        shape=instance.shape,
        use_kdtree=True,
        kappa=0.7,
        beta=0.01,
        gamma=0,
        theta=0,
        max_iterations=1000,
        ae_model=None,
        enc_model=None,
        feature_range=(min_vals, max_vals),
        clip=(min_vals, max_vals),
        c_init=1.,
        c_steps=10,  # vorher 5
        learning_rate_init=1e-2,
    )

    cf.fit(X_train, trustscore_kwargs=None)
    explanation = cf.explain(instance, k=2)

    data = explanation.data
    counterfactual = data["cf"]["X"] if data.get("cf") is not None else instance

    success = 1 if data.get("cf") is not None else 0

    cf_pred = predict(counterfactual).argmax(axis=1)[0]
    cf_prob = predict(counterfactual)[0][cf_pred]

    row = {
        "success": success,
        "original_pred": original_pred,
        "original_prob": original_prob,
        "cf_pred": cf_pred,
        "cf_prob": cf_prob,
    }

    diff = (counterfactual - instance)[0]
    for j, name in enumerate(feature_names):
        row[f"og_{name}"] = X_test[datapoint_indice, j]
        row[f"cf_{name}"] = counterfactual[0, j]
        row[f"diff_{name}"] = diff[j]

    df = pd.DataFrame([row])
    return df