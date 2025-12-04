"""
Train demo models using synthetic data including lifestyle features only
to match compute_features() in app.py.
Output: model.pkl (scaler + classifiers + metrics + feature importance)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss, accuracy_score
from sklearn.inspection import permutation_importance
import joblib

RANDOM_STATE = 1

# FEATURES: match compute_features() in app.py (NO lab biomarkers)
FEATURES = [
    "age", "sex_male", "bmi", "sleep_hours", "sugar_drinks", "fruit_servings",
    "veg_servings", "steps", "stress_level", "systolic", "diastolic", "resting_hr",
    "existing_diabetes", "existing_cvd", "existing_cancer", "existing_asthma",
    "family_history_cancer", "family_history_cvd", "family_history_diabetes"
]


def make_synthetic(n=16000, seed=RANDOM_STATE):
    """Generate synthetic dataset with only lifestyle features."""
    rng = np.random.RandomState(seed)
    age = rng.randint(18, 85, size=n)
    sex_male = rng.randint(0, 2, size=n)
    bmi = np.clip(rng.normal(26, 6, size=n), 15, 50)

    sleep_hours = np.clip(rng.normal(7, 1.5, size=n), 2, 12)
    sugar_drinks = np.clip(rng.poisson(1, size=n), 0, 10)
    fruit_servings = np.clip(rng.poisson(1, size=n), 0, 8)
    veg_servings = np.clip(rng.poisson(2, size=n), 0, 10)
    steps = np.clip(rng.normal(6000, 3000, size=n).astype(int), 0, 40000)
    stress_level = np.clip(rng.normal(4, 2, size=n), 0, 10)

    systolic = np.clip((110 + (age - 30) * 0.5 + (bmi - 25) * 0.9 + rng.normal(0, 8, size=n)).astype(int), 90, 220)
    diastolic = np.clip((70 + (bmi - 25) * 0.4 + rng.normal(0, 6, size=n)).astype(int), 50, 140)
    resting_hr = np.clip((60 + (stress_level - 4) * 2 + rng.normal(0, 6, size=n)).astype(int), 40, 140)

    existing_diabetes = rng.binomial(1, 0.08, size=n)
    existing_cvd = rng.binomial(1, 0.06, size=n)
    existing_cancer = rng.binomial(1, 0.03, size=n)
    existing_asthma = rng.binomial(1, 0.07, size=n)

    family_history_cancer = rng.binomial(1, 0.12, size=n)
    family_history_cvd = rng.binomial(1, 0.10, size=n)
    family_history_diabetes = rng.binomial(1, 0.15, size=n)

    df = pd.DataFrame({
        "age": age, "sex_male": sex_male, "bmi": bmi,
        "sleep_hours": sleep_hours, "sugar_drinks": sugar_drinks,
        "fruit_servings": fruit_servings, "veg_servings": veg_servings,
        "steps": steps, "stress_level": stress_level,
        "systolic": systolic, "diastolic": diastolic, "resting_hr": resting_hr,
        "existing_diabetes": existing_diabetes, "existing_cvd": existing_cvd,
        "existing_cancer": existing_cancer, "existing_asthma": existing_asthma,
        "family_history_cancer": family_history_cancer,
        "family_history_cvd": family_history_cvd,
        "family_history_diabetes": family_history_diabetes
    })
    return df


def gen_labels(df):
    """Generate synthetic binary labels for Diabetes, CVD, and Cancer."""
    # Diabetes risk score
    diab_score = (
        0.035 * df.age + 0.2 * (df.bmi - 24) + 0.9 * df.sugar_drinks
        - 0.0002 * df.steps + 2.0 * df.existing_diabetes
        + 1.0 * df.family_history_diabetes
    )
    diab_prob = 1 / (1 + np.exp(-diab_score / 10))
    y_diab = (diab_prob + np.random.rand(len(df)) * 0.15) > 0.5

    # CVD risk score
    cvd_score = (
        0.04 * df.age + 0.08 * (df.systolic - 120) + 0.04 * (df.diastolic - 80)
        + 0.05 * (df.resting_hr - 70) - 0.0001 * df.steps
        + 1.8 * df.existing_cvd + 1.1 * df.family_history_cvd
    )
    cvd_prob = 1 / (1 + np.exp(-cvd_score / 10))
    y_cvd = (cvd_prob + np.random.rand(len(df)) * 0.12) > 0.5

    # Cancer risk score
    cancer_score = (
        0.02 * (df.age - 40) + 0.4 * (3 - df.veg_servings).clip(0)
        + 0.06 * (df.stress_level - 4) + 0.06 * (6 - df.sleep_hours).clip(0)
        + 1.8 * df.existing_cancer + 1.2 * df.family_history_cancer
    )
    cancer_prob = 1 / (1 + np.exp(-cancer_score / 10))
    y_cancer = (cancer_prob + np.random.rand(len(df)) * 0.2) > 0.6

    return y_diab.astype(int), y_cvd.astype(int), y_cancer.astype(int)


def train_and_save(path="model.pkl"):
    print("Generating synthetic data...")
    df = make_synthetic()
    y_diab, y_cvd, y_cancer = gen_labels(df)

    # Only use selected features
    X = df[FEATURES].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    diseases = {"Diabetes": y_diab, "CVD": y_cvd, "Cancer": y_cancer}
    model_bundle = {
        "meta": {"features": FEATURES, "note": "Synthetic demo models (lifestyle only)"},
        "scaler": scaler,
        "models": {}
    }

    for i, (disease, y) in enumerate(diseases.items()):
        print(f"\nTraining {disease}...")
        X_train, X_test, y_train, y_test = train_test_split(
            Xs, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
        )
        base = RandomForestClassifier(
            n_estimators=300, max_depth=8, random_state=RANDOM_STATE + i
        )
        clf = CalibratedClassifierCV(base, method="sigmoid", cv=5)
        clf.fit(X_train, y_train)

        y_proba = clf.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)
        auc = roc_auc_score(y_test, y_proba)
        brier = brier_score_loss(y_test, y_proba)
        acc = accuracy_score(y_test, y_pred)

        print("Computing permutation importance...")
        try:
            r = permutation_importance(
                clf, X_test, y_test, n_repeats=10,
                random_state=RANDOM_STATE, scoring="roc_auc"
            )
            perm_importances = {
                FEATURES[i]: float(r.importances_mean[i]) for i in range(len(FEATURES))
            }
            perm_importances = dict(
                sorted(perm_importances.items(), key=lambda kv: kv[1], reverse=True)
            )
        except Exception as e:
            print("Permutation importance error:", e)
            perm_importances = {}

        model_bundle["models"][disease] = {
            "classifier": clf,
            "metrics": {"roc_auc": float(auc), "brier": float(brier), "accuracy": float(acc)},
            "permutation_importance": perm_importances
        }
        print(f"{disease} done — AUC: {auc:.3f}, Brier: {brier:.4f}, Acc: {acc:.3f}")

    joblib.dump(model_bundle, path)
    print(f"\nSaved model bundle to {path}")


if __name__ == "__main__":
    train_and_save("model.pkl")
