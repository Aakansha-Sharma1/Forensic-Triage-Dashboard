# Exploratory ML Benchmark Protocol (AIT-LDS v2.1 — 4 Scenarios)

**Experiment ID:** `ait-lds-v2.1-4scenario-loso-exploratory-v1`  
**Machine-readable config:** `backend/ml/experiment_config.json`  
**Status:** Exploratory benchmark (source manifests retain `scientifically_adequate: false` and `selected_assignment: null`).

## 1. Inputs & Provenance

All supervised records are read from the four verified AIT-LDS v2.1 labelled datasets (`fox`, `harrison`, `russellmitchell`, `santos`) referenced by:

- Split manifest: `research_datasets/ait-lds-v2.1/splits/four_scenario_split_manifest.json` (`sha256`: `48c76c8964c303354e1351300661accc7248ddaba143a25cd865c7bc1b4b2694`)
- Quality report: `research_datasets/ait-lds-v2.1/quality/four_scenario_quality_report.json` (`sha256`: `ade042324b5640a0aca081e3c4b8463fb7dbe4e7de51f737feef790b9c983517`)

Before any training run, input manifest and JSONL SHA-256 hashes must match the fingerprints recorded in `experiment_config.json`. Existing dataset files and manifests are never modified.

## 2. Labels & Class Semantics

- **Positive class (`1`):** `attack_associated` (explicit verified ground truth).
- **Negative class (`0`):** `normal` (absence from a verified mirrored ground-truth file).
- **Excluded (`null`):** `unknown` (`617,876` events) — excluded from supervised training, validation, and testing; never relabelled as normal.
- **Normal-traffic false-alarm rate:** Defined unambiguously as **False-Positive Rate (`normal_false_positive_rate = FP / (FP + TN)`)**, i.e., the fraction of true `normal (0)` events incorrectly predicted as `attack_associated (1)`.

## 3. Features & Train-Only Preprocessing

Inputs are restricted strictly to the 13 numeric features defined in `FEATURE_SCHEMA` v`1.0.0` (`backend/ml/schema.py`). Event identifiers, timestamps, IPs, resources, and label fields are excluded from model inputs.

All preprocessing is fitted **exclusively on the training partition** and applied unchanged to validation and test partitions:

- **Imputation:** Median imputation fit on training data only (`fallback_value: 0.0`), paired with the existing missingness indicators.
- **Linear models (`LogisticRegression`):** `log1p` on the 6 non-negative count/volume/delta features followed by train-only `StandardScaler`.
- **Tree models (`DecisionTreeClassifier`, `RandomForestClassifier`, `DummyClassifier`):** Train-only median imputation; no feature scaling.
- **No synthetic resampling:** SMOTE, random oversampling, and random undersampling are prohibited.

## 4. Class-Imbalance Weighting Policy (No Double-Weighting)

Every supervised ML model (except `DummyClassifier` and `RuleBaseline_R_A_to_D`) must set estimator `class_weight=None` and apply weights **exclusively via `sample_weight`** during `fit()`. Passing both estimator `class_weight` and `sample_weight` is prohibited to prevent accidental double-weighting.

Both pre-registered weighting regimes normalize total sample weight over the training partition to `n_train` (mean weight `1.0`) so regularization strength (`C`) remains comparable across regimes:

1. **`class_weighted`:** `w_i = n_train / (2 * n_train_class(y_i))` computed on the pooled training partition only.
2. **`scenario_equalized_class_weighted`:** `w_i = n_train / (2 * n_train_scenarios * n_scenario_class(s_i, y_i))` so each training scenario receives equal total weight (`n_train / n_train_scenarios`), split 50/50 between `normal (0)` and `attack_associated (1)` within each scenario, preventing high-volume scenarios (`fox`, `harrison`) from dominating lower-volume scenarios (`russellmitchell`, `santos`).

## 5. Nested 4-Fold Leave-One-Scenario-Out (LOSO) Evaluation & Threshold Policy

- **Outer folds (4):** Each scenario (`fox`, `harrison`, `russellmitchell`, `santos`) is held out once as the final untouched test scenario.
- **Inner validation (3 candidates per outer fold; 12 total from `four_scenario_split_manifest.json`):** For each outer test scenario, the remaining 3 scenarios form 3 deterministic `(train[2], validation[1])` inner folds.
- **Joint hyperparameter & threshold selection:** For probabilistic ML models, `(hyperparameters, threshold)` pairs are selected **jointly** by mean inner-validation **Macro-F1** across the 3 inner folds and 5 fixed seeds (`[42, 137, 256, 512, 1024]`) over candidate thresholds `[0.1, 0.2, ..., 0.9]`. Deterministic tie-breaking prefers the threshold closest to `0.5`, then the smaller threshold, then the configured hyperparameter grid order. Outer test labels are never inspected during selection.
- **Separate fixed `0.5` and selected-threshold reporting:** For each probabilistic model, evaluate the refitted outer-train model on the held-out outer test scenario and report metrics **separately** at both the fixed `0.5` threshold and the inner-validation-selected threshold.
- **Probability calibration:** Predicted probabilities from `LogisticRegression` and tree models are **not** assumed calibrated; Brier score and calibration diagnostics must be reported before interpreting scores as probabilities.

## 6. Models, Rule Baseline & Runtime Dependencies

**Runtime dependencies (documented for experiment execution; not installed in this step):** `numpy`, `scikit-learn`.

- **`RuleBaseline_R_A_to_D` (non-probabilistic domain baseline):** Evaluates the existing `RULE_REGISTRY` v`1.0.0` (`R-A`, `R-B`, `R-C`, `R-D`; SHA-256 fingerprint `fa4a328f2d820be7ff7bc25efeaf5f5e04fe517fd7b7f38fcb2e85d6946c8447`) via a read-only canonical-event adapter aligned to each held-out scenario's verified labelled source file (`predicted = 1` if ≥1 rule triggers on the event, else `0`), without modifying `detect_threats()` or `RULE_REGISTRY`. Probability-dependent metrics (`pr_auc_attack`, `pr_auc_normal`, `brier_score`) are **N/A**. *Methodological caveats:* `R-C` emits at most one finding per `source_ip` per file (clearing `post_requests[source_ip]` after firing), and both `R-C` and `R-D` use full-file batch information (non-causal).
- **`DummyClassifier`:** `strategy` ∈ `{most_frequent, stratified}` (reference baseline; no weighting/threshold tuning).
- **`LogisticRegression`:** `C` ∈ `{0.01, 0.1, 1.0, 10.0}`, `penalty=l2`, `solver=lbfgs`, `max_iter=1000`, `class_weight=None`.
- **`DecisionTreeClassifier`:** `criterion=gini`, `max_depth` ∈ `{3, 5, 7}`, `min_samples_leaf` ∈ `{10, 50}`, `class_weight=None`.
- **`RandomForestClassifier`:** `n_estimators=100`, `criterion=gini`, `bootstrap=True`, `max_depth` ∈ `{5, 10}`, `min_samples_leaf` ∈ `{10, 50}`, `max_features=sqrt`, `class_weight=None`.

## 7. Metrics & Anti-Cherry-Picking Rules

- **Primary selection metric:** `macro_f1`.
- **Secondary metrics:** `balanced_accuracy`, `attack_precision`, `attack_recall`, `attack_f1`, `normal_precision`, `normal_recall`, `normal_f1`, `pr_auc_attack`, `pr_auc_normal`, `normal_false_positive_rate`, 2×2 `confusion_matrix`, `brier_score`, and `per_attack_category_recall`.
- **Explicit metric edge-case definitions:**
  - `pr_auc_attack`: Average Precision with `y_true_attack = (y == 1)` and `y_score_attack = p_attack`.
  - `pr_auc_normal`: Average Precision with `y_true_normal = (y == 0)` and `y_score_normal = 1.0 - p_attack`.
  - `zero_division = 0` for all precision, recall, and F1 calculations (required for constant-prediction baselines such as `DummyClassifier(strategy="most_frequent")`).
  - `normal_false_positive_rate = FP / (FP + TN)` with `attack_associated (1)` as positive and `normal (0)` as negative.
- **Reporting:** Report every held-out outer test scenario individually plus macro-mean, std, and min–max across the 4 outer folds and across all 5 seeds. Selecting the best outer fold or best random seed is prohibited.

## 8. Scientific Limitations

- All 4 scenarios come from one synthetic testbed (AIT-LDS v2.1); results are exploratory and do not establish external validity on production networks.
- Verified binary labels come from one intranet-server Apache log per scenario (`42.0%` unknown records excluded), confounding host and scenario effects.
- Severe, heterogeneous class imbalance (`4.67:1` to `197.42:1`, `83.84:1` overall) and high-volume `dirb` scans dominate raw event counts.
- Identical attack-category vocabularies and shared feature-vector patterns across scenarios limit independence claims.
