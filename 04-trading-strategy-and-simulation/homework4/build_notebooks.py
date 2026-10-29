"""Build HW4 notebooks (Q1, Q2, Q3) from a fresh copy of the 2025 Module 04 workbook."""
import copy
import json
from pathlib import Path

HERE = Path(__file__).parent
SRC = HERE.parent / '[2025]_Module_04_Colab_Trading_Simulations.ipynb'
base = json.load(open(SRC, encoding='utf-8'))


def md(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': text}


def code(text):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': text}


def src(cell):
    return ''.join(cell['source'])


def find(cells, snippet):
    idx = [i for i, c in enumerate(cells) if snippet in src(c)]
    assert len(idx) == 1, (snippet, idx)
    return idx[0]


def replace_in(cells, snippet, old, new):
    i = find(cells, snippet)
    s = src(cells[i])
    assert old in s, (snippet, old)
    cells[i]['source'] = s.replace(old, new)
    return i


def fresh():
    nb = copy.deepcopy(base)
    for c in nb['cells']:
        if c['cell_type'] == 'code':
            c['outputs'] = []
            c['execution_count'] = None
    cells = nb['cells']

    # Environment: works on Colab (/content/) and locally (./data/)
    replace_in(cells, '!pip install yfinance', '!pip install yfinance', '%pip install -q yfinance')
    replace_in(cells, '!pip uninstall gdown', '!pip uninstall gdown -y && pip install gdown\n!gdown -V',
               '%pip install -q gdown')
    i = find(cells, '!gdown https://drive.google.com')
    cells[i]['source'] = (
        "# https://stackoverflow.com/questions/62759748/downloading-data-from-a-shared-google-drive-link-in-google-colab\n"
        "# HW4: works both on Colab (/content/) and locally (../data/)\n"
        "import os\n"
        "import gdown\n"
        "\n"
        "DATA_DIR = '/content/' if os.path.isdir('/content') else '../data/'\n"
        "os.makedirs(DATA_DIR, exist_ok=True)\n"
        "DATA_FILE = os.path.join(DATA_DIR, 'stocks_df_combined_2025_06_13.parquet.brotli')\n"
        "if not os.path.exists(DATA_FILE):\n"
        "  gdown.download(id='1mb0ae2M5AouSDlqcUnIwaHq7avwGNrmB', output=DATA_FILE)\n"
    )
    replace_in(cells, 'df_full = pd.read_parquet(', '"/content/stocks_df_combined_2025_06_13.parquet.brotli"', 'DATA_FILE')
    return nb


def insert_after(cells, snippet, new_cells):
    i = find(cells, snippet)
    cells[i + 1:i + 1] = new_cells


def save(nb, name):
    nb['metadata']['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
    with open(HERE / name, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print('saved', name)


# ---------------------------------------------------------------- Q1
nb = fresh()
c = nb['cells']
c.insert(0, md(
    "# HW4 — Question 1: Less Features is More\n"
    "Fresh copy of `[2025]_Module_04_Colab_Trading_Simulations.ipynb`.\n\n"
    "**Change:** remove `Ticker` and `month_wom` from `CATEGORICAL` just before `pd.get_dummies()` "
    "(301 → 208 features).\n\n"
    "**Question:** CAGR for `pred9_clf_second_best_rule_92` with fewer dummies (original = 1.155577)."
))
insert_after(c, "CATEGORICAL.append('month_wom')", [
    md("### HW4 Q1: remove `Ticker` and `month_wom` from CATEGORICAL just before generating dummies"),
    code("# HW4 Q1: less dummies -> less features\n"
         "CATEGORICAL.remove('Ticker')\n"
         "CATEGORICAL.remove('month_wom')\n"
         "CATEGORICAL"),
])
insert_after(c, "df_with_dummies[NUMERICAL+DUMMIES].info()", [
    code("# HW4 Q1 check: should be 208 features (was 301)\n"
         "len(NUMERICAL+DUMMIES)"),
])
c += [
    md("# HW4 Q1 Answer"),
    code("# HW4 Q1: CAGR for pred9_clf_second_best_rule_92 (original = 1.155577)\n"
         "q1 = df_sim1_results[df_sim1_results.prediction=='pred9_clf_second_best_rule_92']\n"
         "print(f\"Features used: {len(NUMERICAL+DUMMIES)}\")\n"
         "print(f\"CAGR pred9_clf_second_best_rule_92 = {q1.sim1_CAGR.values[0]:.6f} (original 1.155577)\")\n"
         "q1"),
]
save(nb, 'HW4_Q1_less_features.ipynb')

# ---------------------------------------------------------------- Q2
nb = fresh()
c = nb['cells']
c.insert(0, md(
    "# HW4 — Question 2: Best CAGR for Random Forest Decision Rule\n"
    "Fresh copy of `[2025]_Module_04_Colab_Trading_Simulations.ipynb`.\n\n"
    "**Changes:**\n"
    "1. Comment out the `StandardScaler` in preprocessing\n"
    "2. `rf_best_max_depth = 14`\n"
    "3. Build `tpr_fpr_dataframe` for `rf_best` instead of `clf_best`\n"
    "4. New predictors `pred12_rf_best_rule_56` … `pred20_rf_best_rule_64`\n"
    "5. Find the rule with CAGR ≥ 1.1 and ~5 avg daily investments in `df_sim1_results`"
))
replace_in(c, 'rf_best_max_depth = 19', 'rf_best_max_depth = 19',
           'rf_best_max_depth = 14 # HW4 Q2: reduced from 19 to 14')
replace_in(c, 'scaler = StandardScaler()',
           "scaler = StandardScaler()\nX_train_valid = scaler.fit_transform(X_train_valid)\n"
           "X_valid = scaler.transform(X_valid)\nX_test = scaler.transform(X_test)",
           "# HW4 Q2: scaler is not needed for Random Forest (it biases its output)\n"
           "# scaler = StandardScaler()\n# X_train_valid = scaler.fit_transform(X_train_valid)\n"
           "# X_valid = scaler.transform(X_valid)\n# X_test = scaler.transform(X_test)")
replace_in(c, 'y_pred_test = clf_best.predict_proba(X_test)',
           "y_pred_test = clf_best.predict_proba(X_test)\n\n# y_pred_test = rf_best.predict_proba(X_test)",
           "# HW4 Q2: tpr_fpr_dataframe for the Random Forest\n"
           "# y_pred_test = clf_best.predict_proba(X_test)\n\ny_pred_test = rf_best.predict_proba(X_test)")
replace_in(c, "plt.title(f'The distribution of predictions for the current second best model",
           "plt.title(f'The distribution of predictions for the current second best model (Decision Tree with max_depth={clf_best.get_depth()})')",
           "plt.title(f'HW4 Q2: The distribution of predictions for Random Forest (max_depth={rf_best_max_depth})')")
insert_after(c, "df_scores[(df_scores.threshold>=0.6) & (df_scores.threshold<=0.92)]", [
    code("# HW4 Q2: Random Forest df_scores around the 0.55..0.65 thresholds\n"
         "# higher threshold -> higher precision, lower recall, fewer trades (tp+fp)\n"
         "df_scores_rf = tpr_fpr_dataframe(y_test, np.array(y_pred_test_class1), only_even=False)\n"
         "df_scores_rf['total_trades'] = df_scores_rf.tp + df_scores_rf.fp\n"
         "df_scores_rf[(df_scores_rf.threshold>=0.545) & (df_scores_rf.threshold<=0.655)]"),
])
insert_after(c, "new_df['pred11_rf_best_rule_65'] = (y_pred_all_class1_array >= 0.65).astype(int)", [
    code("# HW4 Q2: new Random Forest decision rules for thresholds 0.56 .. 0.64\n"
         "for i, yy in enumerate(range(56, 65)):\n"
         "  xx = 12 + i\n"
         "  new_df[f'pred{xx}_rf_best_rule_{yy}'] = (y_pred_all_class1_array >= yy/100).astype(int)\n"
         "\n"
         "[k for k in new_df.keys() if k.startswith('pred') and '_rf_best_rule_' in k]"),
])
c += [
    md("# HW4 Q2 Answer"),
    code("# HW4 Q2: all Random Forest decision rules, sorted by threshold\n"
         "q2 = df_sim1_results[df_sim1_results.prediction.str.contains('_rf_best_rule_')].copy()\n"
         "q2['threshold'] = q2.prediction.apply(lambda p: int(p.split('_')[-1])/100)\n"
         "q2 = q2.sort_values('threshold')\n"
         "q2[['prediction','threshold','sim1_count_investments','sim1_net_rev','sim1_avg_investments_per_day','sim1_capital','sim1_CAGR']]"),
    code("# HW4 Q2: rules with CAGR >= 1.1 -> pick the one with ~5 avg daily investments\n"
         "q2_good = q2[q2.sim1_CAGR>=1.1]\n"
         "best = q2_good.iloc[(q2_good.sim1_avg_investments_per_day-5).abs().argsort()].iloc[0]\n"
         "print(f\"Best RF rule: {best.prediction}, threshold = {best.threshold}, \"\n"
         "      f\"CAGR = {best.sim1_CAGR:.4f}, avg investments/day = {best.sim1_avg_investments_per_day:.2f}\")\n"
         "q2_good[['prediction','threshold','sim1_avg_investments_per_day','sim1_CAGR']]"),
]
save(nb, 'HW4_Q2_random_forest_thresholds.ipynb')

# ---------------------------------------------------------------- Q3
nb = fresh()
c = nb['cells']
c.insert(0, md(
    "# HW4 — Question 3: Predicting Strong Future Growth\n"
    "Fresh copy of `[2025]_Module_04_Colab_Trading_Simulations.ipynb`.\n\n"
    "**Change:** redefine `is_positive_growth_30d_future = 1 if growth_future_30d >= 1.08 else 0` "
    "(~27% of class 1 instead of ~60%).\n\n"
    "**Question:** CAGR for the best predictor after redefining the target (original best = 1.1556)."
))
insert_after(c, "df_full.info()", [
    md("### HW4 Q3: redefine the target — strong future growth (>= 8% in 30 days)"),
    code("# HW4 Q3: distribution BEFORE redefinition\n"
         "print(df_full[df_full.Date>='2000-01-01'][['growth_future_30d','is_positive_growth_30d_future']].describe().T)"),
    code("# HW4 Q3: new definition of the target\n"
         "df_full['is_positive_growth_30d_future'] = (df_full['growth_future_30d'] >= 1.08).astype(int)\n"
         "\n"
         "# share of class 1 after redefinition (should be ~27%)\n"
         "df_full[df_full.Date>='2000-01-01']['is_positive_growth_30d_future'].mean()"),
])
c += [
    md("# HW4 Q3 Answer"),
    code("# HW4 Q3: predictors sorted by CAGR\n"
         "q3 = df_sim1_results.sort_values('sim1_CAGR', ascending=False)\n"
         "best = q3.iloc[0]\n"
         "print(f\"Best predictor: {best.prediction}, CAGR = {best.sim1_CAGR:.4f} (original best 1.1556), \"\n"
         "      f\"avg investments/day = {best.sim1_avg_investments_per_day:.2f} (original best ~14.17)\")\n"
         "q3[['prediction','sim1_count_investments','sim1_net_rev','sim1_avg_investments_per_day','sim1_capital','sim1_CAGR']]"),
]
save(nb, 'HW4_Q3_strong_growth_target.ipynb')
