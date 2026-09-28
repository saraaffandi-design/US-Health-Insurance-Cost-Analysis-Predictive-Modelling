import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="US Health Insurance Dashboard",
    page_icon="🏥",
    layout="wide",
)

# Semantic colours: orange = smoker, blue = non-smoker, grey-blue = everything else
SMOKER_COLOR = "#F26B38"
NONSMOKER_COLOR = "#4C9BE8"
NEUTRAL_COLOR = "#8FA3BF"
SMOKER_COLORS = {"Smoker": SMOKER_COLOR, "Non-smoker": NONSMOKER_COLOR}

RAW_COLUMNS = ["age", "sex", "bmi", "children", "smoker", "region", "charges"]


# ---------------------------------------------------------------------------
# Data & analysis (cached so nothing is recomputed on every widget click)
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    data = pd.read_csv("insurance.csv")
    data["smoker_label"] = data["smoker"].map({"yes": "Smoker", "no": "Non-smoker"})
    data["bmi_category"] = pd.cut(
        data["bmi"],
        bins=[0, 18.5, 25, 30, np.inf],
        labels=["Underweight", "Normal", "Overweight", "Obese"],
        right=False,  # BMI 30.0 counts as Obese
    )
    return data


def fmt_p(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


@st.cache_data
def compute_findings(data):
    """Every number shown in the Key Findings section is computed here from
    the full dataset, not typed in by hand."""
    # 1) Smoking: Welch's t-test on average charges
    means = data.groupby("smoker")["charges"].mean()
    _, p_smoker = stats.ttest_ind(
        data.loc[data["smoker"] == "yes", "charges"],
        data.loc[data["smoker"] == "no", "charges"],
        equal_var=False,
    )

    # 2) Region: one-way ANOVA + eta-squared (effect size)
    groups = [g["charges"].values for _, g in data.groupby("region")]
    _, p_region = stats.f_oneway(*groups)
    grand_mean = data["charges"].mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total = ((data["charges"] - grand_mean) ** 2).sum()

    # 3) BMI x smoking interaction: OLS with mean-centred BMI, adjusted for
    #    age, sex, children and region (same model as the notebook)
    d = pd.get_dummies(
        data[RAW_COLUMNS], columns=["sex", "smoker", "region"], drop_first=True
    ).astype(float)
    d["bmi_c"] = d["bmi"] - d["bmi"].mean()
    d["bmi_x_smoker"] = d["bmi_c"] * d["smoker_yes"]
    cols = [
        "age", "bmi_c", "children", "sex_male", "smoker_yes",
        "region_northwest", "region_southeast", "region_southwest", "bmi_x_smoker",
    ]
    X = np.column_stack([np.ones(len(d)), d[cols].values])
    y = d["charges"].values
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = X.shape[0] - X.shape[1]
    sigma2 = resid @ resid / dof
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
    i = cols.index("bmi_x_smoker") + 1  # +1 for the intercept
    p_interaction = 2 * stats.t.sf(abs(beta[i] / se[i]), dof)

    # Fitted lines: vary BMI, hold every other covariate at its average
    grid = np.linspace(d["bmi"].min(), d["bmi"].max(), 50)
    lines = {}
    for flag, label in [(0, "Non-smoker"), (1, "Smoker")]:
        g = pd.DataFrame({c: d[c].mean() for c in cols}, index=range(len(grid)))
        g["bmi_c"] = grid - d["bmi"].mean()
        g["smoker_yes"] = flag
        g["bmi_x_smoker"] = g["bmi_c"] * flag
        lines[label] = np.column_stack([np.ones(len(g)), g[cols].values]) @ beta

    # Obese (BMI >= 30) vs not, split by smoking status
    obese_means = (
        data.assign(obese=data["bmi"] >= 30)
        .groupby(["smoker", "obese"])["charges"]
        .mean()
    )

    return {
        "ratio": means["yes"] / means["no"],
        "mean_smoker": means["yes"],
        "mean_nonsmoker": means["no"],
        "p_smoker": p_smoker,
        "p_region": p_region,
        "eta2": ss_between / ss_total,
        "p_interaction": p_interaction,
        "interaction_coef": beta[i],
        "grid": grid,
        "lines": lines,
        "obese_means": obese_means,
    }


def encode_row(age, bmi, children, sex, smoker, region):
    """Encode ONE customer exactly like the training data (drop_first=True).
    pd.get_dummies() on a single row can't be used: it only sees one category
    per column and would silently encode every selection as the reference level."""
    return {
        "age": age,
        "bmi": bmi,
        "children": children,
        "sex_male": 1 if sex == "male" else 0,
        "smoker_yes": 1 if smoker == "yes" else 0,
        "region_northwest": 1 if region == "northwest" else 0,
        "region_southeast": 1 if region == "southeast" else 0,
        "region_southwest": 1 if region == "southwest" else 0,
    }


@st.cache_resource
def train_model(data):
    """Trained once on the FULL dataset (80/20 split for evaluation), so the
    sidebar filters never change the model."""
    mdf = pd.get_dummies(
        data[RAW_COLUMNS], columns=["sex", "smoker", "region"], drop_first=True
    )
    X = mdf.drop(columns=["charges"])
    y = mdf["charges"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    rf = RandomForestRegressor(n_estimators=300, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)

    pred = rf.predict(X_test)
    errors = y_test.values - pred
    is_smoker = X_test["smoker_yes"].values.astype(bool)

    # 80% error band per smoking group (errors are larger for smokers)
    error_band = {
        flag: np.quantile(errors[is_smoker == flag], [0.10, 0.90])
        for flag in (True, False)
    }

    # Importance: add the three region dummies into one "region" bar
    def feature_group(col):
        if col.startswith("region_"):
            return "region"
        return {"smoker_yes": "smoker", "sex_male": "sex"}.get(col, col)

    importance = (
        pd.Series(rf.feature_importances_, index=X.columns)
        .groupby(feature_group)
        .sum()
        .sort_values()
    )

    return {
        "model": rf,
        "columns": X.columns.tolist(),
        "r2": r2_score(y_test, pred),
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "mae": mean_absolute_error(y_test, pred),
        "y_test": y_test.values,
        "pred": pred,
        "is_smoker": is_smoker,
        "error_band": error_band,
        "importance": importance,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


df = load_data()
findings = compute_findings(df)
bundle = train_model(df)

# ---------------------------------------------------------------------------
# Sidebar filters (these only affect the Overview section)
# ---------------------------------------------------------------------------
st.sidebar.title("Dashboard Filters")
st.sidebar.caption(
    "Filters apply to the **Overview** section only. Key findings and the "
    "prediction model always use the full dataset."
)

selected_smoker = st.sidebar.multiselect(
    "Smoking Status", options=df["smoker"].unique(), default=df["smoker"].unique()
)
selected_region = st.sidebar.multiselect(
    "Region", options=df["region"].unique(), default=df["region"].unique()
)
selected_sex = st.sidebar.multiselect(
    "Sex", options=df["sex"].unique(), default=df["sex"].unique()
)

filtered_df = df[
    df["smoker"].isin(selected_smoker)
    & df["region"].isin(selected_region)
    & df["sex"].isin(selected_sex)
]

# ---------------------------------------------------------------------------
# Header + Overview
# ---------------------------------------------------------------------------
st.title("US Health Insurance Analytics")
st.markdown(
    "What drives individual health insurance charges, how strong is the "
    "statistical evidence, and how well can charges be predicted?"
)

st.header("Overview")

if filtered_df.empty:
    st.warning("No customers match the selected filters. Adjust the sidebar filters.")
else:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Customers", f"{len(filtered_df):,}")
    k2.metric("Average Insurance Charge", f"${filtered_df['charges'].mean():,.0f}")
    k3.metric("Average BMI", f"{filtered_df['bmi'].mean():.1f}")
    k4.metric("Smoker Rate", f"{(filtered_df['smoker'] == 'yes').mean():.1%}")

    left, right = st.columns(2)

    with left:
        smoker_summary = filtered_df.groupby("smoker_label", as_index=False)[
            "charges"
        ].mean()
        fig = px.bar(
            smoker_summary,
            x="smoker_label",
            y="charges",
            color="smoker_label",
            color_discrete_map=SMOKER_COLORS,
            title="Average Insurance Charges by Smoking Status",
            labels={"smoker_label": "", "charges": "Average Charges ($)"},
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.info(
            f"🚬 **Smoking status** (full dataset): smokers' average charges are "
            f"**{findings['ratio']:.1f}×** those of non-smokers "
            f"(${findings['mean_smoker']:,.0f} vs ${findings['mean_nonsmoker']:,.0f}); "
            f"Welch's t-test {fmt_p(findings['p_smoker'])}."
        )

    with right:
        region_summary = (
            filtered_df.groupby("region", as_index=False)["charges"]
            .mean()
            .sort_values("charges", ascending=False)
        )
        fig = px.bar(
            region_summary,
            x="region",
            y="charges",
            title="Average Insurance Charges by Region",
            labels={"region": "", "charges": "Average Charges ($)"},
            color_discrete_sequence=[NEUTRAL_COLOR],
        )
        st.plotly_chart(fig, use_container_width=True)
        st.info(
            f"📍 **Regional differences** (full dataset): statistically significant "
            f"(ANOVA {fmt_p(findings['p_region'])}), but region explains only "
            f"**{findings['eta2'] * 100:.2f}%** of the variance in charges "
            f"(η² = {findings['eta2']:.4f}) — a small effect in practice."
        )

# ---------------------------------------------------------------------------
# Key findings (full dataset)
# ---------------------------------------------------------------------------
st.header("Key Findings")
st.caption("Computed on the full dataset (1,338 customers); not affected by the sidebar filters.")

f1, f2, f3 = st.columns(3)
with f1:
    with st.container(border=True):
        st.markdown("#### 🚬 Smoking")
        st.metric("Higher average charges for smokers", f"{findings['ratio']:.1f}×")
        st.caption(f"Welch's t-test: {fmt_p(findings['p_smoker'])}")
with f2:
    with st.container(border=True):
        st.markdown("#### ⚖️ BMI × Smoking")
        st.metric(
            "Extra cost per BMI point for smokers",
            f"+${findings['interaction_coef']:,.0f}",
        )
        st.caption(
            f"Interaction term: {fmt_p(findings['p_interaction'])} "
            "(adjusted for age, sex, children, region)"
        )
with f3:
    with st.container(border=True):
        st.markdown("#### 📍 Region")
        st.metric("ANOVA", fmt_p(findings["p_region"]))
        st.caption(
            f"Significant but small: η² = {findings['eta2']:.4f} "
            f"(≈{findings['eta2'] * 100:.1f}% of variance)"
        )

# ---------------------------------------------------------------------------
# BMI x smoking
# ---------------------------------------------------------------------------
st.header("BMI × Smoking Relationship")

b_left, b_right = st.columns([3, 2])

with b_left:
    fig = px.scatter(
        df,
        x="bmi",
        y="charges",
        color="smoker_label",
        color_discrete_map=SMOKER_COLORS,
        opacity=0.25,
        title="Estimated Relationship Between BMI and Charges",
        labels={"bmi": "BMI", "charges": "Insurance Charges ($)", "smoker_label": ""},
    )
    for label, y_line in findings["lines"].items():
        fig.add_trace(
            go.Scatter(
                x=findings["grid"],
                y=y_line,
                mode="lines",
                name=f"{label} (model)",
                line=dict(color=SMOKER_COLORS[label], width=4),
            )
        )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Points are actual customers; lines are the fitted interaction model with "
        "all other variables held at their average."
    )

with b_right:
    cat_summary = df.groupby(["bmi_category", "smoker_label"], observed=True).agg(
        charges=("charges", "mean"), customers=("charges", "size")
    ).reset_index()
    fig = px.bar(
        cat_summary,
        x="bmi_category",
        y="charges",
        color="smoker_label",
        barmode="group",
        color_discrete_map=SMOKER_COLORS,
        hover_data={"customers": True},
        title="Average Charges by BMI Category",
        labels={"bmi_category": "", "charges": "Average Charges ($)", "smoker_label": ""},
    )
    st.plotly_chart(fig, use_container_width=True)
    om = findings["obese_means"]
    st.caption(
        f"Smokers with BMI ≥ 30 average ${om[('yes', True)]:,.0f} versus "
        f"${om[('yes', False)]:,.0f} for smokers below 30. For non-smokers the "
        f"gap is small (${om[('no', True)]:,.0f} vs ${om[('no', False)]:,.0f})."
    )

# ---------------------------------------------------------------------------
# Predictive modelling
# ---------------------------------------------------------------------------
st.header("Predictive Modelling")
st.caption(
    f"Random Forest trained on {bundle['n_train']:,} customers and evaluated on "
    f"{bundle['n_test']:,} held-out customers."
)

m1, m2, m3 = st.columns(3)
m1.metric("R²", f"{bundle['r2']:.3f}")
m2.metric("RMSE", f"${bundle['rmse']:,.0f}")
m3.metric("MAE", f"${bundle['mae']:,.0f}")

p_left, p_right = st.columns(2)

with p_left:
    avp = pd.DataFrame(
        {
            "Actual": bundle["y_test"],
            "Predicted": bundle["pred"],
            "Group": np.where(bundle["is_smoker"], "Smoker", "Non-smoker"),
        }
    )
    fig = px.scatter(
        avp,
        x="Actual",
        y="Predicted",
        color="Group",
        color_discrete_map=SMOKER_COLORS,
        opacity=0.6,
        title="Actual vs Predicted Charges (test set)",
        labels={"Actual": "Actual charges ($)", "Predicted": "Predicted charges ($)", "Group": ""},
    )
    top = float(max(avp["Actual"].max(), avp["Predicted"].max()))
    fig.add_trace(
        go.Scatter(
            x=[0, top],
            y=[0, top],
            mode="lines",
            name="Perfect prediction",
            line=dict(color="grey", dash="dash"),
        )
    )
    st.plotly_chart(fig, use_container_width=True)

with p_right:
    imp = bundle["importance"]
    fig = px.bar(
        x=imp.values,
        y=imp.index,
        orientation="h",
        title="Random Forest Feature Importance",
        labels={"x": "Importance", "y": ""},
        color_discrete_sequence=[NEUTRAL_COLOR],
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Interactive prediction
# ---------------------------------------------------------------------------
st.header("Interactive Charge Prediction")

c1, c2, c3 = st.columns(3)
with c1:
    age = st.number_input("Age", min_value=18, max_value=64, value=35)
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=30.0)
with c2:
    children = st.number_input("Children", min_value=0, max_value=5, value=0)
    sex = st.selectbox("Sex", ["female", "male"])
with c3:
    smoker = st.selectbox("Smoker", ["no", "yes"])
    region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])


def predict_charge(smoker_value):
    row = encode_row(age, bmi, children, sex, smoker_value, region)
    x_new = pd.DataFrame([row]).reindex(columns=bundle["columns"], fill_value=0)
    return float(bundle["model"].predict(x_new)[0])


if st.button("Predict Insurance Charge"):
    prediction = predict_charge(smoker)
    low_err, high_err = bundle["error_band"][smoker == "yes"]
    low = max(prediction + low_err, 0)
    high = prediction + high_err

    st.success(f"Estimated insurance charge: ${prediction:,.0f}")
    st.caption(
        f"Typical range: ${low:,.0f} – ${high:,.0f} (the middle 80% of test-set "
        f"errors for {'smokers' if smoker == 'yes' else 'non-smokers'})."
    )

    other = "no" if smoker == "yes" else "yes"
    other_prediction = predict_charge(other)
    gap = abs(prediction - other_prediction)
    if smoker == "yes":
        st.info(
            f"As a non-smoker, the same profile would be estimated at "
            f"${other_prediction:,.0f} — about ${gap:,.0f} less."
        )
    else:
        st.info(
            f"As a smoker, the same profile would be estimated at "
            f"${other_prediction:,.0f} — about ${gap:,.0f} more."
        )

# ---------------------------------------------------------------------------
# Methodology & limitations
# ---------------------------------------------------------------------------
with st.expander("Methodology & limitations"):
    st.markdown(
        f"""
- **Data:** 1,338 customers (age, sex, BMI, children, smoker, region, charges).
- **Tests:** Welch's t-test (smoker vs non-smoker), one-way ANOVA with η² (region),
  and an OLS interaction model with mean-centred BMI, adjusted for age, sex,
  children and region.
- **Model:** Random Forest (300 trees, max depth 5), 80/20 train/test split.
  Test-set R² = {bundle['r2']:.3f}. The notebook also reports 5-fold
  cross-validation, which is a more stable estimate than a single split.
- **Association, not causation:** this is observational data. The results show
  strong associations with charges, not proof that any factor causes them.
- **Prediction ranges** come from test-set errors ({bundle['n_test']} customers, of
  which only a minority are smokers). They describe typical error for that
  smoking group, not the exact uncertainty of the profile entered, and the smoker
  range is rougher than the non-smoker one.
- **Simplifications:** linear BMI lines are an approximation; the data suggests a
  step around BMI 30 for smokers. Importance scores are impurity-based and
  should be read as rough rankings.
"""
    )
