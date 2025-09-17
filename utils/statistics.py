import streamlit as st
import pandas as pd
# success rate?


def compute_stats(df):
    diff_columns = [col for col in df.columns if col.startswith('diff_')]

    sparsity = (df[diff_columns].abs().gt(0.015).sum(axis=1).mean())
    if sparsity > 0:
        proximity = df[diff_columns].abs().sum(axis=1) / sparsity
    else:
        proximity = 0
    return proximity, sparsity

def compute_statistics(df, diff_columns, tolerance=0.015):
    stats = {}

    for cls in [0, 1]:
        df_cls = df[df["original_pred"] == cls]
        if df_cls.empty:
            stats[f"class {cls}"] = {
                "success rate": 0,
                "confidence": 0,
                "proximity": 0,
                "sparsity": 0,
            }
            continue

        # --- success rate ---
        success_rate = df_cls["success"].mean()

        # --- only successful CFs ---
        df_success = df_cls[df_cls["success"] == 1]
        if not df_success.empty:
            avg_change = df_success[diff_columns].abs().where(lambda x: x > tolerance).mean().mean()
            avg_cf_prob = df_success["cf_prob"].mean()
            avg_sparsity = df_success[diff_columns].abs().gt(tolerance).sum(axis=1).mean()
        else:
            avg_change = avg_cf_prob = avg_sparsity = 0

        stats[f"class {cls}"] = {
            "success rate": success_rate,
            "confidence": avg_cf_prob,
            "proximity": avg_change,
            "sparsity": avg_sparsity,
        }

    # --- overall (on all rows, but metrics conditional on success) ---
    if not df.empty:
        success_rate = df["success"].mean()
        df_success = df[df["success"] == 1]
        if not df_success.empty:
            avg_change = df_success[diff_columns].abs().where(lambda x: x > tolerance).mean().mean()
            avg_cf_prob = df_success["cf_prob"].mean()
            avg_sparsity = df_success[diff_columns].abs().gt(tolerance).sum(axis=1).mean()
        else:
            avg_change = avg_cf_prob = avg_sparsity = 0

        stats["Σ"] = {
            "success rate": success_rate,
            "confidence": avg_cf_prob,
            "proximity": avg_change,
            "sparsity": avg_sparsity,
        }

    # convert to DataFrame (metrics as rows, classes as columns)
    statistics = pd.DataFrame(stats)

    return statistics




def print_table(df, val1, val2, methodname):
    diff_columns = [c for c in df.columns if c.startswith("diff_")]

    col1, col2 = st.columns([1, 2])

    col1_name = val1
    col2_name = val2

    with col1:
        val1_val = st.slider(col1_name,
                             float(df[col1_name].min()),
                             float(df[col1_name].max()),
                             float(df[col1_name].min()),
                             step=0.1,
                             key=f"{methodname}_{col1_name}_slider")

        options = sorted(df['beta'].unique())
        val2_val = st.select_slider(
            "beta",
            options=options,
            value=options[0],
            key=f"{methodname}_beta_slider"
        )

    with col2:
        df_filtered = df[(df[col1_name] == val1_val) & (df[col2_name] == val2_val)]
        if df_filtered.empty:
            st.warning(f"No data available for this {col1_name}/{col2_name} combination")
        else:
            statistics = compute_statistics(df_filtered, diff_columns)
            styled = statistics.style.apply(highlight, axis=None).format(precision=3)
            st.write(styled.to_html(), unsafe_allow_html=True)


def print_table_3sliders(df, val1, val2, val3, methodname):
    diff_columns = [c for c in df.columns if c.startswith("diff_")]

    col1, col2, col3 = st.columns([1, 1, 2])

    # Slider 1
    with col1:
        val1_val = st.slider(
            val1,
            float(df[val1].min()),
            float(df[val1].max()),
            float(df[val1].min()),
            step=0.1,
            key=f"{methodname}_{val1}_slider"
        )
        options2 = sorted(df[val2].unique())
        val2_val = st.select_slider(
            val2,
            options=options2,
            value=options2[0],
            key=f"{methodname}_{val2}_slider"
        )
        options3 = sorted(df[val3].unique())
        val3_val = st.select_slider(
            val3,
            options=options3,
            value=options3[0],
            key=f"{methodname}_{val3}_slider"
        )

    # Slider 2
    with col2:
        # Filter and display
        df_filtered = df[
            (df[val1] == val1_val) &
            (df[val2] == val2_val) &
            (df[val3] == val3_val)
        ]

        if df_filtered.empty:
            st.warning(f"No data available for this {val1}/{val2}/{val3} combination")
        else:
            statistics = compute_statistics(df_filtered, diff_columns)
            styled = statistics.style.apply(highlight, axis=None).format(precision=3)
            st.write(styled.to_html(), unsafe_allow_html=True)


def color_values(val, metric):
    if not isinstance(val, (int, float)):  # skip text cells
        return ""
    if metric == "success rate":
        if val > 0.75:
            return "background-color: lightgreen"
        elif val > 0.5:
            return "background-color: orange"
        else:
            return "background-color: red"
    elif metric == "confidence":
        if val > 0.75:
            return "background-color: lightgreen"
        elif val > 0.5:
            return "background-color: orange"
        else:
            return "background-color: red"
    elif metric == "proximity":
        if val < 0.25:
            return "background-color: lightgreen"
        elif val < 0.5:
            return "background-color: orange"
        else:
            return "background-color: red"
    elif metric == "sparsity":
        if val < 10:
            return "background-color: lightgreen"
        elif val < 15:
            return "background-color: orange"
        else:
            return "background-color: red"
    return ""


def highlight(df: pd.DataFrame):
    styles = pd.DataFrame("", index=df.index, columns=df.columns)
    for metric in df.index:
        for col in df.columns:
            styles.loc[metric, col] = color_values(df.loc[metric, col], metric)
    return styles