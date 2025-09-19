from utils.counterfactuals import *
from utils.model import *
from utils.statistics import *
import altair as alt
import glob

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

@st.cache_data
def load_constrained_data():
    files = glob.glob("data/constraints/cfproto_constraint_on_*.csv")
    data = {}
    for f in files:
        key = f.split("cfproto_constraint_on_")[1].replace(".csv", "")
        data[key] = pd.read_csv(f)
    return data

st.set_page_config(page_title="Constrained Interactive Counterfactuals", layout="wide")
app_mode = st.sidebar.selectbox('Select page', ['Counterfactuals', 'Statistics'])

train_model()
model, scaler, data = load_model()
feature_names = data['feature_names']
X_test = data['X_test']
X_test = pd.DataFrame(X_test, columns=data['feature_names'])

if app_mode == 'Counterfactuals':
    st.title('Constrained Interactive Counterfactuals')
    selected_index = st.sidebar.number_input('Select an instance', 0, 114, format="%i")
    selected_method = st.sidebar.selectbox('Select a method', ['None', 'CEM', 'CFProto', 'CFProto with k-d trees'])
    selected_config_maxim = st.sidebar.selectbox('Select preferred metrics', ['None', 'sparsity & proximity', 'proximity & success rate', 'success rate', 'trade-off'])

    if "selected_columns" not in st.session_state:
        st.session_state.selected_columns = X_test.columns.tolist()
    selected_columns = st.sidebar.multiselect("Select features to be constrained", X_test.columns.tolist(), default=st.session_state.selected_columns)
    st.session_state.selected_columns = selected_columns

    st.subheader('Selected instance:')
    st.data_editor(X_test.iloc[[selected_index]], use_container_width=True)

    #st.dataframe(df[selected_columns])
    if st.sidebar.button('Generate counterfactual'):
        if selected_method == 'None':
            st.error('Select method for generating counterfactual!')
        elif selected_config_maxim == 'None':
            st.error('Select configuration maxim!')
        else:
            unselected_columns = [col for col in X_test.columns if col not in selected_columns]
            unselected_indices = [X_test.columns.get_loc(col) for col in unselected_columns]

            if selected_method == 'CEM':
                if selected_config_maxim == 'sparsity & proximity':
                    df = perform_cem(selected_index, unselected_indices, 0.0, 0.05)
                elif selected_config_maxim == 'proximity & success rate':
                    df = perform_cem(selected_index, unselected_indices, 0.0, 0.0)
                elif selected_config_maxim == 'success rate':
                    df = perform_cem(selected_index, unselected_indices, 0.7, 0.0)
                elif selected_config_maxim == 'trade-off':
                    df = perform_cem(selected_index, unselected_indices, 0.2, 0.02)

            elif selected_method == 'CFProto':
                if selected_config_maxim == 'sparsity & proximity':
                    df = perform_cfproto(selected_index, unselected_indices, 0.0, 0.05)
                elif selected_config_maxim == 'proximity & success rate':
                    df = perform_cfproto(selected_index, unselected_indices, 0.0, 0.0)
                elif selected_config_maxim == 'success rate':
                    df = perform_cfproto(selected_index, unselected_indices, 0.7, 0.0)
                elif selected_config_maxim == 'trade-off':
                    df = perform_cfproto(selected_index, unselected_indices, 0.3, 0.02)

            else:
                if selected_config_maxim == 'sparsity & proximity':
                    df = perform_cfproto_kdtrees(selected_index, unselected_indices, 0.0, 0.05, 0.1)
                elif selected_config_maxim == 'proximity & success rate':
                    df = perform_cfproto_kdtrees(selected_index, unselected_indices, 0.0, 0.0, 0.1)
                elif selected_config_maxim == 'success rate':
                    df = perform_cfproto_kdtrees(selected_index, unselected_indices, 0.3, 0.0, 0.5)
                elif selected_config_maxim == 'trade-off':
                    df = perform_cfproto_kdtrees(selected_index, unselected_indices, 0.0, 0.03, 0.1)

            if df['success'].iloc[0] == 0:
                st.error('No counterfactual found.')
            else:
                diff_columns = [col for col in df.columns if col.startswith('diff_')]
                df[diff_columns] = df[diff_columns].applymap(lambda x: 0 if abs(x) <= 0.001 else x)

                st.subheader('Counterfactual:')
                st.dataframe(df[diff_columns])

                plot_df = df[diff_columns].T.reset_index()
                plot_df.columns = ["feature", "value"]
                plot_df["feature"] = plot_df["feature"].str.replace("^diff_", "", regex=True)
                chart = alt.Chart(plot_df).mark_bar().encode(
                    x=alt.X("feature", sort=plot_df["feature"].tolist()),  # enforce given order
                    y="value"
                )

                st.altair_chart(chart, use_container_width=True)

                st.subheader("Evaluation:")

                proximity, sparsity = compute_stats(df[diff_columns].copy())

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("sparsity", sparsity)
                with col2:
                    st.metric("proximity", proximity)
                with col3:
                    st.metric("confidence", df['cf_prob'])


if app_mode == 'Statistics':
    st.title('Statistics')

    tab1, tab2, tab3, tab4 = st.tabs(["CEM", "CFProto", "CFProto with k-d trees", "Constraints on CFProto"])
    with tab1:
        df = load_data('data/cem_kappabeta_komplett.csv')
        print_table(df, 'kappa', 'beta', 'cem')
    with tab2:
        df = load_data('data/cfproto_kappabeta_komplett.csv')
        print_table(df, 'kappa', 'beta', 'cfproto')

    with tab3:
        df = load_data('data/cfproto_kdt_kbt.csv')
        print_table_3sliders(df, 'kappa', 'beta', 'theta', 'cfproto-kdt')

    with tab4:
        data = load_constrained_data()

        col1, col2 = st.columns(2)
        with col1:
            constraint_type = st.radio("Constraint type", ["top", "bottom"])

        with col2:
            num_constraints = st.select_slider('number of constraints', options=[5,10,15,20,25])

        st.divider()

        key = f"{constraint_type}{num_constraints}"
        if key in data:
            df = data[key]
            #diff_cols = [col for col in df.columns if col.startswith('diff_')]
            #columns = ['success'] + diff_cols
            #renamed = df[columns].rename(columns=lambda x: x.replace("diff_", "") if x.startswith("diff_") else x)
            #st.write(f"Showing results for **{key}**")
            #st.dataframe(renamed)

            diff_cols = [col for col in df.columns if col.startswith("diff_")]
            feature_names = [c.replace("diff_", "") for c in diff_cols]


            successes = df[df['success'] == 1]
            sub_abs = successes[diff_cols].abs()

            success_rate = df["success"].mean()
            sparsity = sparsity = (sub_abs > 0.015).mean()
            proximity = sub_abs.where(lambda x: x > 0.015).mean()


            results = pd.DataFrame({
                "feature": feature_names,
                "sparsity": sparsity.values,
                "proximity": proximity.values,
            })
            results["success_rate"] = success_rate * 100

            # Scale success_rate for plotting on same axis as proximity
            results["success_rate_scaled"] = results["success_rate"] / 100 * results["proximity"].max()

            st.subheader("Sparsity and proximity per feature:")

            x = alt.X("feature:N", sort=list(results["feature"]), axis=alt.Axis(labelAngle=-85))

            bar = alt.Chart(results).mark_bar(color="steelblue").encode(
                x=x,
                y=alt.Y("sparsity:Q", axis=alt.Axis(title="sparsity (proportion of instances)", orient="left"))
            )
            line_prox = alt.Chart(results).mark_line(point=True, color="orange").encode(
                x=x,
                y=alt.Y("proximity:Q", axis=alt.Axis(title="proximity (average per feature)", orient="right")),
                tooltip=["feature", "proximity"]
            )

            chart = alt.layer(bar, line_prox).resolve_scale(y="independent")
            st.altair_chart(chart.properties(height=420), use_container_width=True)

            st.divider()

            st.subheader("Overall statistics:")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("sparsity", sparsity.mean())
            with col2:
                st.metric("proximity", proximity.mean())
            with col3:
                st.metric("success rate", success_rate)

        else:
            st.warning("No data available for this selection.")

