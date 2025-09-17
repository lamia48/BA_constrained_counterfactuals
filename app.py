import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils.counterfactuals import *
from utils.model import *
from utils.statistics import *
import altair as alt

@st.cache_data
def load_data(path):
    return pd.read_csv(path)


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

    if "selected_columns" not in st.session_state:
        st.session_state.selected_columns = X_test.columns.tolist()
    selected_columns = st.sidebar.multiselect("Select features to be constrained", X_test.columns.tolist(), default=st.session_state.selected_columns)
    st.session_state.selected_columns = selected_columns

    st.subheader('Selected instance:')
    st.data_editor(X_test.iloc[[selected_index]], use_container_width=True)

    #st.dataframe(df[selected_columns])
    if st.sidebar.button('Generate counterfactual'):
        if selected_method == 'None':
            st.error('Select method for generating counterfactual')
        else:
            unselected_columns = [col for col in X_test.columns if col not in selected_columns]
            unselected_indices = [X_test.columns.get_loc(col) for col in unselected_columns]
            print(unselected_columns)

            if selected_method == 'CEM':
                df = perform_cem(selected_index, unselected_indices)
            elif selected_method == 'CFProto':
                df = perform_cfproto(selected_index, unselected_indices)
            else:
                df = perform_cfproto(selected_index, unselected_indices)

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
                    st.metric("sparsity", sparsity, "low is better")
                with col2:
                    st.metric("proximity (l1-norm)", proximity, "low is better")
                with col3:
                    st.metric("confidence", df['cf_prob'], "high is better")


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
        df = load_data('data/cfproto_kdt_kbt.csv')








