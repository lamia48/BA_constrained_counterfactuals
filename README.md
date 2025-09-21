# Bachelor's thesis

This repository provides the code for the user interface, presented for the bachelor's thesis "Interactive Constrained Counterfactuals: A Model-Based Comparison of Counterfactual Explanation Methods". 

## Starting the UI
The UI can be started by installing all required packages and starting the application with streamlit: 
```
git clone https://github.com/lamia48/BA_constrained_counterfactuals.git
pip install -r requirements.txt
streamlit run app.py
```
## Using the UI
The UI is divided into two pages, which can be selected by the drop-down menu "Select page".

### Counterfactuals
The first page is an interactive workflow for generating counterfactuals. The user can choose which instance a counterfactual should be generated for, which method should be used and the whether the method should focus on a specific pair of metrics, such as sparsity & proximity or generate an approximate trade-off. Additionally, a custom amount of features can be chosen which are constrained, and therefore immutable for the method. By clicking on "Generate counterfactual", a counterfactual is created using the selected configuration. Once the generation process is finished, the counterfactual is presented with additional metrics.

### Statistics
The second page visualises pre-generated statistics for further analysis of the counterfactual generation methods and constraints. The first three subpages visualise the results of success rate, confindence, proximity and sparsity with respect to different hyperparameter configurations which can be interactively changed with sliders for CEM, CFProto and CFProto + k-d trees. The fourth subpage visualises statistics for different feature constraint scenarios, where the user can interactively choose the subset of constrained features. 
