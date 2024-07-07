import datetime
import numpy as np
import pandas as pd
import hvplot.pandas
import statsmodels.api as sm
import panel as pn

pn.extension()

# Sample time series dataset (dummy data)
np.random.seed(0)
dates = pd.date_range('2020-01-01', periods=100, freq='D')
data = pd.DataFrame({
   'date': dates,
   'JXX1': np.random.randn(100).cumsum(),
   'JXX2': np.random.randn(100).cumsum(),
   'BXX1': np.random.randn(100).cumsum(),
   'BXX2': np.random.randn(100).cumsum()
})

# Function to perform regression and plot errors
def perform_regression(dependent_var, predictors, start_date, end_date, frequency, string_param):
    df = data.set_index('date')
    df = df.loc[start_date:end_date]

    y = df[dependent_var]
    X = df[predictors]
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()
    predictions = model.predict(X)

    residuals = y - predictions

    # Prepare results
    results = {
        'R-squared': model.rsquared,
        'Coefficients': model.params,
        'P-values': model.pvalues,
        'Residuals': residuals,
        'Summary': model.summary().as_html()
    }

    # Plot errors
    error_plot = residuals.hvplot(title="Residuals Plot", ylabel="Residuals")

    return results, error_plot

# Widgets for user inputs
dependent_var = pn.widgets.Select(name='Dependent Variable', options=['JXX1', 'JXX2', 'BXX1', 'BXX2'], sizing_mode='stretch_width')
start_date = pn.widgets.DatePicker(name='Start Date', value=datetime.date(2020, 1, 1), sizing_mode='stretch_width')
end_date = pn.widgets.DatePicker(name='End Date', value=datetime.date(2020, 4, 9), sizing_mode='stretch_width')

# Independent variable specific parameters
independent_vars = []

def add_independent_variable(event=None):
    new_variable = {
        'predictors': pn.widgets.MultiChoice(name='Predictors', options=['JXX1', 'JXX2', 'BXX1', 'BXX2'], sizing_mode='stretch_width'),
        'frequency': pn.widgets.Select(name='Frequency', options=['Daily', 'Weekly', 'Monthly'], sizing_mode='stretch_width'),
        'string_param': pn.widgets.TextInput(name='String Parameter', sizing_mode='stretch_width'),
        'delete_button': pn.widgets.Button(name='Delete', button_type='danger', sizing_mode='stretch_width')
    }
    new_variable['delete_button'].on_click(lambda event, var=new_variable: delete_independent_variable(var))
    independent_vars.append(new_variable)
    update_input_page()

def delete_independent_variable(var):
    if var in independent_vars:
        independent_vars.remove(var)
    update_input_page()

def update_input_page():
    global input_page, run_button

    # Input widgets
    input_widgets = [
        pn.pane.Markdown("## Time Series Regression Input", sizing_mode='stretch_width'),
        pn.Row(
            dependent_var,
            start_date,
            end_date
        )
    ]

    for var in independent_vars:
        input_widgets.append(
            pn.Column(
                var['predictors'],
                var['frequency'],
                var['string_param'],
                var['delete_button'],
                sizing_mode='stretch_width'
            )
        )

    input_widgets.append(
        pn.widgets.Button(name="Add Independent Variable", button_type="primary", width=300, sizing_mode='fixed', margin=(10, 10, 10, 0), align="start", click=add_independent_variable)
    )

    run_button = pn.widgets.Button(name="Run Regression", button_type="primary", width=300, sizing_mode='fixed')
    run_button.on_click(run_regression)

    input_widgets.append(run_button)

    input_page = pn.Column(*input_widgets, sizing_mode='stretch_width')

def run_regression(event):
    global output_stats, output_figure

    try:
        # Perform regression for each independent variable
        all_results = {}
        all_plots = []
        for var in independent_vars:
            predictors = var['predictors'].value
            frequency = var['frequency'].value.lower()
            string_param = var['string_param'].value

            results, plot = perform_regression(dependent_var.value, predictors, start_date.value, end_date.value,
                                               frequency, string_param)
            all_results[var['predictors'].value[0]] = results['Summary']
            all_plots.append(plot)

        # Update outputs
        output_stats.object = pn.Column(*[pn.pane.HTML(result) for result in all_results.values()])
        output_figure.object = pn.Row(*all_plots, sizing_mode='stretch_both')

        # Switch to the results tab
        tabs.active = 1
    except Exception as e:
        output_stats.object = f"Error: {str(e)}"
        output_figure.object = None

update_input_page()

# Result widgets
output_stats = pn.pane.HTML(sizing_mode='stretch_width')
output_figure = pn.pane.HoloViews(sizing_mode='stretch_both')

# Tabs for navigation
tabs = pn.Tabs(('Input', input_page), ('Results', pn.Column(
    pn.pane.Markdown("## Time Series Regression Results", sizing_mode='stretch_width'),
    output_stats,
    output_figure,
    sizing_mode='stretch_both'
)))

tabs.servable()
