import datetime
import numpy as np
import pandas as pd
import hvplot.pandas
import statsmodels.api as sm
import panel as pn

pn.extension()

# Create a sample time series dataset
np.random.seed(0)
dates = pd.date_range('2020-01-01', periods=100, freq='D')
data = pd.DataFrame({
   'date': dates,
   'y': np.random.randn(100).cumsum(),
   'x1': np.random.randn(100).cumsum(),
   'x2': np.random.randn(100).cumsum()
})

# Function to perform regression and plot errors
def perform_regression(dependent_var, predictors, start_date, end_date):
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
dependent_var = pn.widgets.Select(name='Dependent Variable', options=['y'], sizing_mode='stretch_width')
predictors = pn.widgets.MultiChoice(name='Predictors', options=['x1', 'x2'], sizing_mode='stretch_width')
start_date = pn.widgets.DatePicker(name='Start Date', value=datetime.date(2020, 1, 1), sizing_mode='stretch_width')
end_date = pn.widgets.DatePicker(name='End Date', value=datetime.date(2020, 4, 9), sizing_mode='stretch_width')

# Output widgets
output_stats = pn.pane.HTML(sizing_mode='stretch_width')
output_figure = pn.pane.HoloViews(sizing_mode='stretch_both')

def update(event):
   try:
       results, plot = perform_regression(dependent_var.value, predictors.value, start_date.value, end_date.value)
       output_stats.object = results['Summary']
       output_figure.object = plot

       # Switch to the results tab
       tabs.active = 1
   except Exception as e:
       output_stats.object = f"Error: {str(e)}"
       output_figure.object = None

# Button to trigger regression analysis
button = pn.widgets.Button(name='Run Regression', button_type='primary', sizing_mode='stretch_width')
button.on_click(update)

# Input page
input_page = pn.Column(
   pn.pane.Markdown("## Time Series Regression Input", sizing_mode='stretch_width'),
   dependent_var,
   predictors,
   start_date,
   end_date,
   button,
   sizing_mode='fixed',
   width=300
)

# Result page
result_page = pn.Column(
   pn.pane.Markdown("## Time Series Regression Results", sizing_mode='stretch_width'),
   output_stats,
   pn.Row(output_figure, sizing_mode='stretch_both', align='center'),  # Align the figure horizontally centered
   sizing_mode='stretch_both'
)

# Tabs for navigation
tabs = pn.Tabs(('Input', input_page), ('Results', result_page))

# Custom CSS for figure positioning
custom_css = """
.result-figure {
   display: flex;
   justify-content: center;
}
"""
pn.config.raw_css.append(custom_css)

# Serve the app
tabs.servable()
