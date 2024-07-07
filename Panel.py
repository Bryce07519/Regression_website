import panel as pn
import hvplot.pandas  # Import hvPlot for Pandas
import pandas as pd
import numpy as np

pn.extension()

def calculate_and_plot(x, y):
    # Perform some calculations
    z = x * y

    # Create a dataframe for plotting
    t = np.linspace(0, 10, 500)
    df = pd.DataFrame({'t': t, 'value': np.sin(t * z)})
    
    # Create a plot using hvPlot
    plot = df.hvplot.line(x='t', y='value', title='Sine Wave')

    return z, plot

def update(event):
    try:
        x = float(input_x.value)
        y = float(input_y.value)
        z, plot = calculate_and_plot(x, y)
        output_value.value = f"Calculated value: {z}"
        output_figure.object = plot
    except Exception as e:
        output_value.value = f"Error: {str(e)}"
        output_figure.object = None

# Define input widgets
input_x = pn.widgets.TextInput(name='Input X', value='1.0')
input_y = pn.widgets.TextInput(name='Input Y', value='1.0')

# Define output widgets
output_value = pn.widgets.StaticText(name='Output Value')
output_figure = pn.pane.HoloViews()

# Button to trigger calculation
button = pn.widgets.Button(name='Calculate', button_type='primary')
button.on_click(update)

# Layout
layout = pn.Column(
    pn.Row(input_x, input_y),
    button,
    output_value,
    output_figure
)

# Serve the app
layout.servable()
