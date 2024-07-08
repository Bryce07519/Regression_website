from flask import Flask, request, render_template_string, redirect, url_for
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib
import matplotlib.pyplot as plt
import io
import base64

# Set the matplotlib backend to 'Agg' to avoid GUI issues
matplotlib.use('Agg')

app = Flask(__name__)

# Example DataFrame
data = pd.DataFrame({
    'date': pd.date_range(start='2020-01-01', periods=100, freq='D'),
    'y': np.random.randn(100),
    'x1': np.random.randn(100),
    'x2': np.random.randn(100),
    'x3': np.random.randn(100)
})
data.set_index('date', inplace=True)

# HTML templates as strings
index_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Regression Analysis Tool</title>
    <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://code.jquery.com/jquery-3.5.1.min.js"></script>
    <script>
        $(document).ready(function() {
            $('#add-filters').change(function() {
                if ($(this).is(':checked')) {
                    $('#filters-row').show();
                } else {
                    $('#filters-row').hide();
                }
            });
        });
    </script>
</head>
<body>
    <div class="container mt-5">
        <h1 class="mb-4">Regression Analysis Tool</h1>
        <form action="/result" method="POST" id="regression-form">
            <div class="form-group">
                <label for="dependent">Dependent Variable:</label>
                <select id="dependent" name="dependent" class="form-control">
                    <option value="y">y</option>
                    <option value="x1">x1</option>
                    <option value="x2">x2</option>
                    <option value="x3">x3</option>
                </select>
            </div>
            <div class="form-group">
                <label for="independents">Independent Variables:</label>
                <select id="independents" name="independents[]" class="form-control" multiple>
                    <option value="x1">x1</option>
                    <option value="x2">x2</option>
                    <option value="x3">x3</option>
                </select>
            </div>
            <div class="form-group">
                <label for="start_date">Start Date:</label>
                <input type="date" id="start_date" name="start_date" class="form-control" required>
            </div>
            <div class="form-group">
                <label for="end_date">End Date:</label>
                <input type="date" id="end_date" name="end_date" class="form-control" required>
            </div>
            <div class="form-group form-check">
                <input type="checkbox" class="form-check-input" id="add-filters" name="add_filters">
                <label class="form-check-label" for="add-filters">Add Filters</label>
            </div>
            <div id="filters-row" style="display: none;">
                <div class="form-group">
                    <label for="p_stat">P stat:</label>
                    <input type="text" id="p_stat" name="p_stat" class="form-control">
                </div>
                <div class="form-group">
                    <label for="error_std">Error Std:</label>
                    <input type="text" id="error_std" name="error_std" class="form-control">
                </div>
                <div class="form-group">
                    <label for="max_error">Max Error:</label>
                    <input type="text" id="max_error" name="max_error" class="form-control">
                </div>
                <div class="form-group">
                    <label for="adf_pstat">ADF Pstats:</label>
                    <input type="text" id="adf_pstat" name="adf_pstat" class="form-control">
                </div>
            </div>
            <button type="submit" class="btn btn-primary">Run Regression</button>
        </form>
    </div>
</body>
</html>
"""

result_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Regression Results</title>
    <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
    <div class="container mt-5">
        <h1 class="mb-4">Regression Results</h1>
        <div>
            {{ summary | safe }}
        </div>
        <h2 class="mt-4">Residuals Plot</h2>
        <img src="data:image/png;base64,{{ residuals_plot_data }}" alt="Residuals Plot" class="img-fluid">
        <h2 class="mt-4">Actual vs Predicted</h2>
        <img src="data:image/png;base64,{{ predictions_plot_data }}" alt="Predictions Plot" class="img-fluid">
        <a href="{{ url_for('index') }}" class="btn btn-primary mt-4">Back to Form</a>
    </div>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(index_html)

@app.route('/result', methods=['POST'])
def result():
    # Extract form data
    dep_var = request.form.get('dependent')
    ind_vars = request.form.getlist('independents[]')
    start_date = request.form.get('start_date')
    end_date = request.form.get('end_date')
    
    # Get filter parameters
    add_filters = 'add_filters' in request.form
    p_stat = request.form.get('p_stat')
    error_std = request.form.get('error_std')
    max_error = request.form.get('max_error')
    adf_pstat = request.form.get('adf_pstat')
    
    # Filter data based on selected time period
    filtered_data = data[start_date:end_date]

    # Prepare data for regression
    X = filtered_data[ind_vars]
    y = filtered_data[dep_var]
    X = sm.add_constant(X)  # Add constant term for intercept

    # Perform regression analysis
    model = sm.OLS(y, X).fit()
    predictions = model.predict(X)
    residuals = y - predictions

    # Generate summary statistics
    summary = model.summary().as_html()

    # Plot errors
    fig1, ax1 = plt.subplots()
    ax1.plot(filtered_data.index, residuals)
    ax1.set_title('Residuals over Time')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Residuals')
    buf1 = io.BytesIO()
    plt.savefig(buf1, format='png')
    buf1.seek(0)
    residuals_plot_data = base64.b64encode(buf1.getvalue()).decode('utf-8')
    buf1.close()

    # Plot predictions vs actual values
    fig2, ax2 = plt.subplots()
    ax2.plot(filtered_data.index, y, label='Actual')
    ax2.plot(filtered_data.index, predictions, label='Predicted')
    ax2.set_title('Actual vs Predicted')
    ax2.set_xlabel('Date')
    ax2.set_ylabel(dep_var)
    ax2.legend()
    buf2 = io.BytesIO()
    plt.savefig(buf2, format='png')
    buf2.seek(0)
    predictions_plot_data = base64.b64encode(buf2.getvalue()).decode('utf-8')
    buf2.close()
    
    return render_template_string(result_html, summary=summary, residuals_plot_data=residuals_plot_data, predictions_plot_data=predictions_plot_data)

if __name__ == '__main__':
    app.run(debug=True)
