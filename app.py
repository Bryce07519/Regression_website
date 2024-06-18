from flask import Flask, request, render_template
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

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

@app.route('/')
def index():
    return render_template('index.html', columns=data.columns)

@app.route('/process', methods=['POST'])
def process():
    # Get parameters from the form
    dep_var = request.form['dependent']
    ind_vars = request.form.getlist('independents[]')
    start_date = request.form['start_date']
    end_date = request.form['end_date']
    
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
    summary = model.summary().as_text()

    # Plot errors
    fig, ax = plt.subplots()
    ax.plot(filtered_data.index, residuals)
    ax.set_title('Residuals over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Residuals')
    
    # Save figure to a BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    
    return render_template('result.html', summary=summary, img_data=img.getvalue().hex())

if __name__ == '__main__':
    app.run(debug=True)
