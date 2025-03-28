from flask import Flask, render_template, jsonify, request

from src.exception import CustomException
from src.pipelines.prediction_pipeline import PredictPipeline
from src.components.data_ingestion import DataIngestion

application = Flask(__name__)

app = application

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('form.html')
    else:
        try:
            # Validate input data and log the values
            carat = request.form.get('carat')
            depth = request.form.get('depth')
            table = request.form.get('table')
            x = request.form.get('x')
            y = request.form.get('y')
            z = request.form.get('z')
            cut = request.form.get('cut')
            color = request.form.get('color')
            clarity = request.form.get('clarity')

            print(f"Input values - Carat: {carat}, Depth: {depth}, Table: {table}, x: {x}, y: {y}, z: {z}, Cut: {cut}, Color: {color}, Clarity: {clarity}")

            # Check for empty fields and log if any are missing
            if not all([carat, depth, table, x, y, z, cut, color, clarity]):
                print("One or more fields are empty.")
                return render_template('form.html', error="Please fill in all fields.")

            # Test with hardcoded values for prediction
            # Test with hardcoded values for prediction
            data = DataIngestion(
                carat=1.0,  # Hardcoded value for testing
                depth=60.0,  # Hardcoded value for testing
                table=55.0,  # Hardcoded value for testing
                x=5.0,  # Hardcoded value for testing
                y=5.0,  # Hardcoded value for testing
                z=3.0,  # Hardcoded value for testing
                cut=cut,
                color=color,
                clarity=clarity
            )


            print("Data ingestion successful.")

            final_new_data = data.get_data_as_dataframe()
            print("Data converted to DataFrame.")

            predict_pipeline = PredictPipeline()
            pred = predict_pipeline.predict(final_new_data)
            print(f"Prediction result: {pred}")

            results = round(pred[0], 2)

            return render_template('results.html', final_result=results)

        except Exception as e:
            print(f"Error occurred: {str(e)}")
            return render_template('form.html', error=str(e))

if __name__ == "__main__":
    app.run(debug=True, port=5000, host='0.0.0.0')
