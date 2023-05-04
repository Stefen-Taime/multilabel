from flask import Flask, request, jsonify
from joblib import load
import pandas as pd

app = Flask(__name__)

# Load the saved LightGBM classifiers
lgbm_air = load("Air_lgbm_classifier.joblib")
lgbm_road = load("Road_lgbm_classifier.joblib")
lgbm_rail = load("Rail_lgbm_classifier.joblib")
lgbm_sea = load("Sea_lgbm_classifier.joblib")


@app.route("/predict", methods=["POST"])
def predict():
    # Get data from the request
    data = request.json

    # Convert the request data to a DataFrame
    df = pd.DataFrame(data, index=[0])

    # Make predictions
    y_pred_air = lgbm_air.predict(df)
    y_pred_road = lgbm_road.predict(df)
    y_pred_rail = lgbm_rail.predict(df)
    y_pred_sea = lgbm_sea.predict(df)

    # Return the predictions as a JSON response
    response = jsonify({
        "Air": int(y_pred_air[0]),
        "Road": int(y_pred_road[0]),
        "Rail": int(y_pred_rail[0]),
        "Sea": int(y_pred_sea[0])
    })

    response.headers.add("Access-Control-Allow-Origin", "*")
    return response



if __name__ == "__main__":
    app.run(port=5001)

