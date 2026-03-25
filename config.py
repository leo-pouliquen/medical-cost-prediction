import os

base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, "data", "medicalcost.csv")


expected_dtypes = {
    "age": "int64",
    "bmi": "float64",
    "children": "int64",
    "charges": "float64",
    "sex": "object",
    "smoker": "object",
    "region": "object"
}


