import os
from flask import Flask, jsonify, render_template, request, redirect, url_for
import pandas as pd
import joblib
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import mysql.connector
from mysql.connector import Error
import numpy as np


app = Flask(__name__)

# Global variables
MODEL_DIR = 'static/models/'

def ensure_model_directory():
    os.makedirs(MODEL_DIR, exist_ok=True)

def get_database_connection():
    db_connection_str = 'mysql+mysqlconnector://root:@localhost/fahri'
    return create_engine(db_connection_str)

def load_data_from_db():
    print("Loading data from database")
    connection = get_database_connection()
    data_latih = pd.read_sql('SELECT * FROM data_latih', con=connection)
    data_uji = pd.read_sql('SELECT * FROM data_uji', con=connection)
    return data_latih, data_uji

def preprocess_data(data_latih, data_uji):
    print("Preprocessing data")
    le_dict = {}
    columns_to_encode = ['penghasilan', 'status_ekonomi', 'layak_pip', 'status_bantuan']

    # Replace missing or empty values with a placeholder like 'unknown'
    data_latih = data_latih.replace('', 'unknown')
    data_uji = data_uji.replace('', 'unknown')
    
    for column in columns_to_encode:
        print(f"Encoding column: {column}")
        le = LabelEncoder()
        
        # Fit LabelEncoder on data_latih
        data_latih[column] = le.fit_transform(data_latih[column].astype(str))
        
        # Handle unseen labels by adding 'unknown' to the classes if necessary
        le_classes = list(le.classes_)
        if 'unknown' not in le_classes:
            le_classes.append('unknown')
            le.classes_ = np.array(le_classes)
        
        # Transform data_uji with the updated classes
        data_uji[column] = data_uji[column].apply(lambda x: 'unknown' if x not in le.classes_ else x)
        data_uji[column] = le.transform(data_uji[column].astype(str))
        
        le_dict[column] = le

    # Encode the target variable for training
    print("Encoding target variable: status_kesesuaian")
    le_status_kesesuaian = LabelEncoder()
    data_latih['status_kesesuaian'] = le_status_kesesuaian.fit_transform(data_latih['status_kesesuaian'].astype(str))
    le_dict['status_kesesuaian'] = le_status_kesesuaian

    columns_to_drop = ['nama', 'alasan_layak_pip']
    X_latih = data_latih.drop(columns=columns_to_drop + ['status_kesesuaian'])
    y_latih = data_latih['status_kesesuaian']
    X_uji = data_uji.drop(columns=columns_to_drop)

    print("Data preprocessing completed")
    return X_latih, y_latih, X_uji, le_dict



@app.route('/train', methods=['GET'])
def train_route():
    ensure_model_directory()  # Ensure directory exists before saving models
    print("Starting training process")
    data_latih, data_uji = load_data_from_db()

    # Preprocess data
    X_latih, y_latih, X_uji, le_dict = preprocess_data(data_latih, data_uji)

    print("Scaling training data")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_latih)
    X_test_scaled = scaler.transform(X_uji)  # Scale the test data using the same scaler
    print("Scaling completed")

    # Perform hyperparameter tuning for SVM
    print("Starting SVM hyperparameter tuning")
    svm_param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }
    svm_grid_search = GridSearchCV(SVC(), svm_param_grid, cv=5, n_jobs=-1)
    svm_grid_search.fit(X_train_scaled, y_latih)
    svm_model = svm_grid_search.best_estimator_
    print(f"SVM best parameters: {svm_grid_search.best_params_}")

    # Perform hyperparameter tuning for AdaBoost
    print("Starting AdaBoost hyperparameter tuning")
    adaboost_param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1],
        'estimator__C': [0.1, 1, 10]
    }
    base_svm = SVC(kernel='linear', probability=True)
    adaboost_grid_search = GridSearchCV(AdaBoostClassifier(estimator=base_svm), 
                                        adaboost_param_grid, cv=5, n_jobs=-1)
    adaboost_grid_search.fit(X_train_scaled, y_latih)
    adaboost_model = adaboost_grid_search.best_estimator_
    print(f"AdaBoost best parameters: {adaboost_grid_search.best_params_}")

    # Save models and preprocessing objects
    print("Saving models and preprocessing objects")
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
    joblib.dump(svm_model, os.path.join(MODEL_DIR, 'svm_model.pkl'))
    joblib.dump(adaboost_model, os.path.join(MODEL_DIR, 'adaboost_svm_model.pkl'))
    joblib.dump(le_dict, os.path.join(MODEL_DIR, 'le_dict.pkl'))
    print("Models and preprocessing objects saved")

    return jsonify({
        "message": "Models trained and saved successfully",
        "svm_best_params": svm_grid_search.best_params_,
        "adaboost_best_params": adaboost_grid_search.best_params_
    })
@app.route('/predict', methods=['GET'])
def predict_route():
    try:
        ensure_model_directory()  # Ensure the directory exists

        # Load the saved models and preprocessing objects
        scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
        svm_model = joblib.load(os.path.join(MODEL_DIR, 'svm_model.pkl'))
        adaboost_model = joblib.load(os.path.join(MODEL_DIR, 'adaboost_svm_model.pkl'))
        le_dict = joblib.load(os.path.join(MODEL_DIR, 'le_dict.pkl'))

        # Load the data from the database
        data_latih, data_uji = load_data_from_db()

        if data_uji.empty:
            return jsonify({"error": "Data uji kosong"}), 400

        # Preprocess the data (only preprocess the test data)
        _, _, X_uji, _ = preprocess_data(data_latih, data_uji)

        if X_uji.empty:
            return jsonify({"error": "Data uji tidak dapat diproses"}), 400

        # Scale the test data
        X_uji_scaled = scaler.transform(X_uji)

        # Predict using the trained models
        svm_predictions = svm_model.predict(X_uji_scaled)
        adaboost_predictions = adaboost_model.predict(X_uji_scaled)

        # Inverse transform the predictions back to original labels
        svm_predictions = le_dict['status_kesesuaian'].inverse_transform(svm_predictions)
        adaboost_predictions = le_dict['status_kesesuaian'].inverse_transform(adaboost_predictions)

        # Prepare the results in a structured format
        results = []
        for index, row in data_uji.iterrows():
            result = {
                "nama": row['nama'],
                "status_bantuan": row['status_bantuan'],
                "svm_predicted_kesesuaian": svm_predictions[index],
                "adaboost_predicted_kesesuaian": adaboost_predictions[index]
            }
            results.append(result)

        # Ensure the response is always JSON
        return jsonify({"predictions": results}), 200

    except Exception as e:
        # If any error occurs, return the error as a JSON object
        return jsonify({"error": str(e)}), 500




@app.route('/evaluate', methods=['GET'])
def evaluate_route():
    if not all(os.path.exists(os.path.join(MODEL_DIR, f)) for f in ['svm_model.pkl', 'adaboost_svm_model.pkl', 'le_dict.pkl']):
        return "Models not trained or test data not loaded.", 400

    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    svm_model = joblib.load(os.path.join(MODEL_DIR, 'svm_model.pkl'))
    adaboost_model = joblib.load(os.path.join(MODEL_DIR, 'adaboost_svm_model.pkl'))
    le_dict = joblib.load(os.path.join(MODEL_DIR, 'le_dict.pkl'))

    data_latih, data_uji = load_data_from_db()
    X_latih, y_latih, _, _ = preprocess_data(data_latih, data_uji)

    X_latih_scaled = scaler.transform(X_latih)

    y_pred_svm = svm_model.predict(X_latih_scaled)
    y_pred_adaboost = adaboost_model.predict(X_latih_scaled)

    # Inverse transform predictions for evaluation
    y_pred_svm = le_dict['status_kesesuaian'].inverse_transform(y_pred_svm)
    y_pred_adaboost = le_dict['status_kesesuaian'].inverse_transform(y_pred_adaboost)
    y_latih = le_dict['status_kesesuaian'].inverse_transform(y_latih)

    evaluation_results = {
        'accuracy': {
            'svm': accuracy_score(y_latih, y_pred_svm),
            'adaboost': accuracy_score(y_latih, y_pred_adaboost)
        },
        'confusion_matrix': {
            'svm': confusion_matrix(y_latih, y_pred_svm).tolist(),
            'adaboost': confusion_matrix(y_latih, y_pred_adaboost).tolist()
        },
        'classification_report': {
            'svm': classification_report(y_latih, y_pred_svm, target_names=le_dict['status_kesesuaian'].classes_),
            'adaboost': classification_report(y_latih, y_pred_adaboost, target_names=le_dict['status_kesesuaian'].classes_)
        }
    }

    # Format evaluation results as a string
    evaluation_text = f"""
    Evaluation Results:

    SVM Accuracy: {evaluation_results['accuracy']['svm']}
    SVM Confusion Matrix: {evaluation_results['confusion_matrix']['svm']}
    SVM Classification Report: {evaluation_results['classification_report']['svm']}

    AdaBoost Accuracy: {evaluation_results['accuracy']['adaboost']}
    AdaBoost Confusion Matrix: {evaluation_results['confusion_matrix']['adaboost']}
    AdaBoost Classification Report: {evaluation_results['classification_report']['adaboost']}
    """

    return evaluation_text, 200

@app.route('/save_data_latih', methods=['POST'])
def save_data_latih():
    try:
        # Ambil data dari JSON payload
        data = request.json
        nama = data.get('nama')
        penghasilan = data.get('penghasilan')
        status_ekonomi = data.get('status_ekonomi')
        jumlah_tanggungan = data.get('jumlah_tanggungan')
        layak_pip = data.get('layak_pip')
        alasan_layak_pip = data.get('alasan_layak_pip')
        tahun_penerimaan = data.get('tahun_penerimaan')
        jumlah_bantuan = data.get('jumlah_bantuan')
        status_bantuan = data.get('status_bantuan')
        status_kesesuaian = data.get('status_kesesuaian')

        # Buat koneksi ke database menggunakan mysql-connector
        connection = mysql.connector.connect(
            host='localhost',
            database='fahri',
            user='root',
            password=''
        )

        if connection.is_connected():
            cursor = connection.cursor()
            query = """
            INSERT INTO data_latih (
                nama, penghasilan, status_ekonomi, jumlah_tanggungan,
                layak_pip, alasan_layak_pip, tahun_penerimaan,
                jumlah_bantuan, status_bantuan, status_kesesuaian
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            params = (
                nama, penghasilan, status_ekonomi, jumlah_tanggungan,
                layak_pip, alasan_layak_pip, tahun_penerimaan,
                jumlah_bantuan, status_bantuan, status_kesesuaian
            )
            cursor.execute(query, params)
            connection.commit()
            cursor.close()
            connection.close()

        return jsonify({"message": "Data berhasil disimpan"}), 201

    except Error as e:
        error_msg = str(e)
        return jsonify({"error": "Gagal menyimpan data", "message": error_msg}), 400

@app.route('/save_data_uji', methods=['POST'])
def save_data_uji():
    try:
        # Ambil data dari JSON payload
        data = request.json
        nama = data.get('nama')
        penghasilan = data.get('penghasilan')
        status_ekonomi = data.get('status_ekonomi')
        jumlah_tanggungan = data.get('jumlah_tanggungan')
        layak_pip = data.get('layak_pip')
        alasan_layak_pip = data.get('alasan_layak_pip')
        tahun_penerimaan = data.get('tahun_penerimaan')
        jumlah_bantuan = data.get('jumlah_bantuan')
        status_bantuan = data.get('status_bantuan')

        # Buat koneksi ke database menggunakan mysql-connector
        connection = mysql.connector.connect(
            host='localhost',
            database='fahri',
            user='root',
            password=''
        )

        if connection.is_connected():
            cursor = connection.cursor()
            query = """
            INSERT INTO data_uji (
                nama, penghasilan, status_ekonomi, jumlah_tanggungan,
                layak_pip, alasan_layak_pip, tahun_penerimaan,
                jumlah_bantuan, status_bantuan
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            params = (
                nama, penghasilan, status_ekonomi, jumlah_tanggungan,
                layak_pip, alasan_layak_pip, tahun_penerimaan,
                jumlah_bantuan, status_bantuan
            )
            cursor.execute(query, params)
            connection.commit()
            cursor.close()
            connection.close()

        return jsonify({"message": "Data berhasil disimpan"}), 201

    except Error as e:
        error_msg = str(e)
        return jsonify({"error": "Gagal menyimpan data", "message": error_msg}), 400

@app.route('/data_latih', methods=['GET'])
def get_data_latih():
    data_latih, _ = load_data_from_db()
    return jsonify(data_latih.to_dict(orient='records'))

@app.route('/data_uji', methods=['GET'])
def get_data_uji():
    _, data_uji = load_data_from_db()
    return jsonify(data_uji.to_dict(orient='records'))

# Page
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    ensure_model_directory()
    app.run(debug=True)
