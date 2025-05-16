import pandas as pd
import numpy as np
import datetime, calendar, os, csv
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import OneClassSVM
from scipy.spatial import ConvexHull, Delaunay
from sklearn.decomposition import PCA
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mae
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score, 
    confusion_matrix, 
    f1_score,
    matthews_corrcoef
)


class OneClassConvexHull():
    hull = None 
    hull_delaunay = None
    def __init__(self):
        pass
    def fit(self, df):
        self.hull = ConvexHull(df)
        self.hull_delaunay = Delaunay(df)
        return self
    def predict(self, X):
        return self.hull_delaunay.find_simplex(X)
    


class AutoEncoder():
    m = None
    cbacks = None
    def __init__(self, shape):
        input_layer = Input(shape=(shape[1],))
        x = Dense(64, activation='relu')(input_layer)
        x = Dense(16, activation='relu')(x)
        code = Dense(2, activation='relu')(x)
        x = Dense(16, activation='relu')(code)
        x = Dense(64, activation='relu')(x)
        output_layer = Dense(shape[1], activation='relu')(x)
        self.m = Model(input_layer, output_layer, name='anomaly')
        earlystopping = EarlyStopping(monitor='val_loss',min_delta = 0, patience = 5, verbose = 1,restore_best_weights=True)
        self.cbacks = [earlystopping]
        self.m.compile(loss='mae', optimizer=Adam())
    def fit(self, X):
        train = X.iloc[:int(0.9 * len(X))]; test = X.iloc[int(0.9 * len(X)):]
        self.m.fit(train, train, epochs=25, batch_size=64, validation_data=(test, test), callbacks=self.cbacks, shuffle=True)
        return self
    def predict(self, X, percentil):
        recon = self.m.predict(X, verbose=0)
        recon_error = mae(recon, X)
        return np.where(recon_error > np.percentile(recon_error, percentil), -1, 1)


def preprocess_data():
    base_path_citic = r"Y:\Datasets"
    base_path_home = r"C:\Users\Wouter\Desktop\anomalias"
    base_path = base_path_citic

    file_path_AC = r"sotavento\data\raw\Analizador AC Fotovoltaica Este (A11).txt"
    file_path_DC = r"sotavento\data\raw\Analizador DC Fotovoltaica Este (A14).txt"
    file_path_irr = r"sotavento\data\raw\Radiacion Fotovoltaica Este (R1).txt"
    file_path_temp = r"sotavento\data\raw\Temperatura Ambiente (Mastil).txt"

    full_path_AC = os.path.join(base_path, file_path_AC)
    full_path_DC = os.path.join(base_path, file_path_DC)
    full_path_irr = os.path.join(base_path, file_path_irr)
    full_path_temp = os.path.join(base_path, file_path_temp)
    df_ac = pd.read_csv(full_path_AC, delimiter=';', decimal=',', na_values='NULL', parse_dates=['data'])
    df_irr = pd.read_csv(full_path_irr, delimiter=';', decimal=',', na_values='NULL', parse_dates=['data'])
    df_temp = pd.read_csv(full_path_temp, delimiter=';', decimal=',', na_values='NULL', parse_dates=['data'])
    df_temp['data'] = pd.to_datetime(df_temp['data'], format='%d/%m/%Y %H:%M:%S')
    df_irr['data'] = pd.to_datetime(df_irr['data'], format='%d/%m/%Y %H:%M:%S')
    df_ac['data'] = pd.to_datetime(df_ac['data'], format='%d/%m/%Y %H:%M:%S') 
    merged_df = df_temp.merge(df_irr, on='data', how='inner').merge(df_ac, on='data', how='inner')

    def assign_month_group(month):
        if month in (12, 1, 2, 11):
            return 1
        elif month in (3, 4, 9, 10):
            return 2
        elif month in (5, 6, 7, 8):
            return 3


    def assign_group_hour(hour):
        if (hour >= 21) or (hour < 6):
            return 1
        elif 6 <= hour < 10:
            return 2
        elif 10 <= hour < 15:
            return 3
        elif 15 <= hour < 21:
            return 4
        
    encoder = OneHotEncoder(sparse_output=False, drop=None)
    merged_df['mes'] = merged_df['data'].dt.month
    merged_df['grupo_mes'] = merged_df['mes'].apply(assign_month_group)

    merged_df['hora'] = merged_df['data'].dt.hour
    merged_df['grupo_hora'] = merged_df['hora'].apply(assign_group_hour)
    

    mes_codificado = encoder.fit_transform(merged_df[['grupo_mes']])
    mes_tmp = pd.DataFrame(mes_codificado, columns=[f'grupo_mes_{i}' for i in encoder.categories_[0]])
    merged_df = pd.concat([merged_df, mes_tmp], axis=1)

    
    hora_codificada = encoder.fit_transform(merged_df[['grupo_hora']])
    hora_tmp = pd.DataFrame(
        hora_codificada,
        columns=[f'grupo_hora_{i}' for i in encoder.categories_[0]]
    )
    merged_df = pd.concat([merged_df, hora_tmp], axis=1)

    merged_df = merged_df.drop(columns=['hora', 'mes', 'data', 'valor_max', 'valor_min', 'id_y', 'n_datos_y', 'id_x', 'V', 'I', 
                             'VAr', 'Wh_i', 'VArh_i', 'Wh_e', 'VArh_e', 'grupo_hora', 'grupo_mes', 'n_datos_x'])
    merged_df = merged_df.dropna()
    print(merged_df.head())
    return merged_df



def label(data, labeler):
    filtered_data = None
    model = None
    outliers = None
    if labeler == "IsolationForest":
        model = IsolationForest(n_estimators=100).fit(data)
        outliers = model.predict(data) < 0
    elif labeler == "LinearRegression":
        model = LinearRegression().fit(data.drop(columns=['W']), data['W'])
        residuals = data['W'] - model.predict(data.drop(columns=['W']))
        outliers = np.abs(residuals - np.mean(residuals)) > (2.5 * np.std(residuals))

    
    filtered_data = data[~outliers]

    return filtered_data



def generate_test(labeled_data, outlier_generator):
    testdata = labeled_data.copy()
    if (outlier_generator == "simple"):
        outliers_indexes = np.random.choice(testdata.index, size=int(0.05 * len(testdata)), replace=False)
        testdata["label"] = False
        testdata.loc[outliers_indexes, "label"] = True
        testdata.loc[outliers_indexes, 'W'] += 50
    elif (outlier_generator == "context"):
        testdata["label"] = False
        high_irradiance = testdata['valor_med'] > testdata['valor_med'].quantile(0.9)
        outliers_1 = testdata[high_irradiance].sample(frac=0.03, random_state=42).index
        testdata.loc[outliers_1, 'W'] *= 0.2
        testdata.loc[outliers_1, "label"] = True
    elif (outlier_generator == "night"):
        outliers_2 = testdata[testdata['grupo_hora_1'] == 1].sample(frac=0.02, random_state=42).index
        testdata.loc[outliers_2, 'W'] += 30
        testdata.loc[outliers_2, "label"] = True
    elif (outlier_generator == "degradation"): # Gradual degradation
        degradation_days = testdata['day'] > 15  # Assume 'day' column exists
        outliers_3 = testdata[degradation_days].sample(frac=0.05, random_state=42).index
        testdata.loc[outliers_3, 'W'] *= np.random.uniform(0.5, 0.8, len(outliers_3))  # Random drop
        testdata.loc[outliers_3, "label"] = True
    elif (outlier_generator == "mixed"):
        pass
    
    return testdata



def generate_metrics(gt, predicted):
    result = {}
    result['F1'] = f1_score(gt, predicted)
    result['AuC'] = roc_auc_score(gt, predicted)
    result['accuracy'] = accuracy_score(gt, predicted)
    result['precision'] = precision_score(gt, predicted)
    result['recall'] = recall_score(gt, predicted)
    result['MCC'] = matthews_corrcoef(gt, predicted)
    return result



def evaluate(trained_models, testdata, metrics):
    results = {}
    print(f"trained_models: {trained_models}")
    for model in trained_models:
        print(f"Nome da clase: {model.__class__.__name__}")
        if model.__class__.__name__ == "NearestNeighbors":
            distances, indexes = model.kneighbors(testdata.drop(columns=['label']))
            outliers = distances.mean(axis=1) > 1.5 * np.std(distances)
        elif model.__class__.__name__ == "OneClassConvexHull":
            outliers = model.predict(testdata[['valor_med','temp_med','W']]) < 0
        elif model.__class__.__name__ == "AutoEncoder":
            outliers = model.predict(testdata.drop(columns=['label']), 95) < 0
        elif model.__class__.__name__ == "LinearRegression":
            residuals = testdata['W'] - model.predict(testdata.drop(columns=['W','label']))
            outliers = np.abs(residuals - np.mean(residuals)) > (2.5 * np.std(residuals))
        elif model.__class__.__name__ == "LocalOutlierFactor":
            outliers = model.fit_predict(testdata.drop(columns=['label'])) < 0
        elif model.__class__.__name__ == "PCA":
            scaled_data = StandardScaler().fit_transform(testdata.drop(columns=['label']))
            pca_transformed = model.transform(scaled_data)
            distances = np.sqrt(np.sum((pca_transformed- np.mean(pca_transformed, axis=0)) ** 2, axis=1))
            outliers = distances > np.percentile(distances, 95)
        else:
            outliers = model.predict(testdata.drop(columns=['label'])) < 0
        result_metrics = generate_metrics(testdata['label'], outliers)
        results[model.__class__.__name__] = {k: result_metrics[k] for k in metrics if k in result_metrics}
    return results



def fit_models(models, data):
    fitted_models = []
    for model in models:
        if model == "IsolationForest":
            fitted_models.append(IsolationForest(n_estimators=100).fit(data))
        elif model == "LinearRegression":
            fitted_models.append(LinearRegression().fit(data.drop(columns=['W']), data['W']))
        elif model == "kNN":
            fitted_models.append(NearestNeighbors(n_neighbors=4).fit(data))
        elif model == "OneClassSVM":
            fitted_models.append(OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1).fit(data))
        elif model == "ConvexHull":
            fitted_models.append(OneClassConvexHull().fit(data[['valor_med','temp_med','W']]))
        elif model == "LOF":
            fitted_models.append(LocalOutlierFactor(n_neighbors=20, contamination=0.1).fit(data))
        elif model == "PCA":
            fitted_models.append(PCA(n_components=2).fit(StandardScaler().fit_transform(data)))
        elif model == "AutoEncoder":
            print(f"shape: {data.values.shape}")
            fitted_models.append(AutoEncoder(data.values.shape).fit(data))
    return fitted_models



def merge_results(list1, list2):
    merged_list = []
    for dict1, dict2 in zip(list1, list2):
        merged_dict = {**dict1, **dict2}
        merged_list.append(merged_dict)
    return merged_list



def export_results(r, _metrics):
    method_metrics = {}
    for entry in r:
        print(f"entry: {entry}")
        for method, metrics in entry.items():
            if method not in method_metrics:
                method_metrics[method] = {}
                for _metric in _metrics:
                    method_metrics[method][_metric] = []
            for _metric in _metrics:
                    method_metrics[method][_metric].append(metrics[_metric])


    result = {}
    for method, metrics in method_metrics.items():
        result[method] = {}
        for _metric in _metrics:
            result[method][_metric] = {"mean": np.mean(metrics[_metric]), "std": np.std(metrics[_metric], ddof=1)}


    rows = []
    for model, metrics in result.items():
        row = {'Model': model}
        for metric, stats in metrics.items():
            row[f"{metric}_mean"] = stats['mean']
            row[f"{metric}_std"] = stats['std']
        rows.append(row)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "model_metrics.csv")
    print(f"rows {rows}")
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    