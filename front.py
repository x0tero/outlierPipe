from back import *

ITERATIONS = 3

# Supported OneClass models: OneClassSVM, ConvexHull, AutoEncoder, PCA
# Supported standard models: LinearRegression, IsolationForest, kNN, LOF, DBSCAN
oc_models = ["AutoEncoder", "OneClassSVM", "ConvexHull", "PCA"]
std_models = ["LinearRegression", "IsolationForest", "kNN", "LOF"]
metrics = ["F1", "MCC", "AuC"]

data = preprocess_data()

# Supported labelers: IsolationForest, LinearRegression
labeler = "IsolationForest"

# Supported outlier generators: simple, context, night, mixed
outlier_generator = "context"

r1, r2 = [], []
for i in range(ITERATIONS):
    filtered_data = label(data, labeler)
    test = generate_test(filtered_data, outlier_generator)
    r1.append( evaluate(fit_models(std_models, data), test, metrics) )
    r2.append( evaluate(fit_models(oc_models, filtered_data), test, metrics) )

# Results exported to the same path as this file
export_results(merge_results(r1, r2), metrics)