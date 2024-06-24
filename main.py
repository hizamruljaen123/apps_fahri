import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import joblib

# Load the data
data = pd.read_excel('data_full.xlsx')

# Fill missing values if any
data.fillna('', inplace=True)

# Encode categorical features using LabelEncoder
label_encoder = LabelEncoder()

# Apply LabelEncoder to categorical columns
data['L/P'] = label_encoder.fit_transform(data['L/P'])
data['Penghasilan'] = label_encoder.fit_transform(data['Penghasilan'])
data['Status Ekonomi'] = label_encoder.fit_transform(data['Status Ekonomi'])
data['Layak PIP'] = label_encoder.fit_transform(data['Layak PIP'])
data['Status Bantuan'] = label_encoder.fit_transform(data['Status Bantuan'])

# Select features and target variable
X = data[['L/P', 'Penghasilan', 'Status Ekonomi', 'Jumlah Tanggungan', 'Layak PIP']]
y = data['Status Bantuan']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the SVM model with a linear kernel
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Make predictions
y_pred = svm_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_report_result = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report_result)

# Save the trained model to a file
model_filename = 'svm_model.joblib'
joblib.dump(svm_model, model_filename)

# Save the scaler to a file
scaler_filename = 'scaler.joblib'
joblib.dump(scaler, scaler_filename)

print(f"Model saved to {model_filename}")
print(f"Scaler saved to {scaler_filename}")

# Visualize the SVM model results with hyperplane
def plot_svm_decision_boundary(model, X, y):
    # Define the mesh grid for plotting
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    # Flatten the grid to get the decision function
    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary and margins
    plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
    plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
    plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred', alpha=0.3)

    # Plot the data points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k')
    
    # Add labels and title
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('SVM Decision Boundary with Hyperplane')
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.show()

# For visualization, reduce to 2 principal components
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
svm_model.fit(X_train_pca, y_train)

# Plot decision boundary with hyperplane
plot_svm_decision_boundary(svm_model, X_train_pca, y_train)
