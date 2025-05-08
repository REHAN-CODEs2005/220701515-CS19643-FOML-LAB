# model.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def train_model():
    # Sample training data
    data = {
        'Steps': [3000, 7000, 10000, 3000, 12000, 5000, 8000],
        'SleepHours': [5, 7, 8, 4, 9, 6, 7],
        'BMI': [25, 22, 20, 28, 21, 24, 23],
        'Weight': [70, 65, 60, 80, 55, 68, 62],
        'Height': [170, 175, 180, 165, 178, 172, 176],
        'Glucose': [90, 85, 80, 95, 78, 88, 82],
        'HeartRate': [80, 72, 68, 85, 65, 77, 70],
        'HealthStatus': ['Average', 'Good', 'Excellent', 'Poor', 'Excellent', 'Good', 'Good']
    }
    df = pd.DataFrame(data)

    # Features and Target
    X = df[['Steps', 'SleepHours', 'BMI', 'Weight', 'Height', 'Glucose', 'HeartRate']]
    y = df['HealthStatus']

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Dynamically adjust n_neighbors for KNN
    knn_neighbors = min(3, len(X_train))

    # Define multiple models
    models = {
        "DecisionTree": DecisionTreeClassifier(),
        "RandomForest": RandomForestClassifier(),
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "KNN": KNeighborsClassifier(n_neighbors=knn_neighbors)
    }

    best_accuracy = 0
    best_model_name = None
    best_model = None

    # Train and evaluate each model
    for model_name, model_instance in models.items():
        model_instance.fit(X_train, y_train)
        y_pred = model_instance.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{model_name} Accuracy: {accuracy:.2f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = model_name
            best_model = model_instance

    # Save the best model
    joblib.dump(best_model, 'health_model.pkl')
    print(f"\n✅ Best model '{best_model_name}' saved as 'health_model.pkl' with accuracy: {best_accuracy:.2f}")

if __name__ == "__main__":
    train_model()
