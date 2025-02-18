import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Read the data
df = pd.read_csv("Movie.csv")
print(df.head())


# Replace '0' with 'Flop' and '1' with 'Hit' in the 'Success' column
df['Success'] = df['Success'].replace({0: 'Flop', 1: 'Hit'})

# Drop rows with unexpected values in the 'Success' column
df = df[df['Success'].isin(['Flop', 'Hit'])]

# Handle missing values
df.fillna(df.mean(numeric_only=True), inplace=True)

# Encode categorical columns
label_encoders = {}
for col in ['Title', 'Genre', 'Description', 'Director', 'Actors']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Convert 'Success' to binary values
success_encoder = LabelEncoder()
df['Success'] = success_encoder.fit_transform(df['Success'])


# Split the dataset
X = df.drop(columns=['Success'])
y = df['Success']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Specify the correct labels for the classification report
report = classification_report(y_test, y_pred, target_names=success_encoder.classes_)

# Print results
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", report)

def predict_movie_success(movie_name):
    # Find the row in the dataset corresponding to the given movie name
    try:
        movie_row = df[df['Title'] == label_encoders['Title'].transform([movie_name])[0]]
        if movie_row.empty:
            return f"Movie '{movie_name}' not found in the dataset."
    except ValueError:
        return f"Movie '{movie_name}' not found in the dataset."
    
    # Drop the 'Success' column to use only the features for prediction
    movie_features = movie_row.drop(columns=['Success'])
    
    # Predict using the trained model
    prediction = model.predict(movie_features)
    predicted_class = success_encoder.inverse_transform(prediction)
    
    return f"The movie '{movie_name}' is predicted to be a '{predicted_class[0]}'."

# Example usage: Input a movie name
movie_name = input("Enter the movie name: ")

# Predict and display the result
result = predict_movie_success(movie_name)
print(result)


