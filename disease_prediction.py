import pandas as pd
import warnings
warnings.filterwarnings('ignore')
# Load the dataset
file_path = 'Training.csv'
data = pd.read_csv(file_path)
data.head()
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Handling missing values and removing unnecessary columns
data = data.drop(columns=data.columns[-1])  # Drop the last column if it's unnamed or irrelevant
data = data.dropna()  # Drop rows with NaN values

# Splitting the dataset into features (X) and target (y)
X = data.drop('prognosis', axis=1)
y = data['prognosis']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Creating and training the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Data")

print(accuracy, report)

# Example usage
print("Please input the values for the following features:")
# predicted_prognosis = user_input_to_prediction(model, top_10_feature_names)
print(f"The predicted prognosis is")

prediction = model.predict([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
print(prediction)
import pickle
pickle_out = open('model.pkl',"wb")
pickle.dump(model , pickle_out)
pickle_out.close()