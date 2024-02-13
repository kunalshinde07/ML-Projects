import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score

# Load your data from Excel (replace 'your_dataset.xlsx' with your actual Excel file)
df_all = pd.read_excel('HOTEL_REVIEW.xlsx')

# Assuming df_all has 'Review' and 'Sentiment' columns
x_train, x_test, y_train, y_test = train_test_split(df_all['Review'], df_all['Sentiment'], test_size=0.2, random_state=42)

# Vectorize the text data
vectorizer = CountVectorizer()
x_train_vec = vectorizer.fit_transform(x_train)
x_test_vec = vectorizer.transform(x_test)

# Models
models = {
    'Logistic Regression': LogisticRegression(),
    'SVM': SVC(kernel='linear'),
    'Naive Bayes': MultinomialNB(),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

# Train models and calculate F1 scores
f1_scores = {}

for model_name, model in models.items():
    model.fit(x_train_vec, y_train)
    y_pred = model.predict(x_test_vec)
    f1 = f1_score(y_test, y_pred, average='weighted')
    f1_scores[model_name] = f1

# Streamlit App
st.title('Hotel Review Classifier Web App')

# User input
user_input = st.text_area("Enter a review:", "Type your review here...")

# Sentiment prediction
if st.button("Predict Sentiment"):
    # Preprocess user input
    user_input_vec = vectorizer.transform([user_input])

    # Predict sentiment using a selected model (e.g., Logistic Regression)
    selected_model = LogisticRegression()  # Change this to the desired model
    selected_model.fit(x_train_vec, y_train)
    user_sentiment = selected_model.predict(user_input_vec)

    # Display sentiment
    st.write(f"Predicted Sentiment: {user_sentiment[0]}")