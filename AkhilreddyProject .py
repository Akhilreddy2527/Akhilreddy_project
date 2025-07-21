## Sentiment Analysis for Amazon Product Reviews

# --- Import Required Libraries ---
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Suppress warnings to keep output clean
import warnings
warnings.filterwarnings('ignore')


# --- Load Dataset ---
df = pd.read_csv("amazon_reviews.csv")

# Initial check of the dataset
print(f"Total number of reviews: {len(df)}")
print("\nPreview of the data:")
print(df.head())


# --- Data Cleaning & Preparation ---

# Check for any missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Fill missing reviews with empty strings, and missing names with "Anonymous"
df['reviewText'].fillna('', inplace=True)
df['reviewerName'].fillna('Anonymous', inplace=True)

# Convert reviewTime to datetime format for consistency (not used in model here)
df['reviewTime'] = pd.to_datetime(df['reviewTime'])

# Create a new binary sentiment label: 1 for positive (4-5 stars), 0 for negative (1-3 stars)
df['sentiment'] = (df['overall'] >= 4).astype(int)


# --- Quick Data Visualization ---

# Plot distribution of star ratings
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='overall')
plt.title('Distribution of Star Ratings')
plt.show()

# Display correlation between numeric columns
numeric_df = df.select_dtypes(include=['number'])
corr = numeric_df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .75})
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()


# --- Text Preprocessing Function ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers

    tokens = word_tokenize(text)
    stops = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stops]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return ' '.join(tokens)

# Apply text cleaning to the review text
print("Cleaning review texts. This may take a moment...")
df['clean_text'] = df['reviewText'].apply(clean_text)


# --- Train/Test Split ---
X = df['clean_text']
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# --- Model Setup & Evaluation ---
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

# Define three different models to compare
models = {
    'LogReg': LogisticRegression(random_state=42, max_iter=1000),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(probability=True, random_state=42)
}

results = {}

# Loop through each model and evaluate its performance
for name, model in models.items():
    print(f"\n{'='*50}")
    print(f"Training and evaluating: {name}")
    
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('classifier', model)
    ])
    
    # Cross-validation on training set
    scores = cross_val_score(pipe, X_train, y_train, cv=5)
    print(f"Cross-validation accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
    
    # Train on full training data
    pipe.fit(X_train, y_train)
    
    # Predict on test set
    preds = pipe.predict(X_test)
    pred_proba = pipe.predict_proba(X_test)[:, 1]
    
    # Evaluate with multiple metrics
    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc_roc = roc_auc_score(y_test, pred_proba)
    
    results[name] = {
        'model': pipe,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_roc': auc_roc
    }

    # Show evaluation metrics
    print("\nTest Set Performance:")
    print(f"Accuracy:  {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 Score:  {f1:.3f}")
    print(f"AUC-ROC:   {auc_roc:.3f}")


# --- Model Comparison Summary ---
print("\n" + "="*70)
print("Summary of All Models:")
print("="*70)
print(f"{'Model':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'AUC-ROC':<10}")
print("-"*70)
for name in results:
    r = results[name]
    print(f"{name:<15} {r['accuracy']:<10.3f} {r['precision']:<10.3f} "
          f"{r['recall']:<10.3f} {r['f1']:<10.3f} {r['auc_roc']:<10.3f}")

# Identify the best model based on F1 Score
best_model = max(results.items(), key=lambda x: x[1]['f1'])
print(f"\nBest Performing Model: {best_model[0]} (F1 Score: {best_model[1]['f1']:.3f})")


# --- Confusion Matrix for Best Model ---
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

# Predict using best model and plot confusion matrix
best_pipe = best_model[1]['model']
test_preds = best_pipe.predict(X_test)
test_probs = best_pipe.predict_proba(X_test)[:, 1]

plot_confusion_matrix(y_test, test_preds, f"Confusion Matrix - {best_model[0]}")


# --- ROC Curve ---
from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(y_test, test_probs)
auc_score = roc_auc_score(y_test, test_probs)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.fill_between(fpr, tpr, alpha=0.1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()


# --- Try Out Predictions on Some Reviews ---
def analyze_review(review_text, actual_sentiment=None):
    # Function to predict sentiment of a single review
    cleaned = clean_text(review_text)
    prob = best_pipe.predict_proba([cleaned])[0, 1]
    pred = 'Positive' if prob > 0.5 else 'Negative'
    
    print(f"\nReview: {review_text[:100]}...")
    print(f"Prediction: {pred} (Confidence: {prob:.2f})")
    if actual_sentiment is not None:
        print(f"Actual Sentiment: {'Positive' if actual_sentiment else 'Negative'}")

# Run sentiment predictions on a few random test reviews
print("\nExample Predictions:")
n_examples = 5
random_indices = np.random.randint(0, len(X_test), n_examples)
for idx in random_indices:
    analyze_review(X_test.iloc[idx], y_test.iloc[idx])

# Show examples where the model made wrong predictions
error_indices = np.where(y_test != test_preds)[0]
if len(error_indices) > 0:
    print("\nExamples where the model prediction was incorrect:")
    for idx in error_indices[:3]:
        analyze_review(X_test.iloc[idx], y_test.iloc[idx])


# --- Save the Best Model (Optional) ---
import pickle
import os

# You can uncomment and use this if needed
# os.makedirs('models', exist_ok=True)
# with open('models/best_sentiment_model.pkl', 'wb') as f:
#     pickle.dump(best_pipe, f)


# --- Final Metrics Report ---
from sklearn.metrics import classification_report

print("\nDetailed Classification Report:")
print(classification_report(y_test, test_preds))

# Manually calculate F1 score just for fun
precision = precision_score(y_test, test_preds)
recall = recall_score(y_test, test_preds)
f1 = 2 * (precision * recall) / (precision + recall)

print("\nFinal Summary:")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1 Score:  {f1:.3f}")
print(f"AUC-ROC:   {auc_score:.3f}")


# Just a placeholder if needed for script structure
if __name__ == "__main__":
    print()
