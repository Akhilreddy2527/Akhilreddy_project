## Sentiment Analysis for Amazon Product Reviews

# Import necessary libraries
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

# Silence those annoying warnings
import warnings
warnings.filterwarnings('ignore')  





# Load our dataset 
df = pd.read_csv("amazon_reviews.csv")  

# Basic data exploration
print(f"Got {len(df)} reviews to analyze")
print("\nQuick peek at the data:")
print(df.head())





# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())





# Handle missing stuff - probably could do this better
df['reviewText'].fillna('', inplace=True)  # empty string for missing reviews
df['reviewerName'].fillna('Anonymous', inplace=True)  # better than leaving it empty

# Convert dates - note to self: check if this actually matters for the model
df['reviewTime'] = pd.to_datetime(df['reviewTime'])

# Create sentiment labels - keeping it simple with binary classification
# Might want to try different thresholds later
df['sentiment'] = (df['overall'] >= 4).astype(int)  # 1 for positive (4-5 stars), 0 for negative (1-3 stars)





# Some rough visualizations
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='overall')
plt.title('Star Ratings Distribution')
plt.show()




# Select numeric columns only
numeric_df = df.select_dtypes(include=['number'])

# Compute the correlation matrix
corr = numeric_df.corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .75})
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()





# This preprocessing function could probably be optimized 
def clean_text(text):
    text = text.lower()
    
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)  # dump punctuation
    text = re.sub(r'\d+', '', text)  # dump numbers
    
    # Tokenize and remove stop words
    tokens = word_tokenize(text)
    stops = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stops]
    
    # Lemmatize - this is slow but helpful
    lemmer = WordNetLemmatizer()
    tokens = [lemmer.lemmatize(t) for t in tokens]
    
    return ' '.join(tokens)

print("Cleaning up review text...")
df['clean_text'] = df['reviewText'].apply(clean_text)  





# Split data - standard 80/20 split
X = df['clean_text']
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Import metrics here to avoid scrolling up and down
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score)  # probably forgetting something

# The usual suspects - tried and tested models
models = {
    'LogReg': LogisticRegression(random_state=42, max_iter=1000),  # good old logistic regression
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),  # reliable but slow
    'SVM': SVC(probability=True, random_state=42)  # my favorite but takes forever
}

# Dictionary to store all our results - might need this later
results = {}

# Let's see how each model performs
for name, model in models.items():
    print(f"\n{'='*50}")  # makes output easier to read
    print(f"Training and evaluating {name}...")
    
    # Basic pipeline - nothing fancy
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),  # limiting features to prevent memory explosion
        ('classifier', model)
    ])
    
    # Cross validation first
    scores = cross_val_score(pipe, X_train, y_train, cv=5)
    print(f"CV accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
    
    # Train model - fingers crossed
    pipe.fit(X_train, y_train)
    
    # Get predictions
    preds = pipe.predict(X_test)
    pred_proba = pipe.predict_proba(X_test)[:, 1]  # need this for AUC-ROC
    
    # Calculate all metrics - probably overkill but good to have
    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc_roc = roc_auc_score(y_test, pred_proba)
    
    # Store everything in our results dict
    results[name] = {
        'model': pipe,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_roc': auc_roc
    }
    
    # Print metrics in a somewhat nice format
    print("\nTest Set Metrics:")
    print(f"Accuracy:  {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1-Score:  {f1:.3f}")
    print(f"AUC-ROC:   {auc_roc:.3f}")

# Compare models side by side
print("\n" + "="*70)
print("Model Comparison Summary:")
print("="*70)
print(f"{'Model':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'AUC-ROC':<10}")
print("-"*70)

for name in results:
    r = results[name]
    print(f"{name:<15} {r['accuracy']:<10.3f} {r['precision']:<10.3f} "
          f"{r['recall']:<10.3f} {r['f1']:<10.3f} {r['auc_roc']:<10.3f}")

# Find best model
best_model = max(results.items(), key=lambda x: x[1]['f1'])
print("\nBest performing model:")
print(f"{best_model[0]} (F1: {best_model[1]['f1']:.3f})")





# Confusion matrix 
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix'):
    # Quick helper function I copied from my other project
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

# Get predictions from best model
best_pipe = best_model[1]['model']
test_preds = best_pipe.predict(X_test)
test_probs = best_pipe.predict_proba(X_test)[:, 1]  # just want positive class probs

# Plot confusion matrix
plot_confusion_matrix(y_test, test_preds, f"Confusion Matrix - {best_model[0]}")

# ROC curve - always a crowd pleaser
from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, _ = roc_curve(y_test, test_probs)
auc_score = roc_auc_score(y_test, test_probs)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')  # diagonal line
plt.fill_between(fpr, tpr, alpha=0.1)  # looks fancy
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()





# Let's look at some example predictions
def analyze_review(review_text, actual_sentiment=None):
    # Useful for testing individual reviews
    # Could probably optimize this but it works
    cleaned = clean_text(review_text)
    prob = best_pipe.predict_proba([cleaned])[0, 1]
    pred = 'Positive' if prob > 0.5 else 'Negative'
    
    print(f"\nReview: {review_text[:100]}...")  # just show first 100 chars
    print(f"Prediction: {pred} (confidence: {prob:.2f})")
    if actual_sentiment is not None:
        print(f"Actual sentiment: {'Positive' if actual_sentiment else 'Negative'}")
        
# Test it on a few random examples
n_examples = 5
random_indices = np.random.randint(0, len(X_test), n_examples)
for idx in random_indices:
    analyze_review(X_test.iloc[idx], y_test.iloc[idx])

# Quick error analysis - might be useful later
errors = y_test != test_preds
error_indices = np.where(errors)[0]

if len(error_indices) > 0:
    print("\nSome examples where model got it wrong:")
    for idx in error_indices[:3]:  # just look at first 3 mistakes
        analyze_review(X_test.iloc[idx], y_test.iloc[idx])

# Save the model - remember to create models folder first!
import pickle
import os


# Some final thoughts and metrics
from sklearn.metrics import classification_report

print("\nDetailed Classification Report:")
print(classification_report(y_test, test_preds))

# Calculate some additional metrics I might need later
from sklearn.metrics import precision_score, recall_score

precision = precision_score(y_test, test_preds)
recall = recall_score(y_test, test_preds)
f1 = 2 * (precision * recall) / (precision + recall)  # could've used f1_score but whatever

print("\nFinal Metrics:")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1 Score: {f1:.3f}")
print(f"AUC-ROC: {auc_score:.3f}")

if __name__ == "__main__":
    print()

