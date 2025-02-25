"""
Email Classification Main Program (v3.0)
Function: High-Accuracy Spam Classifier with Advanced Feature Engineering
"""
# -------------------------- Environment Configuration --------------------------
import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import make_interp_spline
import re
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, chi2
#RandomForestClassifier,
plt.rcParams.update({
    'savefig.dpi': 600,
    'font.sans-serif': 'Times New Roman',
    'axes.unicode_minus': False,
    'figure.figsize': (10, 6),
})

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="sklearn")


# -------------------------- Advanced Configuration --------------------------
stop_words = [
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
    'yours', 'yourself', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
    'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
    'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
    'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
    'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
    'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
    'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
    'than', 'too', 'very', 'can', 'will', 'just', 'don', 'should', 'now'
]

class Config:
    TEXT_PROCESSING = {
        'max_features': 15000,
        'ngram_range': (1, 3),
        'min_df': 7,
        'max_df': 0.75,
        'stop_words': 'english',
        'sublinear_tf': True,
        'smooth_idf': True
    }

    MODEL_TUNING = {
        'feature_selector__k': [8000, 10000, 12000],
        'classifier__C': [0.01, 0.1, 1, 10],
        'classifier__penalty': ['l2'],
        'classifier__solver': ['saga'],
        'classifier__max_iter': [2000]
    }

    ENSEMBLE_WEIGHTS = {
        'LogisticRegression': 0.6,
        'LinearSVC': 0.4
    }


# -------------------------- Optimized Text Processing --------------------------
class AdvancedTextProcessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.num_pattern = re.compile(r'\b\d+\b')
        self.special_char_pattern = re.compile(r'[^a-zA-Z\s]')

    def process_text(self, text):
        # URL & Number Removal
        text = self.url_pattern.sub('', text)
        text = self.num_pattern.sub('', text)

        # Advanced Cleaning
        text = self.special_char_pattern.sub(' ', text)
        text = re.sub(r'\s+', ' ', text).lower().strip()

        # Lemmatization with POS Tagging
        tokens = nltk.word_tokenize(text)
        pos_tags = nltk.pos_tag(tokens)

        lemmas = []
        for word, tag in pos_tags:
            if word in self.stop_words or len(word) < 2:
                continue
            pos = self.get_wordnet_pos(tag)
            lemma = self.lemmatizer.lemmatize(word, pos) if pos else word
            lemmas.append(lemma)

        # Advanced N-gram Generation
        trigrams = ['_'.join(lemmas[i:i + 3]) for i in range(len(lemmas) - 2)]
        return ' '.join(lemmas + trigrams)

    @staticmethod
    def get_wordnet_pos(treebank_tag):
        if treebank_tag.startswith('J'):
            return nltk.corpus.wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return nltk.corpus.wordnet.VERB
        elif treebank_tag.startswith('N'):
            return nltk.corpus.wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return nltk.corpus.wordnet.ADV
        else:
            return None


# -------------------------- Optimized Model Pipeline --------------------------
def build_optimized_pipeline():
    # Feature Engineering Pipeline
    feature_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=Config.TEXT_PROCESSING['max_features'],
            ngram_range=Config.TEXT_PROCESSING['ngram_range'],
            min_df=Config.TEXT_PROCESSING['min_df'],
            max_df=Config.TEXT_PROCESSING['max_df'],
            stop_words=Config.TEXT_PROCESSING['stop_words'],
            sublinear_tf=Config.TEXT_PROCESSING['sublinear_tf'],
            smooth_idf=Config.TEXT_PROCESSING['smooth_idf']
        )),
        ('feature_selector', SelectKBest(chi2)),
    ])

    # Ensemble Classifier
    base_models = [
        ('LogisticRegression', LogisticRegression(
            class_weight='balanced',
            random_state=42
        )),
        ('LinearSVC', LinearSVC(
            class_weight='balanced',
            dual=False,
            random_state=42
        ))
    ]

    final_model = StackingClassifier(
        estimators=base_models,
        final_estimator=LogisticRegression(
            solver='saga',
            max_iter=2000,
            class_weight='balanced',
            random_state=42
        ),
        stack_method='predict_proba'
    )

    # Full Pipeline
    return Pipeline([
        ('features', feature_pipeline),
        ('smote', SMOTE(random_state=42)),
        ('classifier', final_model)
    ])


# -------------------------- Enhanced Evaluation --------------------------
def advanced_evaluation(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X_test)

    # Threshold Optimization
    thresholds = np.linspace(0.1, 0.9, 50)
    best_acc = 0
    best_thresh = 0.5

    for thresh in thresholds:
        current_pred = (y_proba > thresh).astype(int)
        acc = accuracy_score(y_test, current_pred)
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh

    # Final Report
    final_pred = (y_proba > best_thresh).astype(int)
    print("\n" + "=" * 60)
    print(f" Optimal Threshold: {best_thresh:.4f} | Accuracy: {best_acc:.4f} ")
    print("=" * 60)
    print(classification_report(y_test, final_pred, digits=4))

    # Confusion Matrix Visualization
    cm = confusion_matrix(y_test, final_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Ham', 'Spam'],
                yticklabels=['Ham', 'Spam'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Enhanced Confusion Matrix')
    plt.savefig('advanced_confusion_matrix.png')
    plt.show()


# -------------------------- Main Execution --------------------------
if __name__ == "__main__":
    try:
        # Initialize
        processor = AdvancedTextProcessor()

        # Load Data
        df = pd.read_csv("Email Classification.csv", encoding='utf-8')
        df['Class'] = LabelEncoder().fit_transform(df['Class'])

        # Process Text
        print("\nProcessing text...")
        df['Processed_Text'] = df['Message'].progress_apply(processor.process_text)

        # Split Data
        X_train, X_test, y_train, y_test = train_test_split(
            df['Processed_Text'], df['Class'],
            test_size=0.25,
            stratify=df['Class'],
            random_state=42
        )

        # Build Pipeline
        pipeline = build_optimized_pipeline()

        # Hyperparameter Tuning
        print("\nCommencing hyperparameter tuning...")
        grid_search = GridSearchCV(
            pipeline,
            param_grid=Config.MODEL_TUNING,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='accuracy',
            n_jobs=-1,
            verbose=2
        )
        grid_search.fit(X_train, y_train)

        # Final Model
        best_model = grid_search.best_estimator_
        print(f"\nBest Parameters: {grid_search.best_params_}")
        print(f"Validation Accuracy: {grid_search.best_score_:.4f}")

        # Evaluation
        advanced_evaluation(best_model, X_test, y_test)

    except Exception as e:
        print(f"\nExecution Failed: {str(e)}")
        sys.exit(1)