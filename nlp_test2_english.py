"""
Email classification main program (v2.3)
Although I haven't completed the improvement yet, I will upload it to my GitHub: https://github.com/Prodigitree
Function: A spam classification system based on Naive Bayes algorithm
"""
# --------------------------Environment Configuration--------------------------
# Basic Library
import os
import sys
import warnings
import numpy as np
# DataProcessing
import pandas as pd
# VIS
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import make_interp_spline
plt.rcParams.update({
    'savefig.dpi': 600,
    'font.sans-serif': 'Times New Roman',
    'axes.unicode_minus': False,
    'figure.figsize': (10, 6),
})
# TextProcessing
import re
import nltk
from nltk.stem import PorterStemmer
# MachineLearning
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV

# ConfigureWarningFiltering(Failure discovered in practical applications)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="sklearn")

os.environ["LOKY_PICKLER"] = "pickle"
warnings.filterwarnings("ignore", category=DeprecationWarning)
# -------------------------- ConstantDefinition --------------------------
BASIC_STOPWORDS = {
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
}

# -------------------------- CoreFunction --------------------------
def initialize_nltk():
    """
1. Multipath detection: User directory → Project directory
2. Automatic download: Automatically download when local resources are missing
3. Alternative solution: Enable basic stop words when download fails
!!!Warning, due to the inability to obtain stop words properly, offline BASIC_STOPWORDS will be used directly!!!
    """
    global STOPWORDS
    STOPWORDS = BASIC_STOPWORDS
    print(f"\n Warning!Use_BASIC_STOPWORDS({len(BASIC_STOPWORDS)})")


def clean_text(text):
    """
1. Cleaning: Remove non letter characters → Convert to lowercase → Remove spaces at both ends
2. Word segmentation: Divide text by spaces
3. Filtering: Remove stop words
4. Stemming: Porter Stemming Processing
    Args:
        text (str): OriginalText
    Returns:
        str: StandardizedText
    """
    # cleaning
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower().strip()

    # Segmentation And StopWordFiltering
    tokens = text.split()
    stems = [PorterStemmer().stem(w) for w in tokens]
    bigrams = [' '.join(pair) for pair in zip(stems[:-1], stems[1:])]
    return ' '.join(stems + bigrams)




def analyze_features(vectorizer, model, n_top=20):
    """
1. Obtain feature names (compatible with both old and new versions of sklearn)
2. Sort by logarithmic probability of spam category
3. Output TopN important features and their weights
    """
    global STOPWORDS
    features_names = vectorizer.get_feature_names()
    valid_mask = [word not in STOPWORDS for word in features_names]
    filtered_features = [(name, prob) for name, prob in
                        zip(features_names, model.feature_log_prob_[1])
                        if valid_mask[features_names.index(name)]]
    features = sorted(
        filtered_features,
        key=lambda x: (-x[1], -len(x[0]), x[0])
    )[:n_top]
    # DataAlignment
    vocab = [feat[0] for feat in features]
    weights = [feat[1] for feat in features]

    return vocab, weights


def evaluate_model(model, x_test, y_test, thresholds=np.arange(0.10, 1, 0.02)):
    """
Multi threshold model evaluation (version2)
1. Detailed indicator output (including accuracy/recall/F1 of each classification)
2. Optimal threshold recommendation based on weighted F1
3. Return the optimal threshold for future use
    """
    y_proba = model.predict_proba(x_test)[:, 1]

    best_score = 0

    reports = []

    # Evaluation Threshold
    for thresh in thresholds:
        y_pred = (y_proba > thresh).astype(int)
        report = classification_report(y_test, y_pred, output_dict=True)

        # Record Threshold Report
        reports.append((thresh, report))
        weighted_avg = report['weighted avg']
        current_f1 = weighted_avg['f1-score']

        # Update The bestscore
        if current_f1 > best_score:
            best_score = current_f1
            best_thresh = thresh

        # detailedIndicatorOutput
        print("\n" + "-" * 50)
        print(f" threshold {thresh:.2f} detailedIndicators ".center(45))
        print("-" * 50)
        print(f"spam distinguish:")
        print(
            f"  Accuracy: {report['1']['precision']:.3f} | Recall: {report['1']['recall']:.3f} | F1: {report['1']['f1-score']:.3f}")
        print(f"hamdistinguish:")
        print(
            f"  Accuracy: {report['0']['precision']:.3f} | Recall: {report['0']['recall']:.3f} | F1: {report['0']['f1-score']:.3f}")
        print(f"F1: {current_f1:.3f}")

    # best_threshold_recommendation
    print("\n" + "=" * 50)
    print(f" best_threshold_recommendation:{best_thresh:.2f} ".center(45))
    print(f" F1:{best_score:.3f} ".center(45))
    print("=" * 50)

    # AddDataCollectionBeforeReturning
    f1_scores = [report['weighted avg']['f1-score'] for _, report in reports]
    return best_thresh, thresholds, f1_scores


def plot_confusion_matrix(y_true, y_pred, class_names):
    """
Generate confusion matrix visualization
1. Blue color scheme
2. Display of numerical labels
3. Chinese character support
4. Automatically save to PNG format
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure()
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Greens',
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={'size': 12}
    )
    plt.title('EMAIL CLASSIFICATION CONFUSION MATRIX', fontweight='bold')
    plt.xlabel('PREDICTION LABEL')
    plt.ylabel('REAL LABEL')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()


def plot_threshold_analysis(thresholds, f1_scores, best_thresh):
    plt.figure()
    x_new = np.linspace(min(thresholds), max(thresholds), 300)
    spl = make_interp_spline(thresholds, f1_scores, k=3)
    y_smooth = spl(x_new)

    plt.plot(x_new, y_smooth, 'b-', linewidth=2)
    plt.scatter(thresholds, f1_scores, c='red', s=50, zorder=5)
    plt.axvline(best_thresh, color='green', linestyle='--', linewidth=1.5)
    focus_range = 0.1  #control_focus_range
    plt.xlim(0.10, 1.0)
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(0.05))  # Every 0.05 marks on the X-axis
    plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.02))  # Every 0.02 marks on the Y-axis

    plt.title('THRESHOLD SELECTION ANALYSIS', fontsize=14, pad=20)
    plt.xlabel('CLASSIFICATION THRESHOLD', fontsize=12)
    plt.ylabel('F1 SCORE', fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.text(best_thresh + 0.02, max(f1_scores) - 0.03,
             f'Best Threshold: {best_thresh:.2f}',
             fontsize=10, color='green')

    plt.tight_layout()
    plt.savefig('threshold_analysis.png')
    plt.show()


# -------------------------- MainProgram --------------------------
if __name__ == "__main__":
    try:
        print("\n" + "=" * 50)
        print(" EmailClassificationSystem v2.1 ".center(45))
        print("=" * 50)

        # Initialization
        initialize_nltk()
        print("\n" + "=" * 50)
        print(" DataLoading And Preprocessing ".center(45))
        print("=" * 50)
        df = pd.read_csv("Email Classification.csv", encoding="utf-8-sig")
        df["Class"] = LabelEncoder().fit_transform(df["Class"])
        print("DistributionOfDatasetCategories:\n", df["Class"].value_counts())

        # TextPreprocessing
        print("\n TextPreprocessing...")
        df["Cleaned_Message"] = df["Message"].apply(clean_text)

        # Feature Engineering
        print("\n In feature engineering processing...")
        tfidf = TfidfVectorizer(max_features=30000, ngram_range=(1, 2))
        min_df = 5,
        max_df = 0.85
        X = tfidf.fit_transform(df["Cleaned_Message"])
        y = df["Class"]

        # Add validation after feature engineering
        print("\nFeature space analysis:")
        print(f"Actual feature dimension: {X.shape[1]}")

        # Add binary grammar validity validation
        bigram_samples = [phrase for phrase in tfidf.get_feature_names() if ' ' in phrase][:10]
        print("Example binary grammar features:", bigram_samples)


        tfidf_feature_names = tfidf.get_feature_names()

        # DataPartitioning
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.3,
            random_state=42,
            #42 is the ultimate answer to the universe
            stratify=y
        )


        # DataDistributionStatistics
        def print_distribution(name, y_data):
            ham = (y_data == 0).sum()
            spam = (y_data == 1).sum()
            print(f"{name}:")
            print(f"ham: {ham} ({ham / len(y_data):.1%})")
            print(f"spam: {spam} ({spam / len(y_data):.1%})")

        print_distribution("\nTrainingSetDistribution", y_train)
        print_distribution("\nTestSetDistribution", y_test)

        print("\n DuringModelTraining...")
        # GridSearch Configuration best_alpha
        param_grid = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10]}
        grid_search = GridSearchCV(
            estimator=MultinomialNB(),
            param_grid=param_grid,
            cv=5,
            n_jobs=1
        )



        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_

        # Output best_alpha
        print(f"ModelTrainingCompleted，best_alpha: {grid_search.best_params_['alpha']}")

        # spam WordFrequencyStatistics
        print("\n CountingHighFrequencySpamKeywords...")
        spam_indices = y_train[y_train == 1].index
        spam_texts = df.loc[spam_indices, 'Cleaned_Message']
        spam_texts = spam_texts.apply(
            lambda x: ' '.join([w for w in x.split() if w not in STOPWORDS])
        )


        count_vec = CountVectorizer(
            vocabulary=tfidf_feature_names,
            ngram_range=(1, 2))
        spam_counts = count_vec.transform(spam_texts)
        spam_counts_array = spam_counts.sum(axis=0).A1

        # Extract spam text from the training set
        spam_indices = y_train[y_train == 1].index
        spam_texts = df.loc[spam_indices, 'Cleaned_Message']
        spam_texts = spam_texts.apply(
            lambda x: ' '.join([w for w in x.split() if w not in STOPWORDS])
        )
        feature_names = (count_vec.get_feature_names())  # ← Abandon compatibility processing and directly use the old version

        word_counts = zip(feature_names, spam_counts.sum(axis=0).tolist()[0])
        sorted_words = sorted(word_counts, key=lambda x: x[1], reverse=True)[:30]

        train_total = X_train.shape[0]
        freq_data = [
            (word, count, count / train_total)
            for word, count in sorted_words
        ]

        # Model evaluation
        evaluate_model(model, X_test, y_test)
        # best_threshold = evaluate_model(model, X_test, y_test)
        best_threshold, thresholds, f1_scores = evaluate_model(model, X_test, y_test)
        feature_vocab, feature_weights = analyze_features(tfidf, model, n_top=20)  # ← Modify to output 20 features
        # top_features = analyze_features(tfidf, model)

        # Use the optimal threshold for final prediction
        y_proba = model.predict_proba(X_test)[:, 1]
        final_pred = (y_proba > best_threshold).astype(int)

        # Generate the final evaluation report
        print("\n" + "=" * 50)
        print(f" Final evaluation report(Use threshold {best_threshold:.2f}) ".center(45))
        print("=" * 50)
        print(classification_report(y_test, final_pred))

        # Add test accuracy output
        print(f"\nTest set accuracy:{np.mean(final_pred == y_test):.4f}")

        # Generate confusion matrix using optimal threshold
        plot_confusion_matrix(y_test, final_pred, ['ham', 'spam'])

        count_vec = CountVectorizer(vocabulary=feature_vocab)
        spam_counts = count_vec.fit_transform(spam_texts)
        spam_counts_array = np.asarray(spam_counts.sum(axis=0)).flatten()

        # Debugging dimension verification
        #print(f"Feature dimension: {len(feature_vocab)},
        #print(f"Word frequency dimension: {spam_counts.sum(axis=0).shape[1]}")
        #print(f"Length of vocabulary column: {len(feature_vocab)}")
        #print(f"Weight column length: {len(feature_weights)}")
        #print(f"Word frequency statistics dimension: {spam_counts.sum(axis=0).shape}")

        # CreateDataBox
        df_merge = pd.DataFrame({
            'Vocabulary': feature_vocab,
            'Weight': feature_weights,
            'Count': spam_counts_array[:len(feature_vocab)],
            'Frequency': spam_counts_array[:len(feature_vocab)] / X_train.shape[0]
        }).head(20)

        # ConsoleOutput
        print("\n top20_high_frequency_spam_keywords:")
        print(f"{'Vocabulary':<15} | {'Weight':<8} | {'Count':<8} | {'Frequency':<8}")
        print("-" * 45)
        for idx, row in df_merge.iterrows():
            print(f"{row['Vocabulary']:15} | {row['Weight']:8.3f} | {row['Count']:8} | {row['Frequency']:8.3%}")

        # output2CSV
        df_merge.to_csv('top20_high_frequency_spam_keywords.csv', index=False, encoding='utf-8-sig')
        print(" data output to top20_high_frequency_spam_keywords.csv")

        # outpot png
        print("\n Generate_png")
        # (y_test, model.predict(X_test), ['ham', 'spam'])
        plot_threshold_analysis(thresholds, f1_scores, best_threshold)
        print("\n ProgramRunningEnds")

    except Exception as e:
        print(f"\n PROGRAM TERMINATED ABNORMALLY: {str(e)}")
        sys.exit(1)