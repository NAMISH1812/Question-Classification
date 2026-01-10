# train_and_evaluate_ablation.py
import wandb
from src.data_loader import load_dataset
from src.preprocess import preprocess_data
from src.train_model import evaluate_model
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy

# -----------------------------
# 1️⃣ Load dataset
# -----------------------------
print("📦 Loading dataset...")
train_df = load_dataset("MATH/train")
test_df = load_dataset("MATH/test")
train_df = preprocess_data(train_df)
test_df = preprocess_data(test_df)

print(f"✅ Loaded {len(train_df)} training and {len(test_df)} test samples.")

# -----------------------------
# 2️⃣ Define ablation experiments
# -----------------------------
experiments = [
    {"name": "baseline_logreg", "model": "logreg", "ngrams": (1, 2), "lemmatize": False, "stopwords": None},
    {"name": "stopwords_removed", "model": "logreg", "ngrams": (1, 2), "lemmatize": False, "stopwords": "english"},
    {"name": "lemmatization_added", "model": "logreg", "ngrams": (1, 2), "lemmatize": True, "stopwords": None},
    {"name": "svm_model", "model": "svm", "ngrams": (1, 2), "lemmatize": False, "stopwords": None},
    {"name": "trigram_tfidf", "model": "svm", "ngrams": (1, 3), "lemmatize": False, "stopwords": None},
]

nlp = spacy.load("en_core_web_sm")

# -----------------------------
# 3️⃣ Run each ablation separately
# -----------------------------
for exp in experiments:
    print(f"\n🚀 Running experiment: {exp['name']}")
    
    # Initialize W&B run separately per experiment
    wandb.init(project="math-question-classification", name=exp["name"], reinit=True)

    # Optional lemmatization
    if exp["lemmatize"]:
        print("🔠 Applying lemmatization with spaCy...")
        train_df["cleaned"] = train_df["cleaned"].apply(
            lambda t: " ".join([tok.lemma_ for tok in nlp(t)])
        )
        test_df["cleaned"] = test_df["cleaned"].apply(
            lambda t: " ".join([tok.lemma_ for tok in nlp(t)])
        )

    # Build TF-IDF
    vectorizer = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"(?u)\b[a-zA-Z0-9_+\-*/^=().]{1,}\b",
        ngram_range=exp["ngrams"],
        max_features=8000,
        stop_words=exp["stopwords"]
    )
    X_train = vectorizer.fit_transform(train_df["cleaned"])
    X_test = vectorizer.transform(test_df["cleaned"])

    # Choose model
    if exp["model"] == "logreg":
        model = LogisticRegression(max_iter=1000, class_weight='balanced')
    else:
        model = LinearSVC(class_weight='balanced')

    # Train
    model.fit(X_train, train_df["label"])

    # Evaluate
    accuracy = evaluate_model(model, X_test, test_df["label"])
    print(f"✅ {exp['name']} → Accuracy: {accuracy:.4f}")

    # Log result
    wandb.log({"accuracy": accuracy})
    wandb.finish()

print("\n🏁 All ablation experiments complete! Check your W&B dashboard for run comparisons.")
