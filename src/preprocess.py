import pandas as pd
import re
import unicodedata
from sklearn.feature_extraction.text import TfidfVectorizer



def clean_text(text: str) -> str:
    """
    Cleans math problem text while keeping mathematical meaning.
    Converts LaTeX symbols, normalizes Unicode, and preserves variables and operators.
    """

    if not isinstance(text, str):
        return ""

    # Normalize Unicode (ensures consistent symbols for math)
    text = unicodedata.normalize("NFKC", text).lower()

    #  Convert common LaTeX tokens to readable words ---
    latex_to_words = {
        r"\\frac": " fraction ",
        r"\\sqrt": " square_root ",
        r"\\pi": " pi ",
        r"\\sum": " summation ",
        r"\\int": " integral ",
        r"\\theta": " theta ",
        r"\\alpha": " alpha ",
        r"\\beta": " beta ",
        r"\\gamma": " gamma ",
        r"\\times": " times ",
        r"\\div": " divide ",
        r"\\leq": " less_equal ",
        r"\\geq": " greater_equal ",
        r"\\neq": " not_equal ",
        r"\\sin": " sin ",
        r"\\cos": " cos ",
        r"\\tan": " tan ",
        r"\\log": " log ",
        r"\\ln": " ln ",
    }

    for k, v in latex_to_words.items():
        text = re.sub(k, v, text)

    #  Handle fractions like \frac{a}{b} -> a over b ---
    text = re.sub(r"\\frac\{([^\{\}]+)\}\{([^\{\}]+)\}", r"\1 over \2", text)

    #  Remove LaTeX markup (dollars, braces, backslashes) ---
    text = re.sub(r"[{}$\\]", " ", text)

    #  Replace superscripts/subscripts ---
    superscript_map = str.maketrans({
        "²": "2", "³": "3", "⁴": "4", "⁵": "5", "⁶": "6",
        "⁷": "7", "⁸": "8", "⁹": "9", "⁰": "0", "¹": "1"
    })
    subscript_map = str.maketrans({
        "₀": "_0", "₁": "_1", "₂": "_2", "₃": "_3", "₄": "_4",
        "₅": "_5", "₆": "_6", "₇": "_7", "₈": "_8", "₉": "_9"
    })

    text = text.translate(superscript_map)
    text = text.translate(subscript_map)

    #  make superscripts more explicit (x2 -> x^2)
    text = re.sub(r"([a-z])([0-9])", r"\1^\2", text)

    #  Keep relevant characters (letters, digits, math ops) ---
    text = re.sub(r"[^a-z0-9^_+\-*/=()., ]", " ", text)

    #  Collapse extra spaces ---
    text = re.sub(r"\s+", " ", text).strip()

    return text



def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a 'cleaned' column for model input.
    Detects 'problem' or 'question_text' fields automatically.
    """

    # Detect which column to use
    if "problem" in df.columns:
        text_col = "problem"
    elif "question_text" in df.columns:
        text_col = "question_text"
    else:
        raise KeyError(" Expected a text column ('problem' or 'question_text') not found in DataFrame.")

    # Fill missing and clean
    df[text_col] = df[text_col].fillna("").astype(str)
    df["cleaned"] = df[text_col].apply(clean_text)

    # Drop rows that are empty after cleaning
    df = df[df["cleaned"].str.strip() != ""].reset_index(drop=True)
    return df



def get_vectorizer(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Builds TF-IDF features for train/test data.
    Keeps math tokens (no stopword removal).
    """

    if "cleaned" not in train_df.columns or "cleaned" not in test_df.columns:
        raise KeyError(" Both DataFrames must contain a 'cleaned' column. Run preprocess_data() first.")

    if train_df["cleaned"].str.strip().eq("").all():
        raise ValueError(" All training documents are empty after preprocessing. Check regex or input data.")

    vectorizer = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"(?u)\b[a-zA-Z0-9_+\-*/^=().]{1,}\b",  
        max_features=8000,
        ngram_range=(1, 2),
        stop_words=None
    )

    X_train = vectorizer.fit_transform(train_df["cleaned"])
    X_test = vectorizer.transform(test_df["cleaned"])

    return X_train, X_test, vectorizer
