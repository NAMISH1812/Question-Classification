
import wandb
from src.data_loader import load_dataset
from src.preprocess import preprocess_data, get_vectorizer
from src.train_model import train_model, evaluate_model
from src.explain_llm import explain_question_with_gemini
import google.genai as genai
import os


print(" Loading dataset...")
train_df = load_dataset("MATH/train")
test_df = load_dataset("MATH/test")

print(f" Loaded {len(train_df)} training and {len(test_df)} test samples.")


print(" Preprocessing text data...")
train_df = preprocess_data(train_df)
test_df = preprocess_data(test_df)
print(" Sample cleaned questions:")
print(train_df["cleaned"].head(5).tolist())


print(" Converting text to TF-IDF features...")
X_train, X_test, vectorizer = get_vectorizer(train_df, test_df)


print(" Training logistic regression model...")
model = train_model(X_train, train_df["label"], model_name="logreg")


print(" Evaluating model...")
accuracy = evaluate_model(model, X_test, test_df["label"])


try:
    wandb.init(project="math-question-classification", name="gemini-vs-code-run")
    wandb.log({"accuracy": accuracy})
except Exception as e:
    print(" W&B logging skipped:", e)


print("\n Generating student-friendly explanation using Gemini...")

# Set up Gemini client
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError(" GEMINI_API_KEY not set. Please set your API key in the environment variables.")

client = genai.Client(api_key=api_key)

sample = test_df.sample(1).iloc[0]
question = sample["question_text"]
topic = sample["label"]

prompt = f"""
You are a kind and patient high school math tutor.
Solve and explain the following {topic} question step-by-step in a clear, student-friendly way.

Question:
{question}

Include:
1. Step-by-step reasoning
2. The final answer
3. A short intuitive explanation
"""


try:
    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=prompt,
    )
except genai.errors.ClientError as e:
    if "RESOURCE_EXHAUSTED" in str(e):
        print(" Pro quota exhausted — switching to gemini-2.5-flash.")
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
    else:
        raise


print(f"\n Question ({topic}): {question}\n")
if hasattr(response, "text") and response.text:
    print(" Gemini Explanation:\n")
    print(response.text.strip())
elif hasattr(response, "candidates"):
    print(" Gemini Explanation:\n")
    print(response.candidates[0].content.parts[0].text)
else:
    print(" No explanation generated.")

print("\n Pipeline complete.")
