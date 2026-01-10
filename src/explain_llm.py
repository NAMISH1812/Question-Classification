# src/explain_llm.py
import os
import google.genai as genai

def explain_question_with_gemini(question_text: str, topic: str, api_key: str = None):
    """
    Uses Google's Gemini 2.5 Pro model via the google-genai SDK
    to generate step-by-step, student-friendly math explanations.
    """

    #  Configure Gemini
    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Gemini API key not found. Set GEMINI_API_KEY as an environment variable or pass it explicitly.")

    #  Create Gemini client
    client = genai.Client(api_key=api_key)

    #  Build a clean tutoring prompt
    prompt = f"""
    You are a kind and patient high school math tutor.
    Solve and explain the following {topic} problem step-by-step in a clear, student-friendly way.

    Question:
    {question_text}

    Your answer should include:
    1. Step-by-step reasoning
    2. The final answer
    3. A short intuitive explanation
    """

    #  Generate using Gemini 2.5 Pro
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    #  Handle text safely
    if hasattr(response, "text") and response.text:
        return response.text.strip()
    elif hasattr(response, "candidates"):
        return response.candidates[0].content.parts[0].text
    else:
        return " No response generated."

