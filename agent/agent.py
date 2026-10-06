import os
import requests
from dotenv import load_dotenv

load_dotenv()

MODEL = "gemini-3.5-flash-lite"
API_URL = "https://generativelanguage.googleapis.com/v1beta/models"


def _api_key():
    key = (
        os.getenv("GOOGLE_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or ""
    ).strip()

    if not key:
        raise RuntimeError(
            "Gemini API key is missing. "
            "Add GOOGLE_API_KEY=... to the project's .env file."
        )

    return key


def _call_gemini(prompt: str) -> str:
    url = f"{API_URL}/{MODEL}:generateContent"

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ]
    }

    try:
        response = requests.post(
            url,
            params={"key": _api_key()},
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )

        if response.status_code != 200:
            try:
                data = response.json()
                message = (
                    data
                    .get("error", {})
                    .get("message", response.text)
                )
            except Exception:
                message = response.text

            raise RuntimeError(
                f"Gemini API error {response.status_code}: {message}"
            )

        data = response.json()

        candidates = data.get("candidates", [])

        if not candidates:
            raise RuntimeError(
                "Gemini returned no candidates."
            )

        parts = (
            candidates[0]
            .get("content", {})
            .get("parts", [])
        )

        texts = []

        for part in parts:
            if isinstance(part, dict) and part.get("text"):
                texts.append(str(part["text"]))

        result = "\n".join(texts).strip()

        if not result:
            raise RuntimeError(
                "Gemini returned an empty response."
            )

        return result

    except requests.exceptions.Timeout:
        raise RuntimeError(
            "Gemini request timed out after 30 seconds."
        )

    except requests.exceptions.RequestException as exc:
        raise RuntimeError(
            f"Gemini network error: {exc}"
        ) from exc


def ask(prompt: str) -> str:
    return _call_gemini(prompt)


def generate_sql(question: str, schema: str) -> str:
    prompt = f"""
You are an expert SQLite SQL Data Analyst.

Database schema:
{schema}

User question:
{question}

Generate exactly ONE valid SQLite SELECT query.

Rules:
- Return ONLY SQL.
- No markdown.
- No explanation.
- SELECT only.
- Use only tables and columns in the schema.
- Do not invent columns.
"""

    result = ask(prompt)

    result = (
        result
        .replace("```sql", "")
        .replace("```SQL", "")
        .replace("```", "")
        .strip()
    )

    if not result.lower().startswith("select"):
        raise RuntimeError(
            "Gemini returned an invalid SQL response."
        )

    return result


def generate_insight(question: str, result) -> str:
    prompt = f"""
You are a professional Business Data Analyst.

Question:
{question}

SQL result:
{result}

Give 2-4 concise factual business insights.

Rules:
- Mention important numbers when present.
- Do not invent information.
- Focus on business meaning.
"""

    return ask(prompt)