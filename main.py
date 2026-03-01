from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import sys
from io import StringIO
import traceback

from google import genai
from google.genai import types

# =========================
# FastAPI App
# =========================
app = FastAPI()

# =========================
# Request Schema
# =========================
class CodeRequest(BaseModel):
    code: str

# =========================
# Response Schema
# =========================
class CodeResponse(BaseModel):
    error: List[int]
    result: str


# =========================
# Tool Function
# =========================
def execute_python_code(code: str) -> dict:
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    try:
        exec(code)
        output = sys.stdout.getvalue()
        return {"success": True, "output": output}

    except Exception:
        output = traceback.format_exc()
        return {"success": False, "output": output}

    finally:
        sys.stdout = old_stdout


# =========================
# AI Structured Output Model
# =========================
class ErrorAnalysis(BaseModel):
    error_lines: List[int]


# =========================
# Gemini Error Analyzer
# =========================
def analyze_error_with_ai(code: str, traceback_text: str) -> List[int]:

    client = genai.Client(
        api_key="AIzaSyCt3fcMslCKU169JjkZWLf58fRdVyqUq0w"
    )

    prompt = f"""
Analyze this Python code and its error traceback.
Identify the exact line number(s) where the error occurred.

CODE:
{code}

TRACEBACK:
{traceback_text}

Return ONLY JSON:
{{ "error_lines": [line_numbers] }}
"""

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "error_lines": types.Schema(
                        type=types.Type.ARRAY,
                        items=types.Schema(type=types.Type.INTEGER)
                    )
                },
                required=["error_lines"]
            )
        )
    )

    parsed = ErrorAnalysis.model_validate_json(response.text)
    return parsed.error_lines


# =========================
# API Endpoint
# =========================
@app.post("/code-interpreter", response_model=CodeResponse)
def run_code(req: CodeRequest):

    execution = execute_python_code(req.code)

    # If success → return output directly
    if execution["success"]:
        return {
            "error": [],
            "result": execution["output"]
        }

    # If error → analyze with AI
    error_lines = analyze_error_with_ai(
        req.code,
        execution["output"]
    )

    return {
        "error": error_lines,
        "result": execution["output"]
    }