from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import sys
from io import StringIO
import traceback
import os

from google import genai
from google.genai import types

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CodeRequest(BaseModel):
    code: str

class CodeResponse(BaseModel):
    error: List[int]
    result: str


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


class ErrorAnalysis(BaseModel):
    error_lines: List[int]


def analyze_error_with_ai(code: str, traceback_text: str) -> List[int]:

    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY")
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


@app.post("/code-interpreter", response_model=CodeResponse)
def run_code(req: CodeRequest):

    execution = execute_python_code(req.code)

    if execution["success"]:
        return {"error": [], "result": execution["output"]}

    error_lines = analyze_error_with_ai(req.code, execution["output"])

    return {"error": error_lines, "result": execution["output"]}


# REQUIRED FOR RENDER
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
