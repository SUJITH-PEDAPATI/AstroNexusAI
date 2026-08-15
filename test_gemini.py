import os
import sys

# Load .env so GEMINI_API_KEY is available without manual shell export
try:
    from dotenv import load_dotenv
    from pathlib import Path
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass  # python-dotenv not installed, fall back to shell env

try:
    from google import genai
except ImportError:
    print("ERROR: google-genai is not installed.")
    print("Run: pip install -U google-genai")
    sys.exit(1)


def main():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("ERROR: GEMINI_API_KEY is not set.")
        print()
        print("PowerShell:")
        print('$env:GEMINI_API_KEY="YOUR_API_KEY"')
        sys.exit(1)

    print("GEMINI_API_KEY: FOUND")
    print("Testing Gemini API...")

    try:
        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents="Reply with exactly: GEMINI_OK",
        )

        print()
        print("================================")
        print("Gemini API: SUCCESS")
        print("================================")
        print("Response:", response.text.strip())

    except Exception as e:
        print()
        print("================================")
        print("Gemini API: FAILED")
        print("================================")
        print(type(e).__name__)
        print(str(e))


if __name__ == "__main__":
    main()