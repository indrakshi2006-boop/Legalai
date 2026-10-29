import json as JS
import os as OS
from pathlib import Path

# Chroma reads this setting from the environment; keep its product telemetry off
# for every client in this process (playbook, regulation, tests).
OS.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

ARTICLES_FILE ='articles.json'
ARTICLES_FOLDER = 'articles' 
# Absolute, so the app and embed.py find the same index whatever folder they start in.
DB_FOLDER = str(Path(__file__).resolve().parent / 'chroma_storage')
DATA_FOLDER = 'data'
PLAYBOOK_FILE = str(Path(__file__).resolve().parent / 'playbook.json')
PLAYBOOK_COLLECTION = 'playbook_clauses'
DOCUMENT_COLLECTION = 'collection_1'
# Same model for playbook indexing and document retrieval.
EMBEDDING_MODEL = 'text-embedding-ada-002'
DEFAULT_CHAT_MODEL = 'gpt-4o'
SEVERITY_LEVELS = ('low', 'medium', 'high')
PROJECT_ROOT = Path(__file__).resolve().parent
KEY_NAMES = ('OPENAI_API_KEY', 'GEMINI_API_KEY')


def load_project_env():
    """Load .env from the project root without replacing a real key already set.

    Set AEGIS_NO_DOTENV=1 to skip the file entirely (the test suite does this,
    so tests never read a real key).
    """
    if OS.getenv("AEGIS_NO_DOTENV", "").strip() == "1":
        return
    from dotenv import load_dotenv

    for name in KEY_NAMES:
        # An empty inherited value still counts as set, so dotenv would not replace it.
        if name in OS.environ and not OS.environ[name].strip():
            OS.environ.pop(name, None)
    load_dotenv(PROJECT_ROOT / ".env", override=False)


def comparison_model_name():
    """Chat model for clause comparison. Override with OPENAI_MODEL."""
    configured = OS.getenv("OPENAI_MODEL", DEFAULT_CHAT_MODEL).strip()
    return configured or DEFAULT_CHAT_MODEL


def run_info():
    """Provider and models that produce a review. Stored on every result row."""
    return {
        "provider": "openai",
        "embedding_model": EMBEDDING_MODEL,
        "chat_model": comparison_model_name(),
    }


def comparison_timeout_seconds():
    """Request timeout in seconds. Override with OPENAI_TIMEOUT."""
    configured = OS.getenv("OPENAI_TIMEOUT", "60").strip()
    try:
        value = float(configured)
    except ValueError:
        return 60.0
    if value <= 0:
        return 60.0
    return value


def max_upload_bytes():
    """Maximum PDF upload size. Override with MAX_UPLOAD_MB."""
    configured = OS.getenv("MAX_UPLOAD_MB", "15").strip()
    try:
        megabytes = float(configured)
    except ValueError:
        megabytes = 15
    if megabytes <= 0:
        megabytes = 15
    return int(megabytes * 1024 * 1024)


def max_pdf_pages():
    """Maximum PDF page count. Override with MAX_PDF_PAGES."""
    configured = OS.getenv("MAX_PDF_PAGES", "40").strip()
    try:
        pages = int(configured)
    except ValueError:
        return 40
    return pages if pages > 0 else 40
PLAYBOOK_CATEGORIES = (
    'Termination',
    'Liability',
    'Indemnity',
    'Payment',
    'Confidentiality',
    'Auto-renewal',
    'Intellectual property ownership',
    'Non-compete',
    'Governing law',
    'Dispute resolution',
)
# Parliament host returns an empty WAF challenge to non-browser clients.
EUROPEAN_ACT_URL='https://artificialintelligenceact.eu/wp-content/uploads/2024/04/TA-9-2024-0138_EN.pdf'

# load articles data from file_name (.json)
def load_articles(file_name) -> list:
    result = []
    if OS.path.exists(file_name):
        with open(file_name, 'r') as file:
            try:
                result = JS.load(file)
            except JS.JSONDecodeError:
                print("File exists but is not valid JSON. Returning empty object.")
    else:
        with open(file_name, 'w') as file:
            JS.dump('[{}]', file)
        print(f"File '{file_name}' did not exist and was created.")
        OS.mkdir('articles')
        print("'articles' directory was created")
    
    return result 

# save articles data to file_name (.json)
def save_articles(file_name, data):
    try:
        with open(file_name, 'w') as file:
            JS.dump(data, file, indent=4)
            print(f"Data successfully saved to '{file_name}'.")    
    except Exception as e:
        print(f"Error: trying to save articles data [{e}]")

# save articles content to individual files
def save_article_content(file_name, content):
    try:
        with open(file_name, 'w') as file:
            file.write(content)
    except IOError as e:
        print(f"An IOError occurred: {e.strerror}")
    except Exception as e:
        print(f"Error: {e}")
    else:
        print(f"Content successfully written to '{file_name}'.")

# load article body text from file 
def load_article_content(file_name):
    result = ''
    try:
        with open(file_name, 'r') as file:
            result = file.read()
    except Exception as e:
        print(f"An unexpected error occurred while reading content file '{file_name}': {e}")
    
    return result