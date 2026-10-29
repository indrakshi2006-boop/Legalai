# AEGIS: AI-Powered Legal Contract Review Assistant

AEGIS is a course prototype that gives a **preliminary** first reading of a contract PDF. It splits the contract into clauses, compares each clause with a 30-clause illustrative playbook, and labels it **Safe**, **Caution**, **Risky** or **Review required** using deterministic rules.

> **Preliminary review - not legal advice.** Labels are rule outputs for wording differences, not legal conclusions. The tool has not been evaluated for legal accuracy: there is no labelled evaluation, so no precision, recall or F1 figure exists.

AEGIS is built on the [firica/legalai](https://github.com/firica/legalai) Streamlit application. It includes a separate optional EU AI Act chat and a sample customer support chat page.

## Install

Python 3.12. On Windows (PowerShell), from the project folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

On macOS or Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Put your key in `.env` (never commit it; `.env` is git-ignored). A clean install was verified on Linux with Python 3.12.3; Windows and macOS installs have not been verified.

## Settings (.env)

| Name | Default | Meaning |
|---|---|---|
| `AEGIS_PROVIDER` | `openai` | `openai` or `gemini`. There is no automatic fallback between them. |
| `OPENAI_API_KEY` | - | Key for the OpenAI provider. |
| `OPENAI_MODEL` | `gpt-4o` | OpenAI chat model for comparisons and questions (temperature 0). |
| `GEMINI_API_KEY` | - | Key for the Gemini provider. |
| `GEMINI_CHAT_MODEL` | `gemini-3.8-flash` | Gemini chat model. |
| `GEMINI_MIN_SIMILARITY` | not set | Gemini cutoff for contract review and contract Q&A. Set this after `--inspect-scores`; the sample Customer Support Chat does not use vector retrieval. |
| `GEMINI_BROAD_K` | `8` | Passages in the first, broad Gemini search. |
| `OPENAI_TIMEOUT` | `60` | Request timeout in seconds (both providers). |
| `AEGIS_MAX_RETRIES` | `3` | Retries for rate limits, 5xx and network errors only (0-3). |
| `MAX_UPLOAD_MB` / `MAX_PDF_PAGES` | `15` / `40` | Upload limits. |

## Commands

Each command below is run from the project folder with the virtual environment active.

```bash
python embed.py playbook --preflight                 # offline: config, key yes/no, playbook, index state, call plan
python embed.py playbook --preflight --pdf docs/figures/synthetic-contract-v2.pdf   # adds the review call limits
python embed.py playbook                              # build the playbook index (OpenAI: 1 request; Gemini: 30)
python embed.py --inspect-scores --pdf PATH           # embeddings only, no chat: similarity scores per clause
python embed.py regulation                            # optional: download the EU AI Act and embed it with OpenAI
streamlit run app.py                                  # the app (localhost only)
```

`--preflight` makes no network call and exits 0 for PASS, 1 for FAIL. Building an index, reviewing, inspecting scores and asking questions call the provider's API and can incur a charge. The regulation index makes many embedding requests.

## What the app does

1. **Upload and check** a text-based PDF. Image-only (scanned) PDFs are rejected; there is no OCR. Nothing is sent anywhere at this step. A title, banner or parties block before the first clause is shown as **Preamble - not reviewed** and is never sent.
2. **Review.** The page shows the clause count and the hard call limit (two embedding queries and one chat call per clause, retries included), and needs a consent checkbox naming the provider. The button is disabled until the key is set and a matching index exists. Results are cached in the session, so a rerun never repeats calls. If the provider reports no credit or rejects the key, the remaining clauses are not sent and the finished ones are kept.
3. **Results.** Label counts, an overall description, a chart, filters, search and sort; a card per clause with the retrieved reference (id, title, similarity), the differences, a plain-English explanation and warnings; an annotated view of the contract text; CSV and HTML downloads that include the provider, models, date and the preliminary notice.

**Ask about this contract** uses **lexical** matching (shared words) over the session's clauses. Only when a passage matches are the question and those passages sent to the provider; otherwise the reply is "I can't answer that from the retrieved passages." and nothing is sent.

**Customer Support Chat** is a separate sample workflow. It routes questions to the bundled support notes and uses the selected Gemini model to draft a concise answer from those notes and the current conversation. It does not call OpenAI or use the contract playbook index. Supervisor-sensitive requests are marked pending; this prototype does not perform real approvals.

## Data and privacy

- Contract text leaves your computer only when you start a review (each clause goes to the selected provider for embeddings and comparison) or ask a question that matches a passage.
- The review does not write the contract text to disk or to any index. Results stay in the app's memory for the browser session.
- The app listens on `localhost` only. Streamlit usage statistics are off.
- This project does not implement encryption, deletion, anonymisation or separation between users.

**Free-tier Gemini:** Google currently lists `gemini-3.8-flash` and `gemini-embedding-001` with free-tier access. Free-tier rate limits still apply, and Google may use submitted content to improve its products. Use synthetic or non-confidential material. See Google's [current Gemini API pricing](https://ai.google.dev/gemini-api/docs/pricing) and [model list](https://ai.google.dev/gemini-api/docs/models).

**Gemini:** when `AEGIS_PROVIDER=gemini`, the app shows "Gemini API: use synthetic or non-confidential text only." Google's [Gemini API terms](https://ai.google.dev/gemini-api/terms) (checked 29 October 2026) say that content sent to unpaid services may be used to improve Google's products and read by human reviewers, and that paid services are not used that way. This code does not know which tier your key uses. The Gemini similarity cutoff is tuned on the synthetic sample only and is **not validated**. Gemini comparisons run at Google's recommended default temperature, so they may vary between runs.

## Tests

```bash
python -m pytest -q
```

The tests use a fake key, never read `.env`, and refuse outbound network connections, so they cannot call any API. They check program behaviour with fixtures; they do not measure legal accuracy. `python docs/build_report.py` rebuilds the project report and takes its test counts from a fresh run.

## Limits

- No live review has been recorded for this revision; the playbook index has not been built.
- The annotated view marks the extracted text, not the original PDF pages.
- Clause-by-clause comparison misses definitions or exceptions in other clauses.
- The playbook is original example wording, not a validated legal standard.
