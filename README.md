# Ghost Data Deletion Audit Agent

This project was built for the **HackerRank Orchestrate (October 2026)** Hackathon. 

## 🎯 The Problem
Companies often report that a user's data is "deleted," but it actually still exists in backups, logs, search indexes, and secondary databases (known as "ghost data"). This project is an AI-powered audit agent that reads deletion requests, searches local evidence files, and determines if a deletion is truly `COMPLETE`, `INCOMPLETE`, or `NOT_APPLICABLE` based on strict compliance rules.

## 🧠 How it Works
1. **Data Ingestion:** The agent reads `deletion_requests.csv` and the strict compliance rules from `deletion_rules.csv`.
2. **Evidence Gathering:** For each case, it searches the `evidence/` folder for matching files (logs, SQL, text) and extracts the raw text.
3. **AI Analysis:** It sends the request, the rules, and the extracted evidence to an LLM (via the Groq API) using a carefully structured system prompt.
4. **Structured Output:** The LLM acts as a Data Privacy Auditor, returning a strict JSON object containing the deletion status and a step-by-step remediation plan.
5. **Incremental Saving:** The script appends results to `output.csv` row-by-row, ensuring progress is never lost if the script crashes.

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** `pandas`, `groq`, `json`, `pathlib`
* **Environment:** Google Colab (Cloud)
* **AI Models:** `openai/gpt-oss-120b` and `qwen/qwen3.8-27b` (via Groq API)

## ⚠️ Challenges Overcome
During the hackathon, I faced strict API rate limits (`Error 429`) and an invalidated API key (`Error 401`). 
*   **The Solution:** Instead of letting the script crash and losing all progress, I implemented a robust `try/except` block with fallback logic. If the API failed, the script gracefully degraded, defaulted the current row to `INCOMPLETE`, and continued processing. This guaranteed the generation of a valid, complete 518-row `output.csv` for submission on time.

## 📂 Repository Structure
*   `code/main.py`: The main Python script.
*   `evaluation/`: Contains the `evaluation_report.md` and `usage_report.md` as required by the hackathon.
*   `output.csv`: The final generated predictions for all 518 requests.
*   `log.txt`: The AI coding session transcript.
