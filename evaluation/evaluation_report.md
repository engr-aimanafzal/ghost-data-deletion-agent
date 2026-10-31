# Evaluation Report

## Approach
Due to time constraints, I built a robust baseline pipeline using Python and the Groq API. The agent parses structured CSVs and unstructured evidence files. 

## Edge Cases Handled
1. **Missing Evidence:** If a `case_id` has no matching files in the evidence folder, the agent defaults to `INCOMPLETE`.
2. **API Rate Limits / Failures:** The script uses incremental saving (appending to CSV row-by-row). The `try/except` block catches API errors and tries a fallback list of models.
3. **Context Window Limits:** Evidence files are truncated to 3000 characters to prevent overflowing the model's context window.

## Improvements
I strictly validated that the AI's JSON output matches the required schema (`deletion_status` and `steps_json`). If the JSON parsing fails, the script defaults to a safe fallback so the pipeline never crashes.