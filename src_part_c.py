import os
import re
import json
import random
import time
import sys
import pandas as pd
import numpy as np

# Ensure current directory is in path for local imports
sys.path.append(os.path.dirname(__file__))

# Safe import for mock_llm components
try:
    from mock_llm import mock_llm, TransientLLMError, est_tokens
except ImportError:
    import mock_llm as m_module
    mock_llm = getattr(m_module, "mock_llm", getattr(m_module, "MockLLM", None))
    TransientLLMError = getattr(m_module, "TransientLLMError", Exception)
    est_tokens = getattr(m_module, "est_tokens", lambda x: len(str(x).split()) * 1.3)

os.makedirs("outputs", exist_ok=True)
print("--- STARTING PART C: SUPPORT TICKET TRIAGE ---")

# ==========================================
# LOAD DATA
# ==========================================
tickets = pd.read_csv("data/raw/tickets.csv", keep_default_na=False)

try:
    segments = pd.read_csv("outputs/customer_segments.csv")
    print(f"Loaded customer segments: {len(segments)} rows.")
except FileNotFoundError:
    segments = pd.DataFrame(columns=["customer_id", "segment_name", "is_premium"])
    print("Warning: customer_segments.csv not found. Premium flags will default to False.")


# ==========================================
# TASK 1: Input Guardrails
# ==========================================
MAX_INPUT_TOKENS = 400

def check_input_guardrails(text):
    if not text or str(text).strip() == "":
        return "skipped_empty"
    if est_tokens(text) > MAX_INPUT_TOKENS:
        return "flagged_long"
    return "valid"


# ==========================================
# TASK 2: Redact Personal Information (PII)
# ==========================================
email_pattern = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
phone_pattern = re.compile(r"\b[6-9]\d{9}\b")
order_pattern = re.compile(r"ORD-\d{6}")

sub_counts = {"email": 0, "phone": 0, "order_id": 0}

def redact_text(text):
    global sub_counts
    txt = str(text)
    
    e_matches = email_pattern.findall(txt)
    sub_counts["email"] += len(e_matches)
    txt = email_pattern.sub("[EMAIL]", txt)
    
    p_matches = phone_pattern.findall(txt)
    sub_counts["phone"] += len(p_matches)
    txt = phone_pattern.sub("[PHONE]", txt)
    
    o_matches = order_pattern.findall(txt)
    sub_counts["order_id"] += len(o_matches)
    txt = order_pattern.sub("[ORDER_ID]", txt)
    
    return txt

tickets["redacted_text"] = tickets["text"].apply(redact_text)
total_substitutions = sum(sub_counts.values())

print(f"\nTask 2 - Redaction complete:")
print(f"  Emails substituted: {sub_counts['email']} (Reference: 55)")
print(f"  Phones substituted: {sub_counts['phone']} (Reference: 91)")
print(f"  Order IDs substituted: {sub_counts['order_id']} (Reference: 132)")
print(f"  Total substitutions: {total_substitutions} (Reference: 278)")

# Assertion checks: verify no PII left in redacted text
for idx, row in tickets.iterrows():
    rt = row["redacted_text"]
    assert not email_pattern.search(rt), f"Email found in redacted text at row {idx}"
    assert not phone_pattern.search(rt), f"Phone found in redacted text at row {idx}"
    assert not order_pattern.search(rt), f"Order ID found in redacted text at row {idx}"
print("  Assertion Passed: Zero PII remaining in redacted texts.")


# ==========================================
# TASK 3: Two Prompts (Zero-Shot & Few-Shot)
# ==========================================
ALLOWED_CATEGORIES = ["billing", "delivery", "technical", "fraud"]

ZERO_SHOT_SYSTEM_PROMPT = """CLASSIFY the support ticket into one of these four allowed categories: billing, delivery, technical, fraud.
You must reply in valid JSON-only format with the exact keys: {"category": "...", "reason": "..."}."""

FEW_SHOT_SYSTEM_PROMPT = """CLASSIFY the support ticket into one of these four allowed categories: billing, delivery, technical, fraud.
You must reply in valid JSON-only format with the exact keys: {"category": "...", "reason": "..."}.

EXAMPLES:
- Ticket: "My account was charged twice for order [ORDER_ID]." -> JSON: {"category": "billing", "reason": "Customer charged multiple times."}
- Ticket: "Someone unauthorized placed an order using my card details." -> JSON: {"category": "fraud", "reason": "Unauthorized transaction reported."}"""


# ==========================================
# TASK 4: Resilient Calls with Backoff & Jitter
# ==========================================
BASE_DELAY = 0.05
MAX_RETRIES = 3

def call_mock_llm_with_retry(system_prompt, user_text, ticket_id):
    for attempt in range(MAX_RETRIES):
        try:
            response = mock_llm(system_prompt, user_text)
            return response, 0
        except TransientLLMError as e:
            wait_time = BASE_DELAY * (2 ** attempt) * random.uniform(0.9, 1.15)
            print(f"retry ticket={ticket_id} retry={attempt} wait={wait_time:.4f}s")
            time.sleep(wait_time)
        except Exception as e:
            if "Transient" in type(e).__name__ or "Error" in type(e).__name__:
                wait_time = BASE_DELAY * (2 ** attempt) * random.uniform(0.9, 1.15)
                print(f"retry ticket={ticket_id} retry={attempt} wait={wait_time:.4f}s")
                time.sleep(wait_time)
            else:
                raise e
    return None, MAX_RETRIES


# ==========================================
# TASK 5 & 6: Run Pipeline & Validate Answers
# ==========================================
def run_pipeline(system_prompt, token_budget=None):
    if hasattr(mock_llm, "reset"):
        mock_llm.reset()
        
    results = []
    running_tokens = 0
    
    for idx, row in tickets.iterrows():
        tid = row["ticket_id"]
        txt = row["text"]
        red_txt = row["redacted_text"]
        true_cat = row["true_category"]
        
        # Guardrails check
        g_status = check_input_guardrails(txt)
        if g_status != "valid":
            results.append({
                "ticket_id": tid, "customer_id": row["customer_id"], "status": g_status,
                "category": "", "reason": "", "tokens_used": 0, "true_category": true_cat, "text": red_txt
            })
            continue
            
        # Token Budget check
        if token_budget is not None and running_tokens >= token_budget:
            results.append({
                "ticket_id": tid, "customer_id": row["customer_id"], "status": "budget_exhausted",
                "category": "", "reason": "", "tokens_used": 0, "true_category": true_cat, "text": red_txt
            })
            continue
            
        # Resilient Model Call
        response, _ = call_mock_llm_with_retry(system_prompt, red_txt, tid)
        if response is None:
            results.append({
                "ticket_id": tid, "customer_id": row["customer_id"], "status": "flagged_invalid_output",
                "category": "", "reason": "Persistent LLM failure", "tokens_used": 0, "true_category": true_cat, "text": red_txt
            })
            continue
            
        # Extract token usage safely
        tokens = 0
        if hasattr(response, "usage") and hasattr(response.usage, "total_tokens"):
            tokens = response.usage.total_tokens
        running_tokens += tokens
        
        reply_content = getattr(response, "content", str(response)).strip()
        status = "flagged_invalid_output"
        pred_cat = ""
        reason = ""
        
        # 1. Try JSON parsing
        try:
            parsed = json.loads(reply_content)
            if isinstance(parsed, dict) and "category" in parsed:
                cat_val = str(parsed["category"]).strip().lower()
                if cat_val in ALLOWED_CATEGORIES:
                    status = "classified_json"
                    pred_cat = cat_val
                    reason = parsed.get("reason", "")
        except json.JSONDecodeError:
            pass
            
        # 2. Fallback to re.findall keyword extraction if JSON fails
        if status != "classified_json":
            words = re.findall(r"[a-z]+", reply_content.lower())
            found_cats = [w for w in words if w in ALLOWED_CATEGORIES]
            if len(found_cats) == 1:
                status = "classified_keyword"
                pred_cat = found_cats[0]
                reason = "Extracted via keyword fallback"
                
        results.append({
            "ticket_id": tid, "customer_id": row["customer_id"], "status": status,
            "category": pred_cat, "reason": reason, "tokens_used": tokens, "true_category": true_cat, "text": red_txt
        })
        
    return pd.DataFrame(results)

print("\n--- Running Zero-Shot Pipeline ---")
df_zero = run_pipeline(ZERO_SHOT_SYSTEM_PROMPT)

print("\n--- Running Few-Shot Pipeline ---")
df_few = run_pipeline(FEW_SHOT_SYSTEM_PROMPT)

def evaluate_run(df, name):
    classified = df[df["status"].isin(["classified_json", "classified_keyword"])]
    correct = (classified["category"] == classified["true_category"]).sum()
    acc = (correct / len(classified)) * 100 if len(classified) > 0 else 0
    total_tok = classified["tokens_used"].sum()
    tok_per_ticket = total_tok / len(classified) if len(classified) > 0 else 0
    
    print(f"\n--- {name} Evaluation ---")
    print(f"Classified tickets count: {len(classified)} (Reference: 147)")
    print(f"Accuracy: {acc:.1f}% (Correct: {correct}/{len(classified)})")
    print(f"Total tokens used: {total_tok} | Tokens per classified ticket: {tok_per_ticket:.2f}")
    print("Status counts:", df["status"].value_counts().to_dict())

evaluate_run(df_zero, "Zero-Shot")
evaluate_run(df_few, "Few-Shot")

# Zero-Shot Confusion Table
classified_zero = df_zero[df_zero["status"].isin(["classified_json", "classified_keyword"])]
conf_matrix = pd.crosstab(classified_zero["true_category"], classified_zero["category"], rownames=["True"], colnames=["Predicted"])
print("\nZero-Shot Confusion Matrix:")
print(conf_matrix)


# ==========================================
# TASK 7: Budget Demo & Final Routing Output
# ==========================================
print("\n--- Running Budget Demo (Zero-Shot with Budget = 6000) ---")
df_budget = run_pipeline(ZERO_SHOT_SYSTEM_PROMPT, token_budget=6000)
exhausted_count = (df_budget["status"] == "budget_exhausted").sum()
print(f"Budget demo finished. Exhausted count: {exhausted_count}")
df_budget.to_csv("outputs/triage_budget_demo.csv", index=False)

print("\n--- Generating Final Production Routing (outputs/triage_results.csv) ---")
df_final = run_pipeline(FEW_SHOT_SYSTEM_PROMPT, token_budget=100000)

df_final = df_final.merge(segments[["customer_id", "segment_name", "is_premium"]], on="customer_id", how="left")
df_final["is_premium"] = df_final["is_premium"].fillna(False)
df_final["segment_name"] = df_final["segment_name"].fillna("")

def assign_routing(row):
    status = row["status"]
    cat = row["category"]
    is_prem = row["is_premium"]
    
    if status in ["skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"]:
        return "human_review", "NORMAL"
    elif cat == "fraud":
        return "security_team", "HIGH"
    elif is_prem:
        return "premium_desk", "HIGH"
    else:
        return "support_queue", "NORMAL"

routes_and_priorities = df_final.apply(assign_routing, axis=1)
df_final["route"] = [r[0] for r in routes_and_priorities]
df_final["priority"] = [r[1] for r in routes_and_priorities]

output_columns = ["ticket_id", "customer_id", "status", "category", "segment_name", "priority", "route", "tokens_used"]
for idx, row in df_final.iterrows():
    if row["status"] in ["skipped_empty", "flagged_long", "budget_exhausted"]:
        df_final.loc[idx, "category"] = ""
        df_final.loc[idx, "segment_name"] = ""

df_final_export = df_final[output_columns]
df_final_export.to_csv("outputs/triage_results.csv", index=False)

print(f"outputs/triage_results.csv generated successfully with {len(df_final_export)} rows!")
print("PART C COMPLETED SUCCESSFULLY!")