from difflib import SequenceMatcher
from collections import Counter
import re

# -------------------------
# Clean text
# -------------------------
def clean_text(text):
    text = text.replace("Generated :", "")
    text = text.replace("User:", "")
    text = text.replace("Assistant:", "")
    
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()

# -------------------------
# Load files
# -------------------------
with open("fp_output.txt", "r", encoding="utf-8") as f:
    fp_raw = f.read()

with open("q4_output.txt", "r", encoding="utf-8") as f:
    q4_raw = f.read()

fp = clean_text(fp_raw)
q4 = clean_text(q4_raw)

# -------------------------
# DEBUG
# -------------------------
print("\n===== DEBUG =====")
print("FP (first 200 chars):", fp[:200])
print("Q4 (first 200 chars):", q4[:200])
print()

# -------------------------
# Tokenization (MOVE THIS UP)
# -------------------------
fp_tokens = fp.split()
q4_tokens = q4.split()

# -------------------------
# 1. Text similarity
# -------------------------
similarity = SequenceMatcher(None, fp, q4).ratio()

# -------------------------
# 2. Jaccard similarity (FIXED)
# -------------------------
fp_set = set(fp_tokens)
q4_set = set(q4_tokens)

jaccard = len(fp_set & q4_set) / max(len(fp_set | q4_set), 1)

# -------------------------
# 3. Token overlap accuracy
# -------------------------
common_tokens = Counter(fp_tokens) & Counter(q4_tokens)
token_acc = sum(common_tokens.values()) / max(len(fp_tokens), 1)

# -------------------------
# 4. Length difference
# -------------------------
len_diff = abs(len(fp_tokens) - len(q4_tokens))

# -------------------------
# 5. Unique tokens
# -------------------------
fp_unique = len(fp_set)
q4_unique = len(q4_set)

# -------------------------
# Results
# -------------------------
print("===== RESULTS =====")
print(f"Similarity Score   : {similarity:.4f}")
print(f"Jaccard Similarity : {jaccard:.4f}")
print(f"Token Accuracy     : {token_acc:.4f}")
print(f"Length Difference  : {len_diff}")
print(f"FP Unique Tokens   : {fp_unique}")
print(f"Q4 Unique Tokens   : {q4_unique}")