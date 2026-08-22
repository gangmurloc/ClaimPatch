def normalize_claim_text(text: str) -> str:
    return " ".join(text.lower().strip().split())

