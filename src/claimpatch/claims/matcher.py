def exact_normalized_match(a: str, b: str) -> bool:
    from claimpatch.claims.normalizer import normalize_claim_text

    return normalize_claim_text(a) == normalize_claim_text(b)

