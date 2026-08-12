import random
from typing import Dict, Iterable, List, Optional, Tuple

from decap.patch.generator import build_oracle_patch
from decap.schemas.claims import ClaimNode
from decap.schemas.evidence import EvidenceRecord
from decap.schemas.graph import DependencyEdge
from decap.schemas.results import AnswerVersion, SyntheticInstance
from decap.schemas.updates import EvidenceUpdate, ImpactLabel


def _category(diff: int) -> str:
    if diff >= 8:
        return "substantially outperforms"
    if diff >= 3:
        return "outperforms"
    if diff >= 1:
        return "slightly outperforms"
    if diff == 0:
        return "matches"
    return "underperforms"


def _claim(
    claim_id: str,
    answer_id: str,
    version: int,
    text: str,
    claim_type: str,
    evidence_ids: List[str],
    metadata: Dict[str, object],
) -> ClaimNode:
    return ClaimNode(
        claim_id=claim_id,
        answer_id=answer_id,
        version=version,
        text=text,
        normalized_form=text.lower(),
        claim_type=claim_type,  # type: ignore[arg-type]
        evidence_ids=evidence_ids,
        valid_from=version,
        metadata=metadata,
    )


def _evidence(entity: str, version: int, attr: str, value: object) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=f"e_{entity}_{attr}_v{version}",
        entity_id=entity,
        version=version,
        text=f"{entity} {attr.replace('_', ' ')} is {value}.",
        source_uri=f"synthetic://benchmark/{entity}/{attr}/v{version}",
        valid_from=str(version),
        metadata={"attribute": attr, "value": value},
    )


def _answer(
    instance_id: str,
    answer_id: str,
    version: int,
    model_a: str,
    model_b: str,
    a_acc: int,
    b_acc: int,
    split_name: str,
    evidence: List[EvidenceRecord],
    a_evidence_version: Optional[int] = None,
    b_evidence_version: int = 0,
    report_version: Optional[int] = None,
    parent_version: Optional[int] = None,
    patch_id: Optional[str] = None,
) -> AnswerVersion:
    diff = a_acc - b_acc
    category = _category(diff)
    a_ev = version if a_evidence_version is None else a_evidence_version
    report_v = version if report_version is None else report_version
    claims = [
        _claim(
            "c_a_acc",
            answer_id,
            version,
            f"{model_a} achieved {a_acc}% accuracy.",
            "numeric",
            [f"e_{instance_id}_a_accuracy_v{a_ev}"],
            {"slot": "a_accuracy", "value": a_acc},
        ),
        _claim(
            "c_b_acc",
            answer_id,
            version,
            f"{model_b} achieved {b_acc}% accuracy.",
            "numeric",
            [f"e_{instance_id}_b_accuracy_v{b_evidence_version}"],
            {"slot": "b_accuracy", "value": b_acc},
        ),
        _claim(
            "c_diff",
            answer_id,
            version,
            f"{model_a} exceeded {model_b} by {diff} percentage points.",
            "numeric",
            [f"e_{instance_id}_a_accuracy_v{a_ev}", f"e_{instance_id}_b_accuracy_v{b_evidence_version}"],
            {"slot": "difference", "value": diff, "formula": "a_accuracy-b_accuracy"},
        ),
        _claim(
            "c_comparison",
            answer_id,
            version,
            f"{model_a} {category} {model_b}.",
            "comparative",
            [f"e_{instance_id}_a_accuracy_v{a_ev}", f"e_{instance_id}_b_accuracy_v{b_evidence_version}"],
            {"slot": "comparison", "value": category, "depends_on": "difference"},
        ),
        _claim(
            "c_split",
            answer_id,
            version,
            f"Both models were evaluated on the {split_name} split.",
            "factual",
            [f"e_{instance_id}_split_v0"],
            {"slot": "split", "value": split_name},
        ),
        _claim(
            "c_citation",
            answer_id,
            version,
            f"The accuracy numbers are cited from benchmark report version {report_v}.",
            "citation_only",
            [f"e_{instance_id}_a_accuracy_v{a_ev}", f"e_{instance_id}_b_accuracy_v{b_evidence_version}"],
            {"slot": "citation", "value": report_v},
        ),
    ]
    deps = [
        DependencyEdge(
            source_claim_ids=["c_a_acc", "c_b_acc"],
            target_claim_id="c_diff",
            dependency_type="numeric",
            rule="difference = a_accuracy - b_accuracy",
        ),
        DependencyEdge(
            source_claim_ids=["c_diff"],
            target_claim_id="c_comparison",
            dependency_type="comparative",
            rule="comparison category from difference threshold",
        ),
        DependencyEdge(
            source_claim_ids=["c_a_acc", "c_b_acc"],
            target_claim_id="c_citation",
            dependency_type="citation",
            rule="citation binds current metric evidence",
        ),
    ]
    rendered = (
        f"{model_a} achieved {a_acc}% accuracy, while {model_b} achieved {b_acc}% "
        f"on the {split_name} split. {model_a} exceeded {model_b} by {diff} "
        f"percentage points and {category} {model_b}. "
        f"The accuracy numbers are cited from benchmark report version {report_v}."
    )
    return AnswerVersion(
        answer_id=answer_id,
        version=version,
        question=f"How does {model_a} compare with {model_b} on the benchmark?",
        rendered_text=rendered,
        claims=claims,
        dependencies=deps,
        evidence=evidence,
        parent_version=parent_version,
        applied_patch_id=patch_id,
    )


def _impact_labels(
    old_a: int,
    new_a: int,
    old_b: int,
    new_b: int,
    changed_attribute: Optional[str],
    citation_refresh: bool = False,
) -> List[ImpactLabel]:
    old_diff = old_a - old_b
    new_diff = new_a - new_b
    old_category = _category(old_diff)
    new_category = _category(new_diff)
    a_changed = new_a != old_a
    b_changed = new_b != old_b
    citation_changed = citation_refresh or a_changed or b_changed
    labels = [
        ImpactLabel(
            claim_id="c_a_acc",
            state="MUST_CHANGE" if a_changed else "STILL_VALID",
            direct=changed_attribute == "a_accuracy",
            reason="Model A accuracy evidence changed." if a_changed else "Model A accuracy value remains valid.",
            gold_operation="REPLACE" if a_changed else None,
        ),
        ImpactLabel(
            claim_id="c_b_acc",
            state="MUST_CHANGE" if b_changed else "STILL_VALID",
            direct=changed_attribute == "b_accuracy",
            reason="Model B accuracy evidence changed." if b_changed else "Model B accuracy value remains valid.",
            gold_operation="REPLACE" if b_changed else None,
        ),
        ImpactLabel(
            claim_id="c_diff",
            state="MUST_CHANGE" if new_diff != old_diff else "STILL_VALID",
            direct=False,
            reason="Difference depends on Model A and Model B accuracy.",
            gold_operation="REPLACE" if new_diff != old_diff else None,
        ),
        ImpactLabel(
            claim_id="c_comparison",
            state="MUST_CHANGE" if new_category != old_category else "STILL_VALID",
            direct=False,
            reason="Qualitative comparison changes only if threshold category changes.",
            gold_operation="REPLACE" if new_category != old_category else None,
        ),
        ImpactLabel(
            claim_id="c_split",
            state="STILL_VALID",
            direct=False,
            reason="Dataset split evidence is unchanged.",
        ),
        ImpactLabel(
            claim_id="c_citation",
            state="MUST_CHANGE" if citation_changed else "STILL_VALID",
            direct=citation_refresh,
            reason=(
                "Citation binding must move to the updated benchmark report."
                if citation_changed
                else "Citation binding remains valid."
            ),
            gold_operation="REBIND" if citation_changed else None,
        ),
    ]
    return labels


def _candidate_diffs_same_category(diff: int) -> List[int]:
    return [d for d in range(-5, 16) if d != diff and _category(d) == _category(diff)]


def _candidate_diffs_different_category(diff: int) -> List[int]:
    return [d for d in range(-5, 16) if _category(d) != _category(diff)]


def _choose_diff(rng: random.Random, current_diff: int, change_type: str) -> int:
    if change_type.endswith("threshold_hold"):
        candidates = _candidate_diffs_same_category(current_diff)
    elif change_type.endswith("threshold_cross"):
        candidates = _candidate_diffs_different_category(current_diff)
    else:
        candidates = [d for d in range(-5, 16) if d != current_diff]
    return rng.choice(candidates or [current_diff + 1])


_BORDERLINE_CROSS_DIFF_PAIRS: List[Tuple[int, int]] = [
    (-1, 0),  # underperforms -> matches
    (0, 1),  # matches -> slightly outperforms
    (2, 3),  # slightly outperforms -> outperforms
    (7, 8),  # outperforms -> substantially outperforms
    (0, -1),  # matches -> underperforms
    (1, 0),  # slightly outperforms -> matches
    (3, 2),  # outperforms -> slightly outperforms
    (8, 7),  # substantially outperforms -> outperforms
]


_BORDERLINE_HOLD_DIFF_PAIRS: List[Tuple[int, int]] = [
    (-1, -2),  # remains underperforms
    (-2, -1),  # remains underperforms
    (1, 2),  # remains slightly outperforms
    (2, 1),  # remains slightly outperforms
    (3, 4),  # remains outperforms
    (4, 5),  # remains outperforms
    (5, 7),  # remains outperforms
    (7, 6),  # remains outperforms
    (8, 9),  # remains substantially outperforms
    (9, 10),  # remains substantially outperforms
]


def _borderline_case(index: int, step: int) -> Tuple[str, str, int, int]:
    """Return a human-auditable threshold-borderline update specification."""

    is_cross = index % 2 == 0
    pairs = _BORDERLINE_CROSS_DIFF_PAIRS if is_cross else _BORDERLINE_HOLD_DIFF_PAIRS
    old_diff, new_diff = pairs[(index // 2 + step - 1) % len(pairs)]
    changed_attribute = "a_accuracy" if (index // 4 + step) % 2 == 0 else "b_accuracy"
    change_type = "metric_revision_threshold_cross" if is_cross else "metric_revision_threshold_hold"
    return change_type, changed_attribute, old_diff, new_diff


def _initial_scores_for_diff(rng: random.Random, diff: int) -> Tuple[int, int]:
    b_acc = rng.randint(70, 82)
    a_acc = b_acc + diff
    return a_acc, b_acc


def generate_benchmark_instance(
    index: int,
    seed: int = 13,
    sequential_steps: int = 3,
    profile: str = "standard",
) -> SyntheticInstance:
    """Generate one deterministic benchmark-result update chain."""

    if profile not in {"standard", "borderline"}:
        raise ValueError(f"unknown synthetic benchmark profile: {profile}")
    rng = random.Random(seed * 100000 + index)
    instance_id = f"inst_{index:04d}"
    answer_id = f"ans_{index:04d}"
    model_a = f"Model-{chr(65 + index % 20)}A"
    model_b = f"Model-{chr(65 + index % 20)}B"
    split_name = rng.choice(["public-test", "hidden-test", "v2-test"])
    if profile == "borderline":
        _change_type, _changed_attribute, initial_diff, _new_diff = _borderline_case(index, 1)
        a_acc, b_acc = _initial_scores_for_diff(rng, initial_diff)
    else:
        b_acc = rng.randint(63, 82)
        a_acc = b_acc + rng.choice([0, 1, 2, 3, 4, 5, 8, 10, 12])

    evidence_v0 = [
        _evidence(instance_id, 0, "a_accuracy", a_acc),
        _evidence(instance_id, 0, "b_accuracy", b_acc),
        EvidenceRecord(
            evidence_id=f"e_{instance_id}_split_v0",
            entity_id=instance_id,
            version=0,
            text=f"Both models were evaluated on the {split_name} split.",
            source_uri=f"synthetic://benchmark/{instance_id}/split/v0",
            valid_from="0",
            metadata={"attribute": "split", "value": split_name},
        ),
    ]
    answer = _answer(instance_id, answer_id, 0, model_a, model_b, a_acc, b_acc, split_name, evidence_v0)

    updates: List[EvidenceUpdate] = []
    labels_by_step: List[List[ImpactLabel]] = []
    patches = []
    fresh_answers = []
    current_a = a_acc
    current_b = b_acc
    current_a_ev = 0
    current_b_ev = 0
    current_answer = answer
    current_evidence = list(evidence_v0)

    for step in range(1, sequential_steps + 1):
        if profile == "borderline":
            change_type, changed_attribute, _expected_old_diff, new_diff = _borderline_case(index, step)
            if changed_attribute == "a_accuracy":
                new_a = current_b + new_diff
                new_b = current_b
                current_a_ev = step
            else:
                new_a = current_a
                new_b = current_a - new_diff
                current_b_ev = step
        else:
            pattern = (index + step) % 4
            if pattern == 0:
                change_type = "metric_revision_threshold_hold"
                changed_attribute = "a_accuracy"
                new_diff = _choose_diff(rng, current_a - current_b, change_type)
                new_a = current_b + new_diff
                new_b = current_b
                current_a_ev = step
            elif pattern == 1:
                change_type = "metric_revision_threshold_cross"
                changed_attribute = "a_accuracy"
                new_diff = _choose_diff(rng, current_a - current_b, change_type)
                new_a = current_b + new_diff
                new_b = current_b
                current_a_ev = step
            elif pattern == 2:
                change_type = "metric_revision_b_multi_parent"
                changed_attribute = "b_accuracy"
                new_diff = _choose_diff(rng, current_a - current_b, "metric_revision_threshold_cross")
                new_a = current_a
                new_b = current_a - new_diff
                current_b_ev = step
            else:
                change_type = "citation_refresh"
                changed_attribute = None
                new_a = current_a
                new_b = current_b
                current_a_ev = step
                current_b_ev = step
        new_a = max(40, min(98, new_a))
        new_b = max(40, min(98, new_b))
        modified = []
        if changed_attribute == "a_accuracy" or change_type == "citation_refresh":
            modified.append(_evidence(instance_id, current_a_ev, "a_accuracy", new_a))
        if changed_attribute == "b_accuracy" or change_type == "citation_refresh":
            modified.append(_evidence(instance_id, current_b_ev, "b_accuracy", new_b))
        update = EvidenceUpdate(
            update_id=f"u_{instance_id}_{step}",
            entity_id=instance_id,
            from_version=step - 1,
            to_version=step,
            modified_evidence=modified,
            change_type="metric_revision" if change_type.startswith("metric_revision") else change_type,
            metadata={
                "hard_case": change_type,
                "attribute": changed_attribute,
                "old_a_accuracy": current_a,
                "new_a_accuracy": new_a,
                "old_b_accuracy": current_b,
                "new_b_accuracy": new_b,
                "old_diff": current_a - current_b,
                "new_diff": new_a - new_b,
                "old_category": _category(current_a - current_b),
                "new_category": _category(new_a - new_b),
            },
        )
        fresh_evidence = [
            _evidence(instance_id, current_a_ev, "a_accuracy", new_a),
            _evidence(instance_id, current_b_ev, "b_accuracy", new_b),
            evidence_v0[2],
        ]
        fresh = _answer(
            instance_id,
            answer_id,
            step,
            model_a,
            model_b,
            new_a,
            new_b,
            split_name,
            fresh_evidence,
            a_evidence_version=current_a_ev,
            b_evidence_version=current_b_ev,
            report_version=step,
            parent_version=step - 1,
        )
        labels = _impact_labels(
            current_a,
            new_a,
            current_b,
            new_b,
            changed_attribute=changed_attribute,
            citation_refresh=change_type == "citation_refresh",
        )
        patch = build_oracle_patch(current_answer, update, fresh, labels)
        updates.append(update)
        labels_by_step.append(labels)
        patches.append(patch)
        fresh_answers.append(fresh)
        current_a = new_a
        current_b = new_b
        current_answer = fresh
        current_evidence = fresh_evidence

    return SyntheticInstance(
        instance_id=instance_id,
        question=answer.question,
        evidence_v0=current_evidence if False else evidence_v0,
        answer_v0=answer,
        updates=updates,
        gold_impact_labels=labels_by_step,
        gold_patches=patches,
        fresh_answers=fresh_answers,
        metadata={"domain": "benchmark_results", "seed": seed, "hard_cases": True, "profile": profile},
    )


def generate_synthetic_dataset(
    size: int,
    seed: int = 13,
    sequential_steps: int = 3,
    profile: str = "standard",
) -> List[SyntheticInstance]:
    return [
        generate_benchmark_instance(i, seed=seed, sequential_steps=sequential_steps, profile=profile)
        for i in range(size)
    ]


def iter_jsonl(instances: Iterable[SyntheticInstance]) -> Iterable[str]:
    for instance in instances:
        yield instance.model_dump_json()
