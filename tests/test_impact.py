from claimpatch.baselines.descendant_all import descendant_all_labels
from claimpatch.baselines.attribute_no_graph import attribute_no_graph_labels
from claimpatch.baselines.independent_claim_revision import direct_only_labels
from claimpatch.data.synthetic_generator import _category, generate_benchmark_instance, generate_synthetic_dataset
from claimpatch.impact.direct import directly_impacted_claims
from claimpatch.impact.propagation import rule_based_impact


def test_direct_impact_detects_changed_metric_claim():
    instance = generate_benchmark_instance(0)
    direct = directly_impacted_claims(instance.answer_v0, instance.updates[0])
    assert "c_a_acc" in direct


def test_rule_based_impact_changes_downstream_difference():
    instance = generate_benchmark_instance(0)
    labels = {x.claim_id: x for x in rule_based_impact(instance.answer_v0, instance.updates[0])}
    assert labels["c_diff"].state == "MUST_CHANGE"


def test_rule_based_impact_preserves_split():
    instance = generate_benchmark_instance(3)
    labels = {x.claim_id: x for x in rule_based_impact(instance.answer_v0, instance.updates[0])}
    assert labels["c_split"].state == "STILL_VALID"


def test_direct_only_misses_implicit_claims():
    instance = generate_benchmark_instance(4)
    labels = direct_only_labels(instance.gold_impact_labels[0])
    states = {x.claim_id: x.state for x in labels}
    assert states["c_diff"] == "STILL_VALID"


def test_descendant_all_touches_descendants():
    instance = generate_benchmark_instance(5)
    labels = descendant_all_labels(instance.answer_v0, instance.updates[0], instance.gold_impact_labels[0])
    states = {x.claim_id: x.state for x in labels}
    assert states["c_diff"] == "MUST_CHANGE"


def test_hard_generator_includes_citation_refresh():
    instance = generate_benchmark_instance(0)
    assert any(update.metadata["hard_case"] == "citation_refresh" for update in instance.updates)


def test_attribute_no_graph_misses_downstream_metric_claims():
    instance = generate_benchmark_instance(0)
    step = next(i for i, update in enumerate(instance.updates) if update.metadata["attribute"] == "a_accuracy")
    labels = attribute_no_graph_labels(instance.gold_impact_labels[step], instance.updates[step])
    states = {x.claim_id: x.state for x in labels}
    assert states["c_a_acc"] == "MUST_CHANGE"
    assert states["c_diff"] == "STILL_VALID"


def test_citation_refresh_preserves_metric_values():
    instance = generate_benchmark_instance(0)
    step = next(i for i, update in enumerate(instance.updates) if update.metadata["hard_case"] == "citation_refresh")
    labels = {x.claim_id: x for x in instance.gold_impact_labels[step]}
    assert labels["c_a_acc"].state == "STILL_VALID"
    assert labels["c_b_acc"].state == "STILL_VALID"
    assert labels["c_citation"].state == "MUST_CHANGE"


def test_borderline_profile_balances_cross_and_hold_with_clear_gold_labels():
    instances = generate_synthetic_dataset(size=20, seed=3031, sequential_steps=1, profile="borderline")
    hard_cases = [instance.updates[0].metadata["hard_case"] for instance in instances]
    assert hard_cases.count("metric_revision_threshold_cross") == 10
    assert hard_cases.count("metric_revision_threshold_hold") == 10

    for instance in instances:
        update = instance.updates[0]
        labels = {label.claim_id: label for label in instance.gold_impact_labels[0]}
        old_diff = update.metadata["old_diff"]
        new_diff = update.metadata["new_diff"]
        old_category = _category(old_diff)
        new_category = _category(new_diff)
        assert update.metadata["old_category"] == old_category
        assert update.metadata["new_category"] == new_category
        assert old_diff != new_diff
        if update.metadata["hard_case"] == "metric_revision_threshold_cross":
            assert old_category != new_category
            assert labels["c_comparison"].state == "MUST_CHANGE"
            assert labels["c_comparison"].gold_operation == "REPLACE"
        else:
            assert old_category == new_category
            assert labels["c_comparison"].state == "STILL_VALID"
            assert labels["c_comparison"].gold_operation is None
