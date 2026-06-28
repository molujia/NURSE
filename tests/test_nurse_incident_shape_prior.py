from nurse.incident_shape_prior import (
    build_incident_shape_features,
    classify_metric_family,
    compute_counterfactual_shape_drop,
    score_incident_shape_prior,
)


def test_metric_family_groups_tt_and_aiops_names():
    assert classify_metric_family("pod_cpu_usage") == "cpu"
    assert classify_metric_family("memory_usage") == "memory"
    assert classify_metric_family("pod_memory_working_set_bytes") == "memory"
    assert classify_metric_family("rrt_max") == "latency"
    assert classify_metric_family("pod_network_receive_bytes") == "network"
    assert classify_metric_family("request") == "traffic"


def test_shape_features_do_not_include_labels():
    features = build_incident_shape_features(
        metric_scores={"frontend": {"pod_cpu_usage": 5.0, "request": 1.0}},
        topology_edges=[("frontend", "checkoutservice")],
        service_candidates=["frontend", "checkoutservice"],
        dataset="rcaeval-tt",
    )

    serialized = str(features).lower()
    assert "root_cause" not in serialized
    assert "answer_key" not in serialized
    assert "fault_type" not in serialized
    assert features["entities"]["frontend"]["families"]["cpu"] > 0


def test_counterfactual_drop_prefers_concentrated_root_like_signal():
    features = build_incident_shape_features(
        metric_scores={
            "root": {"pod_cpu_usage": 8.0, "memory_usage": 2.0},
            "symptom": {"request": 7.0},
        },
        topology_edges=[("root", "symptom")],
        service_candidates=["root", "symptom"],
        dataset="aiops25",
    )

    drops = compute_counterfactual_shape_drop(features)

    assert drops["root"] > drops["symptom"]


def test_shape_prior_rewards_topology_source_with_downstream_symptom():
    features = build_incident_shape_features(
        metric_scores={
            "source": {"pod_cpu_usage": 6.0},
            "downstream": {"rrt": 4.0, "request": 2.0},
        },
        topology_edges=[("source", "downstream")],
        service_candidates=["source", "downstream"],
        dataset="rcaeval-tt",
    )

    prior = score_incident_shape_prior(features)

    assert prior["source"]["total"] > prior["downstream"]["total"]
    assert prior["source"]["topology_locality"] > 0
