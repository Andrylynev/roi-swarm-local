from roi_swarm.telemetry import RunStore


def test_run_store_tracks_graph_nodes_and_events(tmp_path) -> None:
    store = RunStore(tmp_path / "runs")
    run_id = store.create_run("research task")

    store.record_event(run_id, "router", "completed", {"worker_target": 10})
    store.record_event(run_id, "worker-1", "running", {"role": "researcher"})
    store.record_event(run_id, "worker-1", "completed", {"confidence": 0.8})
    store.complete_run(run_id, "answer")

    snapshot = store.get_run(run_id)

    assert snapshot["status"] == "completed"
    assert snapshot["answer"] == "answer"
    assert snapshot["nodes"]["router"]["status"] == "completed"
    assert snapshot["nodes"]["worker-1"]["status"] == "completed"
    assert len(snapshot["events"]) == 3


def test_run_store_lists_newest_first(tmp_path) -> None:
    store = RunStore(tmp_path / "runs")
    first = store.create_run("first")
    second = store.create_run("second")

    listed = store.list_runs()

    assert [item["id"] for item in listed[:2]] == [second, first]


def test_run_store_lists_external_completion_ids(tmp_path) -> None:
    store = RunStore(tmp_path / "runs")
    store.create_run("from hermes", run_id="chatcmpl-123")

    assert store.list_runs()[0]["id"] == "chatcmpl-123"
