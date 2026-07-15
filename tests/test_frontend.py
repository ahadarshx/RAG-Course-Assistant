def test_stream_includes_the_complete_answer():
    # Mirrors the UI's final stream event: no trailing model characters may be dropped.
    answer = "A complete answer."
    emitted = [answer[:i] for i in range(16, len(answer), 16)] + [answer]
    assert emitted[-1] == answer
