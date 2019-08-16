from ...training.engine import create_rnn_evaluator

def test(model, data, metrics, device):
    tester = create_rnn_evaluator(model, metrics, device)
    tester.run(data)
    return tester.state.metrics
