from ragas_evaluator import evaluate_response_quality
scores = evaluate_response_quality("What is the mission of Apollo 11?", "Apollo 11 was the first mission to land humans on the Moon.", ["Apollo 11 was the American spaceflight that first landed humans on the Moon."])
print(scores)