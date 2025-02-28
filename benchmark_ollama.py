import ollama
import time
import matplotlib.pyplot as plt
import random

# List of models to benchmark
MODELS_TO_TEST = ["deepseek-r1:7b", "deepseek-r1:8b", "deepscaler:latest"]  # Add or replace with your desired models

# Generate 100 test questions and answers
PROMPTS_WITH_ANSWERS = []

# Mathematical questions (basic arithmetic and algebra)
math_questions = [
    {"prompt": "What is 2 + 2?", "expected_answers": ["4"]},
    {"prompt": "What is 10 * 5?", "expected_answers": ["50"]},
    {"prompt": "What is 100 / 10?", "expected_answers": ["10"]},
    {"prompt": "What is the square root of 64?", "expected_answers": ["8", "-8"]},  # Multiple answers
    {"prompt": "What is 3^3?", "expected_answers": ["27"]},
    {"prompt": "What is 15 - 7?", "expected_answers": ["8"]},
    {"prompt": "What is 12 * 12?", "expected_answers": ["144"]},
    {"prompt": "What is 1000 / 100?", "expected_answers": ["10"]},
    {"prompt": "What is 2^10?", "expected_answers": ["1024"]},
    {"prompt": "What is 50 + 25?", "expected_answers": ["75"]},
    {"prompt": "What is 99 - 33?", "expected_answers": ["66"]},
    {"prompt": "What is 8 * 7?", "expected_answers": ["56"]},
    {"prompt": "What is 144 / 12?", "expected_answers": ["12"]},
    {"prompt": "What is the cube root of 27?", "expected_answers": ["3"]},
    {"prompt": "What is 5!", "expected_answers": ["120"]},
    {"prompt": "What is 2 * 3 + 4?", "expected_answers": ["10"]},
    {"prompt": "What is (10 + 5) * 2?", "expected_answers": ["30"]},
    {"prompt": "What is 100 - 50 + 25?", "expected_answers": ["75"]},
    {"prompt": "What is 2^5?", "expected_answers": ["32"]},
    {"prompt": "What is 100 / 4?", "expected_answers": ["25"]},
]

# Coordinate geometry questions
coordinate_geometry_questions = [
    {"prompt": "What is the distance between points (0, 0) and (3, 4)?", "expected_answers": ["5"]},
    {"prompt": "What is the slope of the line passing through points (1, 2) and (3, 4)?", "expected_answers": ["1"]},
    {"prompt": "What is the midpoint of the line segment joining (2, 3) and (4, 7)?", "expected_answers": ["(3, 5)", "3, 5"]},
    {"prompt": "What is the equation of the line with slope 2 and y-intercept 3?", "expected_answers": ["y = 2x + 3", "2x + 3"]},
    {"prompt": "What is the equation of the line passing through points (1, 1) and (2, 3)?", "expected_answers": ["y = 2x - 1", "2x - 1"]},
    {"prompt": "What is the distance between points (1, 1) and (4, 5)?", "expected_answers": ["5"]},
    {"prompt": "What is the slope of the line y = 3x + 2?", "expected_answers": ["3"]},
    {"prompt": "What is the y-intercept of the line y = 4x - 5?", "expected_answers": ["-5"]},
    {"prompt": "What is the equation of the x-axis?", "expected_answers": ["y = 0"]},
    {"prompt": "What is the equation of the y-axis?", "expected_answers": ["x = 0"]},
    {"prompt": "What is the slope of a horizontal line?", "expected_answers": ["0"]},
    {"prompt": "What is the distance between points (-1, -1) and (2, 3)?", "expected_answers": ["5"]},
    {"prompt": "What is the midpoint of the line segment joining (-2, -3) and (4, 5)?", "expected_answers": ["(1, 1)"]},
    {"prompt": "What is the equation of the line parallel to y = 2x + 1 and passing through (0, 0)?", "expected_answers": ["y = 2x", "2x"]},
    {"prompt": "What is the equation of the line perpendicular to y = 2x + 1 and passing through (0, 0)?", "expected_answers": ["y = -0.5x", "y = -1/2x"]},  # Multiple answers
    {"prompt": "What is the area of the triangle with vertices at (0, 0), (3, 0), and (0, 4)?", "expected_answers": ["6"]},
    {"prompt": "What is the perimeter of the triangle with vertices at (0, 0), (3, 0), and (0, 4)?", "expected_answers": ["12"]},
    {"prompt": "What is the equation of the circle with center (0, 0) and radius 5?", "expected_answers": ["x² + y² = 25", "25"]},
]

# Math equations (quadratic, linear, etc.)
math_equations = [
    {"prompt": "Solve for x: 2x + 5 = 15", "expected_answers": ["x = 5"]},
    {"prompt": "Solve for x: 3x - 7 = 14", "expected_answers": ["x = 7"]},
    {"prompt": "Solve for x: x² - 4 = 0", "expected_answers": ["x = 2", "x = -2"]},  # Multiple answers
    {"prompt": "Solve for x: x² + 5x + 6 = 0", "expected_answers": ["x = -2", "x = -3"]},  # Multiple answers
    {"prompt": "Solve for x: 2x² - 8 = 0", "expected_answers": ["x = 2", "x = -2"]},  # Multiple answers
    {"prompt": "Solve for x: x² - 9 = 0", "expected_answers": ["x = 3", "x = -3"]},  # Multiple answers
    {"prompt": "Solve for x: x² + 4x + 4 = 0", "expected_answers": ["x = -2"]},
    {"prompt": "Solve for x: x² - 6x + 9 = 0", "expected_answers": ["x = 3"]},
    {"prompt": "Solve for x: x² + 3x - 10 = 0", "expected_answers": ["x = 2", "x = -5"]},  # Multiple answers
    {"prompt": "Solve for x: x² - 5x + 6 = 0", "expected_answers": ["x = 2", "x = 3"]},  # Multiple answers
    {"prompt": "Solve for x: x² + 2x - 8 = 0", "expected_answers": ["x = 2", "x = -4"]},  # Multiple answers
    {"prompt": "Solve for x: x² - 7x + 12 = 0", "expected_answers": ["x = 3", "x = 4"]},  # Multiple answers
    {"prompt": "Solve for x: x² + 6x + 9 = 0", "expected_answers": ["x = -3"]},
    {"prompt": "Solve for x: x² - 8x + 16 = 0", "expected_answers": ["x = 4"]},
    {"prompt": "Solve for x: x² + x - 6 = 0", "expected_answers": ["x = 2", "x = -3"]},  # Multiple answers
    {"prompt": "Solve for x: x² - 4x + 4 = 0", "expected_answers": ["x = 2"]},
    {"prompt": "Solve for x: x² + 5x - 14 = 0", "expected_answers": ["x = 2", "x = -7"]},  # Multiple answers
    {"prompt": "Solve for x: x² - 9x + 18 = 0", "expected_answers": ["x = 3", "x = 6"]},  # Multiple answers
    {"prompt": "Solve for x: x² + 7x + 12 = 0", "expected_answers": ["x = -3", "x = -4"]},  # Multiple answers
    {"prompt": "Solve for x: x² - 10x + 25 = 0", "expected_answers": ["x = 5"]},
]

# Combine all questions
PROMPTS_WITH_ANSWERS = math_questions + coordinate_geometry_questions + math_equations

# Add more questions to reach 100
while len(PROMPTS_WITH_ANSWERS) < 100:
    a = random.randint(1, 100)
    b = random.randint(1, 100)
    c = random.randint(1, 100)
    PROMPTS_WITH_ANSWERS.append({
        "prompt": f"What is {a} + {b} * {c}?",
        "expected_answers": [str(a + b * c)],
    })

def test_model(model_name, prompt, expected_answers):
    """
    Test a single model with a given prompt and measure response time and correctness.
    
    :param model_name: Name of the model to test.
    :param prompt: The input prompt to send to the model.
    :param expected_answers: List of expected correct answers for the prompt.
    :return: A dictionary containing the response, response time, correctness, and any errors.
    """
    try:
        start_time = time.time()
        response = ollama.generate(model=model_name, prompt=prompt)
        end_time = time.time()
        
        # Check if the response matches any of the expected answers
        is_correct = any(
            expected.lower() in response['response'].lower()
            for expected in expected_answers
        )
        
        return {
            "model": model_name,
            "prompt": prompt,
            "response": response['response'],
            "expected_answers": expected_answers,
            "is_correct": is_correct,
            "response_time": end_time - start_time,
            "error": None,
        }
    except Exception as e:
        return {
            "model": model_name,
            "prompt": prompt,
            "response": None,
            "expected_answers": expected_answers,
            "is_correct": False,
            "response_time": None,
            "error": str(e),
        }

def benchmark_models(models, prompts_with_answers):
    """
    Benchmark multiple models with a list of prompts and expected answers.
    
    :param models: List of model names to benchmark.
    :param prompts_with_answers: List of dictionaries containing prompts and expected answers.
    :return: A list of results for each model and prompt.
    """
    results = []
    for model in models:
        print(f"Testing model: {model}")
        for item in prompts_with_answers:
            prompt = item["prompt"]
            expected_answers = item["expected_answers"]
            print(f"  Prompt: {prompt}")
            result = test_model(model, prompt, expected_answers)
            results.append(result)
            print(f"  Response: {result['response']}")
            print(f"  Expected Answers: {result['expected_answers']}")
            print(f"  Correct: {result['is_correct']}")
            print(f"  Response Time: {result['response_time']:.2f} seconds")
            if result['error']:
                print(f"  Error: {result['error']}")
            print("-" * 40)
    return results

def calculate_metrics(results):
    """
    Calculate accuracy and average response time for each model.
    
    :param results: List of results from the benchmark.
    :return: A dictionary with accuracy and average response time for each model.
    """
    metrics = {}
    for model in MODELS_TO_TEST:
        total = 0
        correct = 0
        total_time = 0
        for result in results:
            if result['model'] == model:
                total += 1
                if result['is_correct']:
                    correct += 1
                if result['response_time']:
                    total_time += result['response_time']
        accuracy = (correct / total) * 100 if total > 0 else 0
        avg_response_time = total_time / total if total > 0 else 0
        metrics[model] = {
            "accuracy": accuracy,
            "avg_response_time": avg_response_time,
        }
    return metrics

def plot_metrics(metrics):
    """
    Plot the accuracy and average response time of each model.
    
    :param metrics: Dictionary containing accuracy and response time for each model.
    """
    models = list(metrics.keys())
    accuracy = [metrics[model]["accuracy"] for model in models]
    avg_response_time = [metrics[model]["avg_response_time"] for model in models]

    # Create a bar chart for accuracy
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.bar(models, accuracy, color='blue', alpha=0.6, label='Accuracy')
    ax1.set_xlabel('Models')
    ax1.set_ylabel('Accuracy (%)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_ylim(0, 110)
    ax1.set_title('Model Performance: Accuracy and Response Time')

    # Create a line plot for average response time
    ax2 = ax1.twinx()
    ax2.plot(models, avg_response_time, color='red', marker='o', label='Avg Response Time')
    ax2.set_ylabel('Average Response Time (seconds)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(0, max(avg_response_time) * 1.2)

    # Add legends
    fig.legend(loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)

    plt.show()

if __name__ == "__main__":
    # Run the benchmark
    benchmark_results = benchmark_models(MODELS_TO_TEST, PROMPTS_WITH_ANSWERS)
    
    # Calculate metrics
    metrics = calculate_metrics(benchmark_results)
    
    # Print metrics
    print("\nModel Metrics:")
    for model, data in metrics.items():
        print(f"{model}:")
        print(f"  Accuracy: {data['accuracy']:.2f}%")
        print(f"  Average Response Time: {data['avg_response_time']:.2f} seconds")
    
    # Plot metrics
    plot_metrics(metrics)