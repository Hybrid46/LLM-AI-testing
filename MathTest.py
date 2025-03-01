import ollama
import time
import matplotlib.pyplot as plt
import random
import json
import signal
import sys
import re

# Global flag to indicate if the test should stop
stop_test = False

# List of models to benchmark
MODELS_TO_TEST = [
    "deepseek-r1:7b",
    "deepseek-r1:7b-qwen-distill-fp16",
    "deepseek-r1:8b",
    "deepseek-r1:14b",
    "deepscaler:latest",
    "olmo2:7b-1124-instruct-q8_0",
    "openthinker:7b-fp16",
    "deepseek-coder-v2:latest",
    "starcoder2:15b",
    "qwen2.5-coder:14b",
]

# Generate 100 YES/NO test questions and answers
PROMPTS_WITH_ANSWERS = []

# Mathematical questions (YES/NO format)
math_questions = [
    {"prompt": "Is 2 + 2 equal to 4?", "expected_answer": "yes"},
    {"prompt": "Is 10 * 5 equal to 50?", "expected_answer": "yes"},
    {"prompt": "Is the square root of 64 equal to 8?", "expected_answer": "yes"},
    {"prompt": "Is 3^3 equal to 27?", "expected_answer": "yes"},
    {"prompt": "Is 15 - 7 equal to 9?", "expected_answer": "no"},
    {"prompt": "Is 12 * 12 equal to 144?", "expected_answer": "yes"},
    {"prompt": "Is the cube root of 27 equal to 3?", "expected_answer": "yes"},
    {"prompt": "Is 5! equal to 120?", "expected_answer": "yes"},
    {"prompt": "Is (10 + 5) * 2 equal to 30?", "expected_answer": "yes"},
    {"prompt": "Is 100 / 4 equal to 25?", "expected_answer": "yes"},
    {"prompt": "Is log10(100) equal to 2?", "expected_answer": "yes"},
    {"prompt": "Is sin(30°) equal to 0.5?", "expected_answer": "yes"},
    {"prompt": "Is cos(60°) equal to 0.5?", "expected_answer": "yes"},
    {"prompt": "Is tan(45°) equal to 1?", "expected_answer": "yes"},
    {"prompt": "Is e^0 equal to 1?", "expected_answer": "yes"},
    {"prompt": "Is ln(1) equal to 0?", "expected_answer": "yes"},
    {"prompt": "Is 2^10 equal to 1024?", "expected_answer": "yes"},
    {"prompt": "Is 1000 / 100 equal to 10?", "expected_answer": "yes"},
    {"prompt": "Is 50 + 25 equal to 75?", "expected_answer": "yes"},
    {"prompt": "Is 99 - 33 equal to 66?", "expected_answer": "yes"},
]

# Coordinate geometry questions (YES/NO format)
coordinate_geometry_questions = [
    {"prompt": "Is the distance between (0, 0) and (3, 4) equal to 5?", "expected_answer": "yes"},
    {"prompt": "Is the slope of the line through (1, 2) and (3, 4) equal to 1?", "expected_answer": "yes"},
    {"prompt": "Is the midpoint of (2, 3) and (4, 7) equal to (3, 5)?", "expected_answer": "yes"},
    {"prompt": "Is the equation of a line with slope 2 and y-intercept 3 equal to y = 2x + 3?", "expected_answer": "yes"},
    {"prompt": "Is the equation of the x-axis equal to y = 0?", "expected_answer": "yes"},
    {"prompt": "Is the slope of a horizontal line equal to 0?", "expected_answer": "yes"},
    {"prompt": "Is the area of a triangle with vertices (0, 0), (3, 0), (0, 4) equal to 6?", "expected_answer": "yes"},
    {"prompt": "Is the equation of a circle with center (0, 0) and radius 5 equal to x² + y² = 25?", "expected_answer": "yes"},
    {"prompt": "Is the equation of the line perpendicular to y = 2x + 1 passing through (0, 0) equal to y = -0.5x?", "expected_answer": "yes"},
    {"prompt": "Is the distance between (-1, -1) and (2, 3) equal to 4?", "expected_answer": "no"},
]

# Advanced geometry questions (triangles, quadrilaterals, circles)
advanced_geometry_questions = [
    {"prompt": "Is the area of a triangle with base 4 and height 3 equal to 6?", "expected_answer": "yes"},
    {"prompt": "Is the perimeter of a rectangle with sides 5 and 10 equal to 30?", "expected_answer": "yes"},
    {"prompt": "Is the area of a circle with radius 7 equal to 153.94?", "expected_answer": "yes"},
    {"prompt": "Is the sum of angles in a triangle equal to 160°?", "expected_answer": "no"},
    {"prompt": "Is the area of a square with side length 5 equal to 25?", "expected_answer": "yes"},
    {"prompt": "Is the diagonal of a rectangle with sides 3 and 4 equal to 5?", "expected_answer": "yes"},
    {"prompt": "Is the circumference of a circle with radius 10 equal to 62.83?", "expected_answer": "yes"},
    {"prompt": "Is the area of a trapezoid with bases 3 and 5 and height 4 equal to 16?", "expected_answer": "yes"},
    {"prompt": "Is the volume of a sphere with radius 3 equal to 113.10?", "expected_answer": "yes"},
    {"prompt": "Is the surface area of a cube with side length 2 equal to 24?", "expected_answer": "yes"},
]

# Advanced math questions (trigonometry, logarithms, factorials, exponents)
advanced_math_questions = [
    {"prompt": "Is sin(90°) equal to 1?", "expected_answer": "yes"},
    {"prompt": "Is cos(0°) equal to 1?", "expected_answer": "yes"},
    {"prompt": "Is tan(90°) undefined?", "expected_answer": "yes"},
    {"prompt": "Is log2(8) equal to 3?", "expected_answer": "yes"},
    {"prompt": "Is ln(e) equal to 1?", "expected_answer": "yes"},
    {"prompt": "Is 10! equal to 3,628,80?", "expected_answer": "no"},
    {"prompt": "Is 2^5 equal to 32?", "expected_answer": "yes"},
    {"prompt": "Is e^1 equal to approximately 2.718?", "expected_answer": "yes"},
    {"prompt": "Is the derivative of x^2 equal to 2x?", "expected_answer": "yes"},
    {"prompt": "Is the integral of 2x equal to x^2 + C?", "expected_answer": "yes"},
]

# Combine all questions
PROMPTS_WITH_ANSWERS = (
    math_questions
    + coordinate_geometry_questions
    + advanced_geometry_questions
    + advanced_math_questions
)

def signal_handler(sig, frame):
    """
    Handle CTRL+C interrupt to stop the test gracefully.
    """
    global stop_test
    print("\nCTRL+C detected. Stopping the test...")
    stop_test = True

def test_model(model_name, prompt, expected_answer):
    """
    Test a single model with a given prompt and measure response time and correctness.
    """
    try:
        # Add instruction to the prompt
        formatted_prompt = (
            "Answer with [YES] or [NO] at the beginning, don't make any explanation just the short yes-no answer! "
            f"Question: {prompt}"
        )
        
        start_time = time.time()
        response = ollama.generate(model=model_name, prompt=formatted_prompt)
        end_time = time.time()
        
        # Extract YES/NO from the response using regex (case-insensitive)
        response_text = response['response'].lower().strip()
        match = re.search(r'^\[?(yes|no)\]?', response_text)
        extracted_answer = match.group(1).lower() if match else ""
        
        # Fallback: Check if the expected answer appears anywhere in the response
        is_correct = (
            extracted_answer == expected_answer.lower() or
            expected_answer.lower() in response_text
        )
        
        return {
            "model": model_name,
            "prompt": prompt,
            "response": response['response'],
            "expected_answer": expected_answer,
            "is_correct": is_correct,
            "response_time": end_time - start_time,
            "error": None,
        }
    except Exception as e:
        return {
            "model": model_name,
            "prompt": prompt,
            "response": None,
            "expected_answer": expected_answer,
            "is_correct": False,
            "response_time": None,
            "error": str(e),
        }

def benchmark_models(models, prompts_with_answers):
    """
    Benchmark multiple models with a list of prompts and expected answers.
    """
    global stop_test
    results = []
    total_models = len(models)
    total_prompts = len(prompts_with_answers)
    
    for model_idx, model in enumerate(models, 1):
        if stop_test:
            break
        print(f"\nTesting model {model_idx}/{total_models}: {model}")
        for prompt_idx, item in enumerate(prompts_with_answers, 1):
            if stop_test:
                break
            prompt = item["prompt"]
            expected_answer = item["expected_answer"]
            print(f"  Prompt {prompt_idx}/{total_prompts}: {prompt}")
            result = test_model(model, prompt, expected_answer)
            results.append(result)
            print(f"    Response: {result['response']}")
            print(f"    Expected Answer: {result['expected_answer'].upper()}")
            print(f"    Correct: {result['is_correct']}")
            print(f"    Response Time: {result['response_time']:.2f} seconds")
            if result['error']:
                print(f"    Error: {result['error']}")
            print("-" * 40)
    return results

def save_results_to_json(results, filename="benchmark_results.json"):
    """
    Save the benchmark results to a JSON file.
    """
    if not results:
        print("\nNo results to save.")
        return
    
    with open(filename, mode="w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)
    print(f"\nResults saved to {filename}")

def calculate_metrics(results):
    """
    Calculate accuracy and average response time for each model.
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
    # Register the CTRL+C signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Run the benchmark
    benchmark_results = benchmark_models(MODELS_TO_TEST, PROMPTS_WITH_ANSWERS)
    
    # Save results to a JSON file
    save_results_to_json(benchmark_results)
    
    # Calculate metrics
    if benchmark_results:
        metrics = calculate_metrics(benchmark_results)
        
        # Print metrics
        print("\nModel Metrics:")
        for model, data in metrics.items():
            print(f"{model}:")
            print(f"  Accuracy: {data['accuracy']:.2f}%")
            print(f"  Average Response Time: {data['avg_response_time']:.2f} seconds")
        
        # Plot metrics
        plot_metrics(metrics)
    else:
        print("\nNo results to display.")