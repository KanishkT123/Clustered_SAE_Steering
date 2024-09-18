import evals
import numpy as np
import os
from tqdm import tqdm

example_prompt = "How many times did I tell you to do your homework? I am so disappointed in you."
model = "gpt-4o-mini"

def main():
    results = evals.Sentiment.run_sentiment_analysis(example_prompt, model_name=model, num_iterations=5)  
    print(results)
    
if __name__ == "__main__":
    main()