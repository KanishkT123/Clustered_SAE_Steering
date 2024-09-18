import re
from collections import Counter
import math
import json
from dotenv import load_dotenv
import numpy as np
import os
from tqdm import tqdm
from typing import List, Dict
from openai import OpenAI

MODEL = "gpt-4o-mini"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def preprocess(text):
    # Convert to lowercase and remove punctuation
    return text.lower().split()
  
def calculate_breakage_coefficient(text, max_sequence_length=100):
    words = preprocess(text)
    word_freq = Counter(words)

    total_repeats = sum(freq - 1 for freq in word_freq.values() if freq > 1)

    # normalize for word length
    # TODO: should it be tokenized sequence length? idk.
    normalization_factor = max_sequence_length / len(words)
    normalized_repeats = total_repeats * normalization_factor

    return normalized_repeats

def calculate_tokenized_breakage(model, text):
  tokens = model.tokenizer.tokenize(text)
  unique_tokens = len(set(tokens))
  repeat_count = len(tokens) - unique_tokens
  return repeat_count*100/len(tokens)

def calculate_all_breakage(string_list):
  return [calculate_breakage_coefficient(result) for result in string_list]

def calculate_all_breakage(model, string_list):
  return [calculate_tokenized_breakage(model, result) for result in string_list]

def get_breakage_dict(results_dict):
  return {k:[calculate_breakage_coefficient(result) for result in v] for k, v in results_dict.items()}

def get_breakage_dict(model, results_dict):
  return {k:[calculate_tokenized_breakage(model, result) for result in v] for k, v in results_dict.items()}

### Rollout Calculations ###
# @title helper functions for rollout
def rollout_success(output, word_list):
  # returns boolean (T, F) of whether rollout has words in the word_list
  output = '_'.join(output.split(' ')) # replace spaces with underscores (like angry_words)
  return any([w in output.lower() for w in word_list])


class Sentiment:
    
    def extract_token_logprobs(logprobs_content: List, tokens_of_interest: set) -> Dict[str, float]:
        result = {
            logprob.token: round(logprob.logprob, 3)
            for logprob in logprobs_content
            if logprob.token in tokens_of_interest
        }
        
        # Fill in any missing tokens with negative infinity
        result.update({token: -math.inf for token in tokens_of_interest if token not in result})
        
        return result


    def openai_query(prompt: str, text: str, model_name: str) -> tuple:
        
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Classify the sentiment of the given text."},
                {"role": "user", "content": f"{prompt}\n\nText: {text}"}
            ],
            temperature=0,
            logprobs=True,
            top_logprobs=10
        )
        
        top_token = completion.choices[0].message.content
        logprob_content = completion.choices[0].logprobs.content[0].top_logprobs
        
        logprob_dict = Sentiment.extract_token_logprobs(logprob_content, Sentiment.tokens_of_interest)

        return logprob_dict, top_token

    emotion_mapping = {
        'joy': ['joy', 'joyful', 'joyous', 'joyfulness'],
        'contentment': ['contentment', 'content', 'contented'],
        'excitement': ['excitement', 'excited', 'exciting'],
        'gratitude': ['gratitude', 'grateful', 'gratified'],
        'sadness': ['sadness', 'sad', 'saddened'],
        'anger': ['anger', 'angry', 'angered'],
        'fear': ['fear', 'fearful', 'afraid'],
        'disgust': ['disgust', 'disgusted', 'disgusting'],
        'indifference': ['indifference', 'indifferent', 'uninterested'],
        'ambivalence': ['ambivalence', 'ambivalent', 'uncertain'],
        'nostalgia': ['nostalgia', 'nostalgic', 'longing'],
        'bittersweet': ['bittersweet', 'bitter', 'sweet'],
        'anticipation': ['anticipation', 'anticipating', 'expectation'],
        'surprise': ['surprise', 'surprised', 'unexpected'],
        'empathy': ['empathy', 'empathetic', 'compassion'],
        'pride': ['pride', 'proud', 'self-satisfaction'],
        'shame': ['shame', 'ashamed'],
        'guilt': ['guilt', 'guilty', 'remorse'],
        'curiosity': ['curiosity', 'curious', 'inquisitive'],
        'confusion': ['confusion', 'confused', 'puzzled'],
        'certainty': ['certainty', 'certain', 'sure'],
        'doubt': ['doubt', 'doubtful', 'uncertain']
    }

    # Create a reverse mapping for easy lookup
    reverse_emotion_mapping = {variation: base for base, variations in emotion_mapping.items() for variation in variations}

    tokens_of_interest = set(emotion_mapping.keys())

    #using the hybrid model combining Ekman, dimensional, and arousal models of sentiment
    prompt = """Classify the sentiment of this text. Choose the most appropriate sentiment from the following categories:

    curiosity, confusion, certainty, doubt, joy, contentment, excitement, gratitude, sadness, anger, fear, disgust, ambivalence, indifference, bittersweet, nostalgia, anticipation, surprise, empathy, pride, shame, guilt

    Respond with a single word that best represents the dominant sentiment."""
    results = []


    def run_sentiment_analysis(text: str, model_name: str, num_iterations: int = 5, prompt=prompt) -> Dict:
        if not text:
            return {
        "text": "text is empty",
        "responses": "response is empty",
        "avg_logprobs": "avg_logprobs is empty",
        "most_frequent_response": "empty text"
    }
        
        logprob_list = []
        response_list = []
        
        for _ in tqdm(range(num_iterations), desc=f"Analyzing: {text[:30]}..."):
            logprob_dict, top_token = Sentiment.openai_query(prompt, text, model_name)
            
            # Map the response to its base emotion
            base_emotion = Sentiment.reverse_emotion_mapping.get(top_token.lower(), top_token)
            response_list.append(base_emotion)
            logprob_list.append(logprob_dict)
        
        # Calculate average logprobs
        # ** don't average over the missing or negative infinity values **
        avg_logprobs = {token: np.mean([iteration[token] for iteration in logprob_list if token in iteration]) 
                    for token in set().union(*logprob_list)}
        
        # Get most frequent response using the base emotions
        most_frequent_response = Counter(response_list).most_common(1)[0][0]
        
        print(f"response is {response_list}")
        return {
            "text": text,
            "responses": response_list,
            "avg_logprobs": avg_logprobs,
            "most_frequent_response": most_frequent_response
        }
        
    sample_texts = [
        "I am so happy today",
        "I am sick of your behavior",
        "I am feeling very sad",
        "I don't know what to feel about this"
    ]

def main():
    for text in sample_texts:
        result = run_sentiment_analysis(text, model_name= MODEL, num_iterations=5, prompt=prompt)
        results.append(result)
        
        print(f"\nText: {text}")
        print(f"Most frequent response: {result['most_frequent_response']}")
        print("Average log probabilities:")
        for token, logprob in result['avg_logprobs'].items():
            print(f"{token}: {logprob}")

    # Save results
    with open("sentiment_analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nResults saved in sentiment_analysis_results.json")
    
if __name__ == "__main__":
    main()
    
    
    
    