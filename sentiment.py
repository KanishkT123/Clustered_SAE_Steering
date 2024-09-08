import os
import json
from typing import List, Dict
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm
from collections import Counter
from openai import OpenAI
import math

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
MODEL = "gpt-4o-mini"  

# Sample texts for sentiment analysis
sample_texts = [
    # "I'm sorry to hear that you feel that you deserve an apology, but you don't.",
    # "1. When I go on a walk, I like to want to see something. Anxiety has a huge impact on my anger at anger. Anger is rage-rage, rage can rage in anger and rage-rage can rage in fury. Anger does not feel irrational, angry anger will not upset you if",
    # "I'm so angry right now, I can't believe what just happened.",
    # "Dude what am I supposed to do with that information?",
    "Part of me wants to stay in this relationship, while another part thinks it might be time to move on.",
    "Clearing out my childhood home brought back wonderful memories, even as I realized I could never return to those simpler times.",
    "Receiving my dream job offer meant leaving behind the city and friends I'd grown to love.",
    "I've stayed away from home too long. I wanna go back."
    # # Add more sample texts as needed
]

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
    
    logprob_dict = extract_token_logprobs(logprob_content, tokens_of_interest)

    return logprob_dict, top_token

from collections import Counter

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
        logprob_dict, top_token = openai_query(prompt, text, model_name)
        
        # Map the response to its base emotion
        base_emotion = reverse_emotion_mapping.get(top_token.lower(), top_token)
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

def main():

    for text in sample_texts:
        result = run_sentiment_analysis(text, prompt, MODEL)
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