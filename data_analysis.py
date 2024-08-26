import re
from collections import Counter

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

def bool2prob(bool_list):
  return sum(bool_list)/len(bool_list)

# note, that variance is calculated via this post:
# https://math.stackexchange.com/questions/1285621/finding-variance-and-expectation-of-boolean-variable

def bool2stats (bool_list):
  proportion = sum(bool_list)/len(bool_list)
  variance = proportion - (proportion ** 2)
  sem = (variance / len(bool_list)) ** (1/2)
  return proportion, sem, variance

def separate_stats (stats:list): # go from list of tuples (p, sem, var) to 3 lists for p, sem and var
  p_rollout = [s[0] for s in stats]
  sem_rollout = [s[1] for s in stats]
  var_rollout = [s[2] for s in stats]
  return p_rollout, sem_rollout, var_rollout