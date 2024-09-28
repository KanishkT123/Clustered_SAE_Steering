import torch
import re
from functools import partial
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
eos_token = tokenizer.eos_token_id

def steering_hook(resid_pre, hook, sae_out, steering_on, steering_vector):
    if resid_pre.shape[1] == 1:
        return

    position = sae_out.shape[1]
    if steering_on:
      # using our steering vector
      resid_pre[:, :position - 1, :] += steering_vector

def remove_eos_token(tensor, eos_token_id=eos_token):
    # Create a mask where True indicates non-EOS tokens
    mask = tensor != eos_token_id
    mask[0] = True # keep BOS token

    # Use the mask to remove EOS tokens
    cleaned_tensor = tensor[mask]
    return cleaned_tensor

def hooked_generate(prompt_batch, model, fwd_hooks=[], seed=None, max_tokens=50, num_clamped_tokens=0, forget=False, **kwargs):
    if seed is not None:
        torch.manual_seed(seed)

    remain_tokens = max_tokens - num_clamped_tokens
    print("remaining tokens after clamp", remain_tokens)

    if num_clamped_tokens == 0:
      print("Case 1: Unconditional Steering")
      with model.hooks(fwd_hooks=fwd_hooks):
          tokenized_context = model.to_tokens(prompt_batch)
          steered_result = model.generate(
            input=tokenized_context,
            max_new_tokens=remain_tokens,
            do_sample=True,
            eos_token_id=None,
            **kwargs
        )
    # return here if non-conditional
      return steered_result, tokenized_context

    else:
      print("Case 2: Conditional Steering")
    # Case 2: Steer for num_clamped_tokens tokens, then generate the remaining tokens without steering
      eos_token_id = 50256

      # Step 1: Steer for num_clamped_tokens tokens using hooks
      with model.hooks(fwd_hooks=fwd_hooks):
        tokenized = model.to_tokens(prompt_batch)
        steered_result = model.generate(
            input=tokenized,
            max_new_tokens=num_clamped_tokens,
            do_sample=True,
            eos_token_id=None,
            **kwargs
        )
        # print(f"steered result: {steered_result}")

      # Case 2.1: conditional steering second part has original context
      # TODO: Forgetting not working now; needs a fix
      # Kanishk: 10 tokens as the new input
      if forget:
        print("Case 2.1: reset hooks")
        model.reset_hooks() # reset hooks before new generation
        context = torch.cat([tokenized, steered_result[:, tokenized.shape[1]:]], dim=1)
        tokenized = model.to_tokens(prompt_batch)
        next_result = model.generate(
            input=context,
            max_new_tokens=remain_tokens,
            eos_token_id=None,
            do_sample=True,
            **kwargs)

      # Case 2.2: model "remembers" its steered outputs
      else:
        print("Case 2.2: hooks not reset")
        context = torch.cat([tokenized, steered_result[:, tokenized.shape[1]:]], dim=1)
        next_result = model.generate(
            input=context,
            max_new_tokens=remain_tokens,
            eos_token_id=None,
            do_sample=True,
            **kwargs)
        
      # steered_result = cut_off_output(steered_result, eos_token_id, num_clamped_tokens)
      print(f"steered result shape: {steered_result.shape}")
      print(f"next result shape: {next_result.shape}")
      final_result = torch.cat([steered_result, next_result], dim=1)

      # apply this function to each sequence in final_result
      cleaned_results = [remove_eos_token(seq) for seq in final_result]
      cleaned_results = torch.cat([torch.unsqueeze(torch.tensor(seq), dim=0) for seq in cleaned_results], dim=0) # convert list into tensor

      print(f"cleaned_results: {type(cleaned_results)}")

      print("type of context" ,type(context))

    return cleaned_results, context

def create_average_steering_vector(feature_set, sae, multiplier, device):
  steering_vectors = torch.stack([sae.W_dec[feature_id] for feature_id in feature_set])
  coefficient_magic = (multiplier/len(steering_vectors))
  coefficients = torch.ones(len(steering_vectors))*coefficient_magic
  coefficients = coefficients.view(-1, 1)
  steering_vector = coefficients.to(device) * steering_vectors.to(device)
  steering_vector = torch.sum(steering_vector, dim=0)
  return steering_vector

# A version of the steering vector without normalization
def create_additive_steering_vector(feature_set, sae, multiplier, device):
  steering_vectors = torch.stack([sae.W_dec[feature_id] for feature_id in feature_set])
  #print("multiplier is", multiplier)
  coefficients = torch.ones(len(steering_vectors))*multiplier
  coefficients = coefficients.view(-1, 1)
  steering_vector = coefficients.to(device) * steering_vectors.to(device)
  steering_vector = torch.sum(steering_vector, dim=0)
  return steering_vector

# SAE(positive) - SAE(negative)
def create_diff_steering_vector(feature_set_a, feature_set_b, sae, multiplier, device):
  steering_a = create_average_steering_vector(feature_set_a, sae, multiplier, device)
  steering_b = create_average_steering_vector(feature_set_b, sae, multiplier, device)
  return steering_a - steering_b

def remove_redundant_phrase(text: str, phrase: str) -> str:
    """
    Removes all redundant occurrences of 'phrase' in 'text', keeping only the first occurrence.
    
    Args:
        text (str): The generated text from the LLM.
        phrase (str): The phrase to remove duplicates of.
    
    Returns:
        str: The cleaned text with redundant phrases removed.
    """
    # Escape special characters in the phrase for regex
    escaped_phrase = re.escape(phrase)
    
    # Use a regex pattern to find all occurrences after the first
    pattern = f"({escaped_phrase})"
    
    # This lambda function keeps the first occurrence and removes the rest
    def replacer(match, first=[True]):
        if first[0]:
            first[0] = False
            return match.group(0)
        return ''
    
    # Substitute redundant phrases with an empty string
    cleaned_text = re.sub(pattern, replacer, text)
    
    # Optionally, remove any extra whitespace that might result from the removal
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    return cleaned_text


def run_generate_n(model, 
                   layer, 
                   example_prompt, 
                   steering_vector, 
                   steering_on, 
                   sae_out, 
                   n, 
                   sampling_kwargs, 
                   coefficient, 
                   conditional: bool = False, 
                   forget = False, 
                   num_clamped_tokens: int = None) -> str:

  print(f"conditional: {conditional}")
  print(f"forget: {forget}")
  print(f"num_clamped_tokens: {num_clamped_tokens}")
  print(f"coefficient: {coefficient}")
  
  if not isinstance(n, int) or n <= 0:
    raise ValueError("Parameter 'n' must be a positive integer.")

  model.reset_hooks() #make sure no other hooks are set before steering

  # create a partial function with the steering vector
  steering_hook_prefilled = partial(steering_hook, steering_on=steering_on, steering_vector=steering_vector, sae_out=sae_out)

  editing_hooks = [(f"blocks.{layer}.hook_resid_post", steering_hook_prefilled)]

  # give user freedom to choose number of tokens to generate with sae steering
  if conditional:
    res, context = hooked_generate([example_prompt]*n, model, editing_hooks, seed=None, forget = forget, **sampling_kwargs, num_clamped_tokens = num_clamped_tokens)
  else :
    res, context = hooked_generate([example_prompt]*n, model, editing_hooks, seed=None, **sampling_kwargs)

  # return results, removing the ugly beginning of sequence token
  res_str = model.to_string(res[:, 1:])
  context_str = model.to_string(context[:, 1:])

  # post-process to remove the redundant context
  print("res_str length", len(res_str))
  print("context_str length", len(context_str))
  if conditional:
    cleaned_res_str_list = [
      remove_redundant_phrase(text, phrase) 
      for text, phrase in zip(res_str, context_str)
    ]
  else:
    cleaned_res_str_list = res_str
  return cleaned_res_str_list

def pretty_print_outputs(result):
   print(("\n\n" + "-" * 80 + "\n\n").join(result))
