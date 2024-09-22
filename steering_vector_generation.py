import torch
from functools import partial

# copied steering_vector_generation code because import was not working

def steering_hook(resid_pre, hook, sae_out, steering_on, steering_vector):
    if resid_pre.shape[1] == 1:
        return

    position = sae_out.shape[1]
    if steering_on:
      # using our steering vector
      resid_pre[:, :position - 1, :] += steering_vector

def hooked_generate(prompt_batch, model, fwd_hooks=[], seed=None, max_tokens=50, num_clamped_tokens=0, forget=False, **kwargs):
    if seed is not None:
        torch.manual_seed(seed)

    remain_tokens = max_tokens - num_clamped_tokens
    print("remaining tokens after clamp", remain_tokens)

    with model.hooks(fwd_hooks=fwd_hooks):
        tokenized = model.to_tokens(prompt_batch)
        result = model.generate(
            input=tokenized,
            max_new_tokens=50,
            do_sample=True,
            **kwargs)
    
    # return here if non-conditional
    if num_clamped_tokens == 0:
      return result

    model.reset_hooks() #reset hooks

    if forget:
      model.reset_hooks() # just to make sure
      next_result = model.generate(
          input=tokenized,
          max_new_tokens=remain_tokens,
          do_sample=True,
          **kwargs)
    else:
      context = torch.cat([tokenized, result[:, tokenized.shape[1]:]], dim=1)
      next_result = model.generate(
          input=context,
          max_new_tokens=remain_tokens,
          do_sample=True,
          **kwargs)
      
    final_result = torch.cat([result, next_result], dim=1)

    return final_result
# def hooked_generate(prompt_batch, 
#                     model, 
#                     fwd_hooks=[], 
#                     seed=None, 
#                     max_tokens=50, 
#                     forget = False, #false unless specified
#                     clamped_tokens=0, #no clamped tokens unless specified
#                     **kwargs):
#     if seed is not None:
#         torch.manual_seed(seed)

#     remain_tokens = max_tokens - clamped_tokens
#     print("remaining tokens after clamp", remain_tokens)

#     with model.hooks(fwd_hooks=fwd_hooks):
#         tokenized = model.to_tokens(prompt_batch)
        
#         # Generate clamped tokens
#         clamped_result = model.generate(input=tokenized, max_new_tokens=clamped_tokens,do_sample=True,**kwargs)
        
#     # Reset hooks for normal generation post-clamping
#     model.reset_hooks()

#     if forget:
#       context = tokenized[:, clamped_tokens:]
      
#     # Concatenate original input with clamped result
#     else: 
#       context = torch.cat([tokenized, clamped_result[:, tokenized.shape[1]:]], dim=1)
    
#     print("context", context)

#     # Generate remaining tokens normally
#     final_result = model.generate(
#         input=context,
#         max_new_tokens=remain_tokens,
#         do_sample=True,
#         **kwargs)
#     return final_result

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
  print("multiplier is", multiplier)
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


# def run_generate_n(model, layer, example_prompt, steering_vector, steering_on, sae_out, n, sampling_kwargs):
#   model.reset_hooks()

#   # create a partial function with the steering vector
#   steering_hook_prefilled = partial(steering_hook, steering_on=steering_on, steering_vector=steering_vector, sae_out=sae_out)

#   editing_hooks = [(f"blocks.{layer}.hook_resid_post", steering_hook_prefilled)]
#   res = hooked_generate([example_prompt]*n, model, editing_hooks, seed=None, **sampling_kwargs)

#   # return results, removing the ugly beginning of sequence token
#   res_str = model.to_string(res[:, 1:])
#   return res_str
def run_generate_n( model, layer, example_prompt, steering_vector, steering_on, sae_out, n, sampling_kwargs, conditional: bool = False, forget = False, num_clamped_tokens: int = None) -> str:
  
  if not isinstance(n, int) or n <= 0:
    raise ValueError("Parameter 'n' must be a positive integer.")

  model.reset_hooks() #reset hooks to forget

  # create a partial function with the steering vector
  steering_hook_prefilled = partial(steering_hook, steering_on=steering_on, steering_vector=steering_vector, sae_out=sae_out)

  editing_hooks = [(f"blocks.{layer}.hook_resid_post", steering_hook_prefilled)]
  
  # give user freedom to choose number of tokens to generate with sae steering
  if conditional:
    res = hooked_generate([example_prompt]*n, model, editing_hooks, seed=None, forget = forget, **sampling_kwargs, num_clamped_tokens = num_clamped_tokens)
  else :
    res = hooked_generate([example_prompt]*n, model, editing_hooks, seed=None, **sampling_kwargs)
  
  # return results, removing the ugly beginning of sequence token
  res_str = model.to_string(res[:, 1:])
  return res_str

def pretty_print_outputs(result):
   print(("\n\n" + "-" * 80 + "\n\n").join(result))
