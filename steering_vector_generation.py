import torch
from functools import partial

def steering_hook(resid_pre, hook, sae_out, steering_on, steering_vector):
    if resid_pre.shape[1] == 1:
        return

    position = sae_out.shape[1]
    if steering_on:
      # using our steering vector
      resid_pre[:, :position - 1, :] += steering_vector

def hooked_generate(prompt_batch, model, fwd_hooks=[], seed=None, **kwargs):
    if seed is not None:
        torch.manual_seed(seed)

    with model.hooks(fwd_hooks=fwd_hooks):
        tokenized = model.to_tokens(prompt_batch)
        result = model.generate(
            input=tokenized,
            max_new_tokens=50,
            do_sample=True,
            **kwargs)
    return result

def create_average_steering_vector(feature_set, sae, multiplier, device):
  steering_vectors = torch.stack([sae.W_dec[feature_id] for feature_id in feature_set])
  coefficient_magic = (multiplier/len(steering_vectors))
  coefficients = torch.ones(len(steering_vectors))*coefficient_magic
  coefficients = coefficients.view(-1, 1)
  steering_vector = coefficients.to(device) * steering_vectors.to(device)
  steering_vector = torch.sum(steering_vector, dim=0)
  return steering_vector

def create_diff_steering_vector(feature_set_a, feature_set_b, sae, multiplier, device):
  steering_a = create_average_steering_vector(feature_set_a, sae, multiplier, device)
  steering_b = create_average_steering_vector(feature_set_b, sae, multiplier, device)
  return steering_a - steering_b

def run_generate_n(model, layer, example_prompt, steering_vector, steering_on, sae_out, n, sampling_kwargs):
  model.reset_hooks()

  # create a partial function with the steering vector
  steering_hook_prefilled = partial(steering_hook, steering_on=steering_on, steering_vector=steering_vector, sae_out=sae_out)

  editing_hooks = [(f"blocks.{layer}.hook_resid_post", steering_hook_prefilled)]
  res = hooked_generate([example_prompt]*n, editing_hooks, seed=None, **sampling_kwargs)

  # return results, removing the ugly beginning of sequence token
  res_str = model.to_string(res[:, 1:])
  return res_str