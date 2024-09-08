import torch

def get_feature_acts_and_sae_out(model, sae, prompt):
  hook_point = sae.cfg.hook_name
  _, cache = model.run_with_cache(prompt, prepend_bos=True)
  sv_feature_acts = sae.encode(cache[hook_point].to(sae.device))
  sae_out = sae.decode(sv_feature_acts)
  return sv_feature_acts, sae_out

def get_value_feature_pairs(top_acts, num_features):
  '''Get Top features from a torch.topk tensor
  that has both values and indices
  num_features: Number of features to get
  Returns a tensor with pairs of values and indices'''
  values = top_acts.values
  indices = top_acts.indices

  flattened_values = values.flatten()
  flattened_indices = indices.flatten()

  # Get the top k values and their indices in the flattened tensor
  topk_values, topk_indices = torch.topk(flattened_values, num_features)

  # Map back to the original indices
  topk_original_indices = flattened_indices[topk_indices]

  # Combine values and indices into pairs
  topk_pairs = torch.stack((topk_values, topk_original_indices), dim=1)
  return topk_pairs

def add_to_dicts(positive_prompt_flag, topk_pairs, pos_sum_dict, pos_count_dict, neg_sum_dict, neg_count_dict):
  '''Positive Prompt Flag is used to tell if this count goes to
  positive or negative dictionary
  Adds values to value dict and count to count dict'''
  for value, index in topk_pairs:
    feature = int(index.item())
    act_value = float(value.item())
    if positive_prompt_flag:
      pos_sum_dict[feature] += act_value
      pos_count_dict[feature] += 1
    else:
      neg_sum_dict[feature] += act_value
      neg_count_dict[feature] += 1
