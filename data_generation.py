from .steering_vector_generation import run_generate_n
from .feature_filtering import get_feature_acts_and_sae_out
import torch
import pickle
from transformers import BitsAndBytesConfig
from transformer_lens import HookedTransformer

def load_model(model_name, device, hf_token):
  if model_name == "google/gemma-2b":
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    model = HookedTransformer.from_pretrained(model_name, device=device, token=hf_token, quantization_config=quantization_config)
  else:
    model = HookedTransformer.from_pretrained(model_name, device=device)
  model.eval()
  return model

def generate_and_save_data(model, sae, steering_vector_func, layer,
                           drive_path,
                           coefficient_range=range(0, 1000, 100),
                           sample_count=50,
                           sample_prompt="My mood is",
                           sampling_kwargs = dict(temperature=1.0, top_p=0.5, freq_penalty=1.0)
                           ):
  '''
  Generates and saves large datasets to a Google drive path
  Steering_vector_func is a function that needs to take a coefficient input
  and output a tensor steering vector
  '''
  results_dict = {}

  with torch.no_grad():
    _, sae_out = get_feature_acts_and_sae_out(model, sae, sample_prompt)
    for coefficient in coefficient_range:
        # Generate the steering vector using the provided function
        steering_vector = steering_vector_func(coefficient)
        steering_on = True
        # Generate results based on the sample prompt
        results_steered = run_generate_n(model, layer, sample_prompt, steering_vector, steering_on, sae_out, sample_count, sampling_kwargs)
        # Store the results and breakage data in the respective dictionaries
        results_dict[coefficient] = results_steered

  with open(drive_path, 'wb') as f:
    pickle.dump(results_dict, f)

  return results_dict
