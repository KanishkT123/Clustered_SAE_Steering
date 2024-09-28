# SAE LENS HELPERS FOR STEERING

from tqdm import tqdm
from functools import partial 

def list_flatten(nested_list):
    return [x for y in nested_list for x in y]

# A very handy function Neel wrote to get context around a feature activation
def make_token_df(tokens, len_prefix=5, len_suffix=3, model = model):
    str_tokens = [model.to_str_tokens(t) for t in tokens]
    unique_token = [[f"{s}/{i}" for i, s in enumerate(str_tok)] for str_tok in str_tokens]
    
    context = []
    prompt = []
    pos = []
    label = []
    for b in range(tokens.shape[0]):
        for p in range(tokens.shape[1]):
            prefix = "".join(str_tokens[b][max(0, p-len_prefix):p])
            if p==tokens.shape[1]-1:
                suffix = ""
            else:
                suffix = "".join(str_tokens[b][p+1:min(tokens.shape[1]-1, p+1+len_suffix)])
            current = str_tokens[b][p]
            context.append(f"{prefix}|{current}|{suffix}")
            prompt.append(b)
            pos.append(p)
            label.append(f"{b}/{p}")
    # print(len(batch), len(pos), len(context), len(label))
    return pd.DataFrame(dict(
        str_tokens=list_flatten(str_tokens),
        unique_token=list_flatten(unique_token),
        context=context,
        prompt=prompt,
        pos=pos,
        label=label,
    ))


def find_max_activation(model, sae, activation_store, feature_idx, num_batches=100):
    '''
    Find the maximum activation for a given feature index. This is useful for 
    calibrating the right amount of the feature to add.
    '''
    max_activation = 0.0

    pbar = tqdm(range(num_batches))
    for _ in pbar:
        tokens = activation_store.get_batch_tokens()
        
        _, cache = model.run_with_cache(
            tokens, 
            stop_at_layer=sae.cfg.hook_layer + 1, 
            names_filter=[sae.cfg.hook_name]
        )
        sae_in = cache[sae.cfg.hook_name]
        feature_acts = sae.encode(sae_in).squeeze()

        feature_acts = feature_acts.flatten(0, 1)
        batch_max_activation = feature_acts[:, feature_idx].max().item()
        max_activation = max(max_activation, batch_max_activation)
        
        pbar.set_description(f"Max activation: {max_activation:.4f}")

    return max_activation

def steering(activations, hook, steering_strength=1.0, steering_vector=None, max_act=1.0):
    # Note if the feature fires anyway, we'd be adding to that here.
    return activations + max_act * steering_strength * steering_vector

def generate_with_steering(model, sae, prompt, steering_feature, max_act, steering_strength=1.0, max_new_tokens=95):
    input_ids = model.to_tokens(prompt, prepend_bos=sae.cfg.prepend_bos)
    
    steering_vector = sae.W_dec[steering_feature].to(model.cfg.device)
    
    steering_hook = partial(
        steering,
        steering_vector=steering_vector,
        steering_strength=steering_strength,
        max_act=max_act
    )
    
    # standard transformerlens syntax for a hook context for generation
    with model.hooks(fwd_hooks=[(sae.cfg.hook_name, steering_hook)]):
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            stop_at_eos = False if device == "mps" else True,
            prepend_bos = sae.cfg.prepend_bos,
        )
    
    return model.tokenizer.decode(output[0])

# # Choose a feature to steer
# steering_feature = steering_feature = 20115  # Choose a feature to steer towards

# # Find the maximum activation for this feature
# max_act = find_max_activation(model, sae, activation_store, steering_feature)
# print(f"Maximum activation for feature {steering_feature}: {max_act:.4f}")

# # note we could also get the max activation from Neuronpedia (https://www.neuronpedia.org/api-doc#tag/lookup/GET/api/feature/{modelId}/{layer}/{index})

# # Generate text without steering for comparison
# prompt = "Once upon a time"
# normal_text = model.generate(
#     prompt,
#     max_new_tokens=95, 
#     stop_at_eos = False if device == "mps" else True,
#     prepend_bos = sae.cfg.prepend_bos,
# )

# print("\nNormal text (without steering):")
# print(normal_text)

# # Generate text with steering
# steered_text = generate_with_steering(model, sae, prompt, steering_feature, max_act, steering_strength=2.0)
# print("Steered text:")
# print(steered_text)