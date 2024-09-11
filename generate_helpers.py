import pandas as pd
from pandas import DataFrame
from pathlib import Path
from sae_lens import SAE, HookedSAETransformer
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory


# LOAD DATASET FROM .JSON OR .JSONL
def load_json_dataset(str_path:str):
  '''
  load_json_dataset(path_to_file)
  Takes in a .json or .jsonl dataset and loads it into a DataFrame
  '''
  path = Path(str_path)
  if path.suffix == '.json':
      df = pd.read_json(str_path)
  elif path.suffix == '.jsonl' :
      df = pd.read_json(str_path,lines=True)
  else: 
      raise ValueError('Invalid file type: Currently handles .json and .jsonl')
  return df

# LOAD MULTIPLE SAES AND GET A LIST OF OUTPUTS
def load_pretrained_SAEs (sae_release:str, sae_id_list:list, device) -> list:
    '''
    load_SAEs : Instead of loading each pretrained SAE individually, 
    this helps you load all the SAEs you want all at once 
    and outputs a list of the SAEs, and corresponding lists
    with their cfg dicts and sparsity values (note: ok if sparsity values are None) 
    '''

    #initialize
    n_saes = len(sae_id_list)
    sae_list = [None]*n_saes
    cfg_dict_list = [None]*n_saes
    sparsity_list = [None]*n_saes


    for i in range(n_saes):
        sae_list[i], cfg_dict_list[i], sparsity_list[i] = \
        SAE.from_pretrained(
                    release = sae_release, # <- Release name
                    sae_id = sae_id_list[i], # <- SAE id (not always a hook point!)
                    device = device
                )
    return sae_list, cfg_dict_list, sparsity_list



# LOAD SAE DIRECTORY AND FILTER BY MODEL AND RELEASES OF INTEREST
# ADDED TO GENERATE_HELPERS.PY : LOAD SAE DIRECTORY AND FILTER BY MODEL AND RELEASES OF INTEREST
def sae_directory_info(model=None, release=None, exact_match_model:bool=True, exact_match_release:bool=False)-> DataFrame:
    '''
    Returns a filtered version of the sae directory data frame - using get_pretrained_saes_directory of SAELens
    if model is None, then returns everything that  get_pretrained_saes_directory of SAELens would
    Model and Release can be keyword that should be in the name e.g. 'gemma-scope' or an exact name
    By default, need to match the model exactly but not the release
    '''
    # directly from tutorial 2.0 of sae lens
    model_df = pd.DataFrame.from_records({k:v.__dict__ for k,v in get_pretrained_saes_directory().items()}).T
    model_df.drop(columns=["expected_var_explained", "expected_l0", "config_overrides", "conversion_func"], inplace=True)

    # case: return everything
    if model is None and release is None: # get everything
        return model_df

    # case Model is not None
    if model is not None:
        if exact_match_model:
            model_df = model_df[model_df['model']==model]
        else:
            model_df = model_df[[model in r for r in model_df['model'].to_list()]]

    # case release is not None
    if release is not None:
        if exact_match_release:
            model_df = model_df[model_df['release']==release]
        else:
            model_df = model_df[[release in r for r in model_df['release'].to_list()]]
    return model_df


