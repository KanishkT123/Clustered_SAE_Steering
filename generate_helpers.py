import pandas as pd
from pathlib import Path
from sae_lens import SAE, HookedSAETransformer


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


def load_SAEs (sae_release:str, sae_id_list:list, device) -> list:
    '''
        Instead of loading each SAE individually, this helps you load all the SAEs you want all at once and outputs a list of the SAEs 
    '''
    #initialize
    n_saes = len(sae_list)
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

