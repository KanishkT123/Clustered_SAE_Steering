import pandas as pd
from pandas import DataFrame
from pathlib import Path
from sae_lens import SAE, HookedSAETransformer
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory


# LOAD DATASET FROM .JSON OR .JSONL INTO A DATAFRAME
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

######################################################################################################
### FUNCTIONS TO FILTER OUT DATAFRAME RESULTS TO HELP LOAD APPROPRIATE SAES ####
### GEMMA-SCOPE SPECIFIC FUNCTIONS UNLESS OTHERWISE SPECFIED  ################## 
### WILL  NEED TO UPDATE IN THE FUTURE DEPENDING ON WHAT PPL NAME THEIR SAES #####  
######################################################################################################
def get_saeids_for_layer(sae_id_list:list, layer:int=0, width:int=16, L0:int=None):
    '''
    get_saeids_for_layer(sae_id_list:list, layer:int=0, width:int=16)

    WORKS WITH GEMMA-SCOPE AND GPT2-SMALL RIGHT NOW
    FOR GEMMA: assumes that there will be a layer_XX and a width_YY somewhere in the name
    FOR GPT SMALL: assumes there will be blocks.XX. in the name
    '''
    if L0==None:
        # first tries gemmascope expectation
        newlist= [s for s in sae_id_list if (f'width_{width}' in s and f'layer_{layer}/' in s)]
        # then tries gpt2small exepctation
        if len(newlist) == 0:
            newlist= [s for s in sae_id_list if (f'blocks.{layer}.' in s)]

        return newlist
    else:
        newlist= [s for s in sae_id_list if (f'width_{width}' in s and f'layer_{layer}/' in s and f'l0_{L0}' in s)]


def get_sae_ids_with_L0_dict(model_df:DataFrame, width=16,layer_lo=None):
    '''
    FOR GEMMASCOPE, the SAE_IDs and Neuronpedia IDs in the sae directory are a fucked
    layer_lo is a dict with integer key,value pairs {layer: L0}
      '''
    if layer_lo==None:
        layer_lo = {0: 105,
          1: 102,
          2: 141,
          3: 59,
          4: 124,
          5: 68,
          6: 70,
          7: 69,
          8: 71,
          9: 73,
          10: 77,
          11: 80,
          12: 82,
          13: 84,
          14: 84,
          15: 78,
          16: 78,
          17: 77,
          18: 74,
          19: 73,
          20: 71,
          21: 70,
          22: 72,
          23: 75,
          24: 73,
          25: 116
      }
    
      # initialize list
    sae_id_list = [None]*len(layer_lo)


    # THIS  - [0] - IS WHERE THE SINGLE DATAFRAME ENTRY COMES IN, CAN BE FIXED IN THE FUTURE
    all_sae_ids = list(model_df.saes_map.to_list()[0].keys())


    for layer, L0 in layer_lo.items():
        layer_sae_ids = get_saeids_for_layer(all_sae_ids, layer=layer, width=width, L0=L0)
        sae_id_list[layer] = layer_sae_ids[0]

    return sae_id_list


def parse_sae_LO(sae_id):
    '''
    FOR GEMMASCOPE
    assumes that there will be a _l0_X at the END OF THE NAME
    '''
    return int(sae_id.split('_l0_')[-1])

def parse_sae_layer(sae_id):
    '''
    FOR GEMMASCOPE
    assumes that there will be a 'layer_XX/' at the BEGINNING OF THE NAME
    '''
    return int(sae_id.split('/')[0].split('_')[-1])

def lowest_L0_sae_id(sae_id_list:list):
    '''
    FOR GEMMASCOPE
    assumes that there will be a _l0_X at the END OF THE NAME
    '''
    return min(sae_id_list, key=parse_sae_LO)


def get_lowest_L0_sae_id_for_each_layer(model_df:DataFrame, layer=None, width=16):
    '''
    FOR GEMMASCOPE
    This function takes DataFrame with a SINGLE ENTRY (single release) produced by sae_directory_info
    e.g. model_df = sae_directory_info(model=model_name, release=release_id, exact_match_release=True)
    Then it finds the lowest L0 for each layer of interest by name, not by dataframe
    (WE SHOULD CHECK: tutorial 2.0 removes the L0 column, so I think it's the correc way?)
    # layer should be a list of natural numbers or None
    If Layer is None, returns lowest L0 sae_id for each of all layers'''

    # THIS  - [0] - IS WHERE THE SINGLE DATAFRAME ENTRY COMES IN, CAN BE FIXED IN THE FUTURE
    all_sae_ids = list(model_df.saes_map.to_list()[0].keys())

    # layer is a list of natural numbers
    if layer is None:
       layer = list(set([parse_sae_layer(s) for s in all_sae_ids]))

    # initialize
    nlayers = len(layer)
    lowest_L0_sae_id_list = [None]*nlayers

    for idx, ilayer in enumerate(layer):
        layer_sae_ids = get_saeids_for_layer(all_sae_ids, layer=ilayer, width=width)
        if len(layer_sae_ids) == 0:
            print(f'No SAEs found for layer {ilayer}')
            continue
        elif len(layer_sae_ids) == 1:
            lowest_L0_sae_id_list[idx]= layer_sae_ids
        else:
            lowest_L0_sae_id_list[idx] = lowest_L0_sae_id(layer_sae_ids)

    return lowest_L0_sae_id_list

