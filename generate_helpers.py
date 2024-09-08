import pandas as pd
from pathlib import Path

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
      raise ValueError('Invalid file type') 
  return df



