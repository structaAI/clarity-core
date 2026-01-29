import yaml
from types import SimpleNamespace
from typing import Any, Dict

def load_config(config_path: str)-> SimpleNamespace:
  with open(config_path, 'r') as file:
    config_dict = yaml.safe_load(file)
  
  def dict_to_namespace(d: Any)-> Any:
    if isinstance(d, dict):
      return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    return d
  
  return dict_to_namespace(config_dict)