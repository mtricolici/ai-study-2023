import os
import importlib

current_dir = os.path.dirname(__file__)

module_files = [f for f in os.listdir(current_dir)
                if os.path.isfile(os.path.join(current_dir, f)) and
                   f.endswith('.py') and f != '__init__.py']

for file in module_files:
  module_name = file[:-3]  # Remove the '.py' from the filename
  importlib.import_module('.' + module_name, package=__name__)

