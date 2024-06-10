from pathlib import Path
import sys

class TestUtils: 
  
  """
    This class can be defined at the top of a unit test file in order
    to modify the interpreter's module search path to include the 
    modules present in the src folder in the test directory. This allows 
    unit test files to import the modules in the src directory.
  """


  def __init__(self):
    sys.path.append(str(Path(__file__).parent.parent / 'src'))