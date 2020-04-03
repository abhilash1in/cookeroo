## Cookeroo

### Basic Usage
```python
import os
from cookeroo import DataPrep, CookerooModel

d = DataPrep('/Users/abhilash1in/Documents/Projects/Cookeroo/data/', 'wav')
d.slice(3000)
d.export()

base_path = '/Users/abhilash1in/Documents/Projects/Cookeroo'
data_base_path = os.path.join(base_path, 'data')
model_base_path = os.path.join(base_path, 'models')

model = CookerooModel(data_base_path, model_base_path, 'wav')
model.train()
```