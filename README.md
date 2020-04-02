## Cookeroo

### Basic Usage
```python
from cookeroo import DataPrep, CookerooModel

d = DataPrep('/Users/abhilash1in/Documents/Projects/Cookeroo/data/', 'wav')
d.slice(3000)
d.export()

model = CookerooModel('/Users/abhilash1in/Documents/Projects/Cookeroo/data/', 'wav')
model.train()
```