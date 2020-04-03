from cookeroo import DataPrep, CookerooModel

# d = DataPrep('/Users/ayushbihani/Downloads/Personal projects/cookeroo/data', 'wav')
# d.slice(3000)
# d.export()

base_path = '/Users/ayushbihani/Downloads/Personal projects/cookeroo/'
model = CookerooModel(base_path + 'data', base_path, 'wav')
model.train()
model.predict("/Users/ayushbihani/Desktop/data/sliced/test/1.wav")
