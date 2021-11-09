from pandas import read_csv
from matplotlib import pyplot
from pandas.plotting import scatter_matrix




names = ["sepal-length", "sepal-width", "petal-lenght", "petal-width", "class"]
dataset = read_csv("iris.csv", names = names)
print( dataset.shape)
print(dataset.head(20))
print (dataset.describe())
print(dataset.groupby("class").size())
dataset.plot(kind="box", subplots=True, layout= (2,2), sharex=False, sharey= False)
pyplot.show()
dataset.hist()
pyplot.show()
scatter_matrix(dataset)
pyplot.show()
