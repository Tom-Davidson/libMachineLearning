from pylab import figure, title, scatter, show, legend, xlabel, ylabel

class mlgraph():
	"""
	Graphing Helper
	"""
	figures = {}

	def __init__(self):
		False

	def scatter(self, name, X, y, labels):
		# Get (or set up) the figure using the name
		if name not in self.figures:
			self.figures[name] = len(self.figures.keys())
		figure(self.figures[name])
		title(name)
		m = X.shape[0]
		for i in range(m):
			if y[i] == 1:
				scatter(X[i, 0], X[i, 1], marker='o', c='g')
			else:
				scatter(X[i, 0], X[i, 1], marker='x', c='r')
		xlabel(labels[0])
		ylabel(labels[1])
		#legend(['Negative', 'Positive'])

	def show(self):
		show()
