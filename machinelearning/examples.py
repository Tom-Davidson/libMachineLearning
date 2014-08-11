from numpy import loadtxt

class examples:
	"""
	Training examples for a machine learning problem
	"""
	X = {
		'train':	[],
		'cv':		[],
		'test':		[]
	}
	y = {
		'train':	[],
		'cv':		[],
		'test':		[]
	}
	m = 0
	noFeatures = 0
	def __init__(self, filename):
		data = loadtxt(filename, delimiter=',')
		self.m = data.shape[0]
		self.noFeatures = data.shape[1] - 1	# Last col is our answers 'y'
		X = data[:, 0:self.noFeatures]		# Zero indexed
		y = data[:, self.noFeatures ]
		mTwenty = round(self.m/5)
		self.X['train'] = X[0:(self.m-2*mTwenty), :]
		self.X['cv'] = X[(self.m-2*mTwenty):(self.m-mTwenty), :]
		self.X['test'] = X[(self.m-mTwenty):self.m, :]
		self.y['train'] = y[0:(self.m-2*mTwenty)]
		self.y['cv'] = y[(self.m-2*mTwenty):(self.m-mTwenty)]
		self.y['test'] = y[(self.m-mTwenty):self.m]

	def getX(self, type):
		return self.X[type]
	def getY(self, type):
		return self.y[type]
