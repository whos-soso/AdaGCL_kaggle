import pickle
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, dok_matrix
from Params import args
import scipy.sparse as sp
from Utils.TimeLogger import log
import torch as t
import torch.utils.data as data  #允许用户自定义数据集类来加载和处理自己的数据
import torch.utils.data as dataloader

class DataHandler:
	def __init__(self):
		if args.data == 'yelp':
			predir = './Datasets/sparse_yelp/'
		elif args.data == 'lastfm':
			predir = './Datasets/lastFM/'
		elif args.data == 'beer':
			predir = './Datasets/beerAdvocate/'
		self.predir = predir
		self.trnfile = predir + 'trnMat.pkl'
		self.tstfile = predir + 'tstMat.pkl'

	def loadOneFile(self, filename):
		with open(filename, 'rb') as fs:
			ret = (pickle.load(fs) != 0).astype(np.float32)
		if type(ret) != coo_matrix:
			ret = sp.coo_matrix(ret)
		return ret

	def normalizeAdj(self, mat):
		degree = np.array(mat.sum(axis=-1))
		dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
		dInvSqrt[np.isinf(dInvSqrt)] = 0.0
		dInvSqrtMat = sp.diags(dInvSqrt)
		return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()

	def makeTorchAdj(self, mat):
		# make ui adj
		a = sp.csr_matrix((args.user, args.user))
		b = sp.csr_matrix((args.item, args.item))
		mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
		mat = (mat != 0) * 1.0
		mat = (mat + sp.eye(mat.shape[0])) * 1.0
		mat = self.normalizeAdj(mat)

		# make cuda tensor
		idxs = t.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
		vals = t.from_numpy(mat.data.astype(np.float32))
		shape = t.Size(mat.shape)
		return t.sparse.FloatTensor(idxs, vals, shape).cuda()

	def LoadData(self):
		trnMat = self.loadOneFile(self.trnfile)
		tstMat = self.loadOneFile(self.tstfile)
		self.trnMat = trnMat
		args.user, args.item = trnMat.shape
		self.torchBiAdj = self.makeTorchAdj(trnMat)
		trnData = TrnData(trnMat)
		self.trnLoader = dataloader.DataLoader(trnData, batch_size=args.batch, shuffle=True, num_workers=0)
		tstData = TstData(tstMat, trnMat)
		self.tstLoader = dataloader.DataLoader(tstData, batch_size=args.tstBat, shuffle=False, num_workers=0)

class TrnData(data.Dataset):
	def __init__(self, coomat):
		self.rows = coomat.row
		self.cols = coomat.col
		self.dokmat = coomat.todok()
		self.negs = np.zeros(len(self.rows)).astype(np.int32)

	def negSampling(self): #负采样：随机找一个没交互信息的物品作为负样本
		for i in range(len(self.rows)):
			u = self.rows[i]
			while True:
				iNeg = np.random.randint(args.item)
				if (u, iNeg) not in self.dokmat:
					break
			self.negs[i] = iNeg

	def __len__(self):
		return len(self.rows)

	def __getitem__(self, idx):
		return self.rows[idx], self.cols[idx], self.negs[idx] #根据索引 idx 返回一个测试样本，包括测试用户索引 tstUsrs[idx] 和该用户在训练数据矩阵中的邻接向量（转换为一维数组）。

class TstData(data.Dataset):


	
	
	
	
	
	def __init__(self, coomat, trnMat):  #初始化测试数据集类，接收测试数据矩阵 coomat 和训练数据矩阵 trnMat。
		self.csrmat = (trnMat.tocsr() != 0) * 1.0  # 将训练数据矩阵转换为压缩稀疏行矩阵 csrmat，并将非零元素设置为 1.0。
		tstLocs = [None] * coomat.shape[0] # 初始化一个列表 tstLocs，用于存储每个用户的测试物品索引。
		tstUsrs = set()
		for i in range(len(coomat.data)):   # 遍历测试数据矩阵，将每个用户的测试物品索引添加到 tstLocs 中，并记录测试用户的集合 tstUsrs。
			row = coomat.row[i]
			col = coomat.col[i]
			if tstLocs[row] is None:
				tstLocs[row] = list()
			tstLocs[row].append(col)
			tstUsrs.add(row)
		tstUsrs = np.array(list(tstUsrs))   # 将测试用户集合转换为 numpy 数组，并保存为类的属性 tstUsrs 和 tstLocs。
		self.tstUsrs = tstUsrs 
		self.tstLocs = tstLocs

	def __len__(self):
		return len(self.tstUsrs)

	def __getitem__(self, idx):
		return self.tstUsrs[idx], np.reshape(self.csrmat[self.tstUsrs[idx]].toarray(), [-1])





# #在机器学习，尤其是在推荐系统、图神经网络等涉及数据样本处理的领域中，“对每个正样本进行负采样” 是一种常见的数据处理策略，以下为你详细解释：

# ### 1. 正样本和负样本的定义
# - **正样本**：通常指的是符合某种特定关系或类别，在数据集中被认为是 “正例” 的样本。例如在推荐系统中，正样本可以是用户与他们实际交互过（如购买、评分、点击等）的物品之间的关系；在二分类问题中，正样本就是属于目标类别的样本。
# - **负样本**：与正样本相对，是指不符合特定关系或不属于目标类别的样本。在推荐系统里，负样本可以是用户未交互过的物品；在二分类问题中，就是不属于目标类别的样本。

# ### 2. 负采样的目的
# - **增加数据多样性**：在很多实际应用中，正样本的数量往往相对较少，而负样本的数量可能非常庞大。通过对每个正样本进行负采样，可以从大量的负样本中选取一部分作为训练数据，使得训练数据更加多样化，从而提高模型的泛化能力。
# - **平衡正负样本比例**：如果正负样本数量差距过大，模型在训练时可能会过度关注正样本或负样本中的某一类，导致模型的性能不佳。负采样可以调整正负样本的比例，使模型在训练过程中能够更好地学习到正负样本之间的差异，提升模型的准确性和鲁棒性。
# - **加速训练过程**：处理全部的负样本可能会导致计算量过大，训练时间过长。通过负采样，只选取一部分负样本参与训练，能够显著减少计算量，加快模型的训练速度。

# ### 3. 示例说明
# 以推荐系统为例，假设我们有一个用户 - 物品交互数据集，其中记录了用户对某些物品的购买行为。
# - 一个用户购买了物品 A、B、C，那么 (用户, 物品 A)、(用户, 物品 B)、(用户, 物品 C) 就可以看作是正样本。
# - 现在对每个正样本进行负采样，比如对于正样本 (用户, 物品 A)，我们从用户未购买的物品中随机选取一个（假设是物品 D），那么 (用户, 物品 D) 就成为了与 (用户, 物品 A) 对应的负样本。同样地，对于 (用户, 物品 B) 和 (用户, 物品 C) 也进行类似的负采样操作，得到相应的负样本。

# 在代码中，`TrnData` 类的 `negSampling` 方法就是对每个正样本进行负采样的具体实现。它遍历每个正样本（通过行索引 `self.rows` 表示），为每个正样本对应的用户随机选取一个未交互过的物品作为负样本（通过 `np.random.randint` 生成物品索引，并检查该索引对应的物品是否与用户有交互）。 


