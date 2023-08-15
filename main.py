from Source.DDO import DDO
from Source.Optimizer import Optimizer
from Source.Scatter import Scatter
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF

ddo = DDO()
ddo.read_excel(file_name='LHS180_t1.xlsx')
ddo.shuffle()
kernel = ConstantKernel() * Matern(length_scale=[1.]*ddo.n_var, nu=2.5)
ddo.fit(kernel=kernel, n_restarts_optimizer=1, random_state=42, normalize_y=True)

plot_  = Scatter(ddo)
plot_.do(save_fig=True)

optimizer = Optimizer(ddo)

opt_results = optimizer.do(pop_size=100)
optimizer.save(opt_results, filename="Opt_max")

Bayesian_results = optimizer.do(Bayesian=True, pop_size=100)
optimizer.save(Bayesian_results, filename="Bayesian_max")

optimizer.get_queries([opt_results, Bayesian_results], num_pts=[2, 3], method="random", filename="Queries_max")
