from Source.DDO import DDO
from Source.Optimizer import Optimizer
from Source.Scatter import Scatter
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF

proj_name = "t3_iter1"
# save_index = "t3_iter1"
ddo = DDO(proj_name)
ddo.read_excel(file_name='LHS180_t3_iter1.xlsx')
ddo.shuffle()
kernel = ConstantKernel() * Matern(length_scale=[1.]*ddo.n_var, nu=2.5)
# kernel = ConstantKernel() * RBF(length_scale=[1.]*ddo.n_var)
ddo.fit(kernel=kernel, n_restarts_optimizer=30, random_state=42, normalize_y=True)

plot_  = Scatter(ddo)
plot_.do(save_fig=True)

optimizer = Optimizer(ddo)

opt_results = optimizer.do(pop_size=100)
optimizer.save(opt_results, filename=f"Opt")

Bayesian_results = optimizer.do(Bayesian=True, pop_size=100)
optimizer.save(Bayesian_results, filename=f"Bayesian")

optimizer.get_queries([opt_results, Bayesian_results], num_pts=[30, 30], method="random", filename=f"Queries")
