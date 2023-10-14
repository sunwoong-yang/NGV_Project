from src.DDO import DDO
from src.Optimizer import Optimizer
from src.Scatter import Scatter
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF

proj_name = "231009_LDW"
excel_name = "lhs600_init_re.xlsx"

ddo = DDO(proj_name)
ddo.read_excel(file_name=excel_name, train_ratio=0.4, coef_outlier=2)

kernel = ConstantKernel() * Matern(length_scale=[1.]*ddo.n_var, nu=2.5, length_scale_bounds=(1e-5, 1e+7))
ddo.fit(kernel=kernel, n_restarts_optimizer=1, random_state=42, normalize_y=True)

plot_  = Scatter(ddo)
plot_.do(save_fig=True)

optimizer = Optimizer(ddo)

opt_results = optimizer.do(pop_size=100)
optimizer.save(opt_results, filename=f"Opt_{proj_name}")

Bayesian_results = optimizer.do(Bayesian=True, pop_size=100)
optimizer.save(Bayesian_results, filename=f"Bayesian_{proj_name}")

x_queries, y_queries = optimizer.get_queries([opt_results, Bayesian_results], num_pts=[30, 30], method="random", filename=f"Queries_{proj_name}")
# x_queries, y_queries = optimizer.get_queries([opt_results], num_pts=[30], method="random", filename=f"Queries_{proj_name}")
