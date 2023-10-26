from src.DDO import DDO
from src.Optimizer import Optimizer
from src.Scatter import Scatter
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF

proj_name = "231009_lhs600_DE"
excel_name = "lhs600_init.xlsx"

ddo = DDO(proj_name)
ddo.read_excel(file_name=excel_name, train_ratio=0.8, coef_outlier=2)

# kernel = ConstantKernel() * Matern(length_scale=[1.]*ddo.n_var, nu=2.5, length_scale_bounds=(1e-5, 1e+7))
DE_kwargs = {"layers": [60,60,60],
             "lr":[1e-2,1e-3],
             "iter":[500,2000]}
# layers = kwargs["layers"]
#             lr = kwargs["lr"]
#             iter = kwargs["iter"]
ddo.fit(model="DE", **DE_kwargs)

plot_  = Scatter(ddo)
plot_.do(save_fig=True)

optimizer = Optimizer(ddo)

opt_results = optimizer.do(pop_size=100)
optimizer.save(opt_results, filename=f"Opt_{proj_name}")

Bayesian_results = optimizer.do(Bayesian=True, pop_size=100)
optimizer.save(Bayesian_results, filename=f"Bayesian_{proj_name}")

x_queries, y_queries = optimizer.get_queries([opt_results, Bayesian_results], num_pts=[30, 40],
                                             method="random", filename=f"Queries_{proj_name}")

