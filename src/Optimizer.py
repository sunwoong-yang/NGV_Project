import pandas as pd
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from scipy.stats import norm
import os

class Optimizer():

    def __init__(self, ddo_cls):
        self.ddo_cls = ddo_cls

    def do(self, Bayesian=False, pop_size=100):
        # self.Bayesian = Bayesian
        problem = self.def_problem(Bayesian)
        algorithm, termination = self.def_algorithm(pop_size)

        res = minimize(problem,
               algorithm,
               termination,
               seed=42,
               save_history=True,
               verbose=False)

        if not Bayesian:
            res.F = res.F * self.ddo_cls.QoI_direction
        else: # If Bayesian, convert the sign of res.F since EI was maximized
            res.F *= -1.
        res.F = res.F[res.F[:, 0].argsort()]  # Rearrage res by first QoI
        res.X = res.X[res.F[:, 0].argsort()]  # Rearrage res by first QoI

        return res


    def def_problem(self, Bayesian=False):

        ddo_cls = self.ddo_cls


        class MyProblem(ElementwiseProblem):

            def __init__(self):
                super().__init__(n_var=ddo_cls.n_var,
                                 n_obj=ddo_cls.n_obj,
                                 xl=np.array([0] * ddo_cls.n_var),
                                 xu=np.array([1] * ddo_cls.n_var))

            def _evaluate(self, x, out, *args, **kwargs):
                if not Bayesian:
                    f = np.hsplit(ddo_cls.predict(x.reshape(1,-1)), indices_or_sections = ddo_cls.n_obj)
                    f = np.array(f).reshape(1,-1) * ddo_cls.QoI_direction
                if Bayesian:
                    f = self.cal_EI(x.reshape(1,-1)) * -1. # Since EI should be maximized
                out["F"] = [f]

            def cal_EI(self, x, xi=0.0):
                mu, std = ddo_cls.predict(x, return_std=True)
                mu_train = ddo_cls.predict(ddo_cls.x_train)
                std = std.reshape(-1, 1)

                ei = np.zeros((4))
                for y_idx in range(mu.shape[1]):
                    if ddo_cls.QoI_direction[y_idx] == 1.:  # minimization case
                        mu_train_opt = np.min(mu_train[y_idx])
                        imp = mu_train_opt - mu[:,y_idx] - xi

                    elif ddo_cls.QoI_direction[y_idx] == -1.:  # maximization case
                        mu_train_opt = np.max(mu_train[y_idx])
                        imp = mu[:,y_idx] - mu_train_opt - xi

                    Z = imp / std[y_idx]
                    ei[y_idx] = imp * norm.cdf(Z) + std[y_idx] * norm.pdf(Z)
                ei[ei <= 0.0] = 0.0

                return ei

        problem = MyProblem()

        return problem

    def def_algorithm(self,pop_size):

        algorithm = NSGA2(
            pop_size=pop_size,
            eliminate_duplicates=True
        )
        termination = get_termination("n_gen", 100)

        return algorithm, termination

    def get_queries(self, results, num_pts, method="random", filename=None):
        if len(results) != len(num_pts):
            raise Exception("Different size between results and num_pts")

        x_queries_temp, y_queries_temp = [], []
        for results_, num_pts_ in zip(results, num_pts):

            x_optimized = results_.X
            y_optimized = results_.F
            if method == "random":
                pareto_idx = np.random.choice(range(0, x_optimized.shape[0]), size=num_pts_, replace=False)
            elif method == "even":
                pareto_idx = np.linspace(0, x_optimized.shape[0], num_pts).astype(int)
            else:
                raise Exception("Invalid method")

            x_queries_ = results_.X[pareto_idx]
            y_queries_ = results_.F[pareto_idx]
            x_queries_temp.append(x_queries_)
            y_queries_temp.append(y_queries_)

        x_queries, y_queries = [], []
        for x_, y_ in zip(x_queries_temp, y_queries_temp):
            for x__, y__ in zip(x_, y_):
                x_queries.append(x__)
                y_queries.append(y__)

        x_queries = np.array(x_queries)
        # y_queries = np.array(y_queries)
        y_queries = self.ddo_cls.predict(x_queries)
        if filename is not None:
            self.save_queries(x_queries, y_queries, filename=filename)

        return x_queries, y_queries


    def save(self, res, filename=""):
        x_optimized = res.X
        y_optimized = res.F
        write_data = pd.DataFrame(np.concatenate([x_optimized, y_optimized],axis=1))
        write_data.columns = self.ddo_cls.x_list + self.ddo_cls.y_list
        self.write_data = write_data
        if not os.path.exists(f"Projects/{self.ddo_cls.proj_name}/results"):
            os.makedirs(f"Projects/{self.ddo_cls.proj_name}/results")
        write_data.to_excel(f"Projects/{self.ddo_cls.proj_name}/results/{filename}.xlsx", index=False)

    def save_queries(self, x, y, filename=""):
        write_data = pd.DataFrame(np.concatenate([x, y],axis=1))
        write_data.columns = self.ddo_cls.x_list + self.ddo_cls.y_list
        self.write_data = write_data
        if not os.path.exists(f"Projects/{self.ddo_cls.proj_name}/results"):
            os.makedirs(f"Projects/{self.ddo_cls.proj_name}/results")
        write_data.to_excel(f"Projects/{self.ddo_cls.proj_name}/results/{filename}.xlsx", index=False)

        return write_data