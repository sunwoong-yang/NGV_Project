import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

class Scatter():
    def __init__(self, ddo_cls):
        self.ddo_cls = ddo_cls

    def do(self, x_pred=None, y_true=None, save_fig=True):
        if y_true is None:
            y_true = self.ddo_cls.y_test

        if x_pred is None: # If y_pred is not given, use the prediction from x_test as "y_pred"
            x_pred = self.ddo_cls.x_test
            y_pred = self.ddo_cls.predict(x_pred)
        else:
            y_pred = self.ddo_cls.predict(x_pred)

        self.y_true = y_true
        self.y_pred = y_pred

        files = glob.glob(f"Projects/{self.ddo_cls.proj_name}/figures/*")
        for file in files:
            os.remove(file)

        for y_idx in range(self.ddo_cls.n_obj + self.ddo_cls.n_con):

            Rsq = self.ddo_cls.gpr_models.models[y_idx].score(x_pred, y_true[:,y_idx])
            y_true_ = self.y_true[:,y_idx]
            y_pred_ = self.y_pred[:,y_idx]
            fig, ax = self.make_plot(y_true_, y_pred_, y_idx, Rsq)
            if save_fig:
                if not os.path.exists(f"Projects/{self.ddo_cls.proj_name}/figures"):
                    os.makedirs(f"Projects/{self.ddo_cls.proj_name}/figures")
                plt.close(fig)
                fig.savefig(f"Projects/{self.ddo_cls.proj_name}/figures/{self.ddo_cls.y_list[y_idx]}.png")

    def make_plot(self, y_true, y_pred, y_idx=0, Rsq=None):
        sns.set_style("white")
        sns.set_palette("Set2")
        min_plt, max_plt = np.min([np.min(y_true), np.min(y_pred)]), np.max([np.max(y_true), np.max(y_pred)])
        min_show, max_show = min_plt - 0.1 * (max_plt - min_plt), max_plt + 0.1 * (max_plt - min_plt)
        fig, ax = plt.subplots(dpi=100)
        ax.scatter(y_true, y_pred, edgecolors='k')
        ax.plot([min_show, max_show], [min_show, max_show], '--k')
        ax.set_ylim(min_show, max_show)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel("Ground truth", fontsize=17)
        ax.set_ylabel("Prediction", fontsize=17)
        ax.set_title(self.ddo_cls.y_list[y_idx], fontsize=20)

        if Rsq is not None:
            plt.text(0.6, 0.1, f"$R^2 = {Rsq:.3f}$", transform = ax.transAxes, fontdict={'size':16})

        return fig, ax

