import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

class Scatter():
    def __init__(self, parent_cls):
        self.parent_cls = parent_cls

    def do(self, y_true=None, y_pred=None, save_fig=True):
        if y_true is None:
            y_true = self.parent_cls.y_test
        if y_pred is None: # If y_pred is not given, use the prediction from x_test as "y_pred"
            y_pred = self.parent_cls.predict()
        self.y_true = y_true
        self.y_pred = y_pred
        # def make_plot(y_true, y_pred):
        #     sns.set_style("whitegrid")
        #     sns.set_palette("Set2")
        #     min_plt, max_plt = np.min([np.min(y_true), np.min(y_pred)]), np.max([np.max(y_true), np.max(y_pred)])
        #     min_show, max_show = min_plt - 0.1 * (max_plt - min_plt), max_plt + 0.1 * (max_plt - min_plt)
        #     fig, ax = plt.subplots(dpi=100)
        #     ax.scatter(y_true, y_pred, edgecolors='k')
        #     ax.plot([min_show, max_show], [min_show, max_show], '--k')
        #     ax.set_ylim(min_show, max_show)
        #     ax.set_aspect('equal', adjustable='box')
        #     ax.set_xlabel("Ground truth", fontsize=17)
        #     ax.set_ylabel("Prediction", fontsize=17)
        #     ax.set_title(self.parent_cls.y_list[y_idx], fontsize=20)
        #
        #     return fig, ax

        for y_idx in range(self.parent_cls.n_obj):

            y_true_ = self.y_true[:,y_idx]
            y_pred_ = self.y_pred[:,y_idx]
            fig, ax = self.make_plot(y_true_, y_pred_, y_idx)
            if save_fig:
                if not os.path.exists("figures"):
                    os.makedirs("figures")
                fig.savefig(f"figures/{self.parent_cls.y_list[y_idx]}")
    def make_plot(self, y_true, y_pred, y_idx=0):
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
        ax.set_title(self.parent_cls.y_list[y_idx], fontsize=20)
        # sns.despine(bottom=False, left=False)
        # sns.set_context(rc={'patch.linewidth': 0.0})

        return fig, ax

