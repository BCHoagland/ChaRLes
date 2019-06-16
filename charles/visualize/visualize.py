import numpy as np
from visdom import Visdom

viz = Visdom()

class Visualizer:
    def __init__(self):
        self.win = None

    def update_viz(self, ep, ep_reward, env, algo, color, xlabel='episodes'):
        title = env + ' (' + algo + ')'
        if self.win is None:
            self.win = viz.line(
                X=np.array([ep]),
                Y=np.array([ep_reward]),
                win=title,
                opts=dict(
                    title=title,
                    xlabel=xlabel,
                    linecolor=np.array([color])
                )
            )
        else:
            viz.line(
                X=np.array([ep]),
                Y=np.array([ep_reward]),
                win=self.win,
                update='append'
            )
