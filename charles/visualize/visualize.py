import numpy as np
import torch
from visdom import Visdom
import pickle
from pathlib import Path

def get_line(x, y, name, color='transparent', isFilled=False, fillcolor='transparent', width=1, showlegend=False):
        if isFilled:
            fill = 'tonexty'
        else:
            fill = 'none'

        return dict(
            x=x,
            y=y,
            mode='lines',
            type='custom',
            line=dict(
                color=color,
                width=width),
            fill=fill,
            fillcolor=fillcolor,
            name=name,
            showlegend=showlegend
        )

class Visualizer:
    def __init__(self, env_name):
        self.env_name = env_name
        self.visdom = Visdom()

    def get_lines_for_algo(self, data, algo_name):
        color = data['color']
        x = data['x']
        lower_line = get_line(x, data['y']['lower'], 'lower', color='rgba(' + color + ', 0.2)')
        mean_line = get_line(x, data['y']['mean'], algo_name, color='rgb(' + color + ')', width=1.4, showlegend=True)
        upper_line = get_line(x, data['y']['upper'], 'upper', color='rgba(' + color + ', 0.2)', isFilled=True, fillcolor='rgba(' + color + ', 0.1)')
        return [lower_line, upper_line, mean_line]

    def update_saved_data(self, filepath, data_type, algo_name, x, y, color):
        # load and add to the saved data
        with open(filepath, 'rb') as f:
            saved_data = pickle.load(f)

            # create dictionary structure if it doesn't exist yet
            if algo_name not in saved_data:
                saved_data[algo_name] = {'x': [], 'y': {}}
                saved_data[algo_name]['y'] = {'lower': [], 'mean': [], 'upper': []}

            # append new data to saved data
            x = (len(saved_data[algo_name]['x']) + 1) if x is None else x
            saved_data[algo_name]['x'].append(x)

            saved_data[algo_name]['y']['lower'].append((y[0] - y[1]).item())
            saved_data[algo_name]['y']['mean'].append(y[0].item())
            saved_data[algo_name]['y']['upper'].append((y[0] + y[1]).item())

            saved_data[algo_name]['color'] = color

        # save the modified data
        with open(filepath, 'wb') as f:
            pickle.dump(saved_data, f)

        # return the modified data
        return saved_data

    def reset_all_data(self):
        pathlist = Path('.tmp/plot_data/').glob(f'*{self.env_name}*')
        for filepath in pathlist:
            with open(filepath, 'rb') as f:
                saved_data = pickle.load(f)
            types = [type for type in saved_data]
            for type in types:
                saved_data.pop(type, None)
            with open(filepath, 'wb') as f:
                pickle.dump(saved_data, f)

    def reset_data_for_algo(self, algo_name):
        pathlist = Path('.tmp/plot_data/').glob(f'*{self.env_name}*')
        for filepath in pathlist:
            with open(filepath, 'rb') as f:
                saved_data = pickle.load(f)
            saved_data.pop(algo_name, None)
            with open(filepath, 'wb') as f:
                pickle.dump(saved_data, f)

    def plot(self, algo_name, data_type, xlabel, x, y, color=[5, 119, 177]):
        color = f'{color[0]}, {color[1]}, {color[2]}'

        # get the filepath, creating directories as necessary
        path = Path('.tmp/plot_data/')
        path.mkdir(parents=True, exist_ok=True)
        filepath = f'{path}/{data_type + self.env_name}'

        # convert y to the format (y.mu, y.std) if it's not already
        if not isinstance(y, tuple):
            y = (y.mean(), y.std())

        # update and retrieve saved data
        try:
            saved_data = self.update_saved_data(filepath, data_type, algo_name, x, y, color)
        except:
            with open(filepath, 'wb') as f:
                pickle.dump({}, f)
            saved_data = self.update_saved_data(filepath, data_type, algo_name, x, y, color)

        # get lines for each algorithm on the current data
        data = []
        for algo in saved_data:
            data += self.get_lines_for_algo(saved_data[algo], algo)

        # set format for the plot
        title = self.env_name
        layout = dict(
            title=title,
            xaxis={'title': xlabel},
            yaxis={'title': data_type}
        )

        # plot the data
        self.visdom._send({'data': data, 'layout': layout, 'win': title + self.env_name})
