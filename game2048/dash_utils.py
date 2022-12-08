import dash
from dash import no_update as NUP
from dash import dcc, html
from dash.dependencies import ClientsideFunction
import dash_daq as daq
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from dash_extensions.enrich import DashProxy, MultiplexerTransform, Output, Input, State
from dash_extensions import EventListener
import plotly.express as px

from .r_learning import *

# Some necessary variables and useful functions
mode_list = {
    'description_button': ('HELP!', 'Guide'),
    'train_agent_button': ('Train Agent', 'Train Agent'),
    'agent_stat_button': ('Collect Agent Statistics', 'Test Agent'),
    'watch_agent_button': ('Watch Agent Play', 'Watch Agent'),
    'replay_button': ('Replay Game', 'Replay Game'),
    'play_button': ('Play Yourself', 'Play'),
    'open_admin': ('Manage Files', 'Admin')
}
act_list = {
    'download': 'Download',
    'upload': 'Upload',
    'delete': 'Delete'
}
params_list = ['name', 'n', 'alpha', 'decay', 'decay_step', 'low_alpha_limit', 'Training episodes']
params_dict = {
    'name': {'element': 'input', 'type': 'text', 'value': 'test_agent', 'disable': False},
    'n': {'element': 'select', 'value': 4, 'options': [2, 3, 4, 5, 6], 'disable': True},
    'alpha': {'element': 'input', 'type': 'number', 'value': 0.25, 'step': 0.0001, 'disable': False},
    'decay': {'element': 'input', 'type': 'number', 'value': 0.75, 'step': 0.01, 'disable': False},
    'decay_step': {'element': 'input', 'type': 'number', 'value': 10000, 'step': 1000, 'disable': False},
    'low_alpha_limit': {'element': 'input', 'type': 'number', 'value': 0.01, 'step': 0.0001, 'disable': False},
    'Training episodes': {'element': 'input', 'type': 'number', 'value': 100000, 'step': 1000, 'disable': False},
}
keyboard_dict = {
    'Left': 0,
    'Up': 1,
    'Right': 2,
    'Down': 3
}
cell_size = CONF['cell_size']
x_position = {i: f'{i * cell_size}px' for i in range(4)}
y_position = {i: f'{i * cell_size + 35}px' for i in range(4)}
numbers = {i: str(1 << i) if i else '' for i in range(16)}
colors = CONF['colors']
colors = {int(v): colors[v] for v in colors}


def display_table(row, score, odo, next_move, self_play=False):
    header = f'Score = {score}    Moves = {odo}    '
    if next_move == -1:
        header += 'Game over!'
    elif not self_play:
        header += f'Next move = {Game.actions[next_move]}'
    return dbc.Card([
        html.H6(header, className='game-header'),
        dbc.CardBody([html.Div(numbers[row[j, i]], className='cell',
                               style={'left': x_position[i], 'top': y_position[j], 'background': colors[row[j, i]]})
                      for j in range(4) for i in range(4)])
        ], style={'width': '400px'}
    )


def opt_list(l):
    return [{'label': v, 'value': v} for v in l]


def my_alert(text, info=False):
    return dbc.Alert(f' {text} ', color='info' if info else 'success', dismissable=True, duration=5000,
                     className='admin-notification')


def dash_send(name):
    temp, _ = temp_local_name(name)
    s3_bucket.download_file(name, temp)
    to_send = dcc.send_file(temp)
    os.remove(temp)
    return to_send


def markdown_text(md_file):
    with open(md_file, 'r') as f:
        return f.read()


def while_loading(idx, top):
    return dcc.Loading(id=idx, type='cube', color='#77b300', className='loader', style={'top': f'{top}px'})


def params_line(e):
    data = params_dict[e]
    if data['element'] == 'input':
        if 'step' in data:
            return dbc.InputGroup([
                dbc.InputGroupText(e, className='par-input-text no-border'),
                dbc.Input(id=f'par_{e}', type=data['type'], step=data['step'],
                          className='par-input-field no-border')], className='no-border')
        else:
            return dbc.InputGroup([
                dbc.InputGroupText(e, className='par-input-text no-border'),
                dbc.Input(id=f'par_{e}', type=data['type'],
                          className='par-input-field no-border')], className='no-border')
    else:
        return dbc.InputGroup([
            dbc.InputGroupText(e, className='par-input-text no-border'),
            dbc.Select(id=f'par_{e}', options=opt_list(data['options']),
                       className='par-select-field no-border')], className='no-border')
