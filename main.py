import dash
from dash import no_update as NUP
import dash_auth
from dash import dash_table, dcc, html
import dash_daq as daq
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from dash_extensions.enrich import DashProxy, MultiplexerTransform, Output, Input, State
from dash_extensions import Keyboard
from dash.dash_table.Format import Format, Scheme, Trim
import plotly.express as px

from game2048.r_learning import *

# Some necessary variables and useful functions
mode_list = {
    'train_agent_button': ('Train Agent', 'Train agent'),
    'watch_agent_button': ('Watch Agent play', 'Watch agent'),
    'agent_stat_button': ('Collect Agent statistics', 'Test agent'),
    'replay_button': ('Replay game', 'Replay game'),
    'play_button': ('Play yourself (desktop only)', 'Play'),
    'open_admin': ('Manage files', 'Admin')
}
act_list = {
    'download': 'Download',
    'upload': 'Upload',
    'delete': 'Delete'
}
params_list = ['name', 'weights_type', 'reward', 'decay_model', 'n', 'alpha', 'decay', 'decay_step', 'low_alpha_limit']
params_dict = {
    'name': {'element': 'input', 'type': 'text', 'value': 'generate random', 'disable': False},
    'weights_type': {'element': 'select', 'value': 'random', 'options': ['random', 'zero'], 'disable': True},
    'reward': {'element': 'select', 'value': 'basic', 'options': ['basic', 'log'], 'disable': True},
    'decay_model': {'element': 'select', 'value': 'simple', 'options': ['simple', 'scaled'], 'disable': True},
    'n': {'element': 'select', 'value': 4, 'options': [2, 3, 4], 'disable': True},
    'alpha': {'element': 'input', 'type': 'number', 'value': 0.25, 'step': 0.01, 'disable': False},
    'decay': {'element': 'input', 'type': 'number', 'value': 0.75, 'step': 0.05, 'disable': False},
    'decay_step': {'element': 'input', 'type': 'number', 'value': 10000, 'step': 5000, 'disable': False},
    'low_alpha_limit': {'element': 'input', 'type': 'number', 'value': 0.01, 'step': 0.0025, 'disable': False}
}
cell_size = CONF['cell_size']
x_position = {i: f'{i * cell_size}px' for i in range(4)}
y_position = {i: f'{i * cell_size + 35}px' for i in range(4)}
numbers = {i: str(1 << i) if i else '' for i in range(16)}
colors = CONF['colors']
colors = {int(v): colors[v] for v in colors}


def display_table(row, score, odo, next_move):
    if next_move == -1:
        header = f'Score = {score}    Moves = {odo}    Game over!'
    else:
        header = f'Score = {score}    Moves = {odo}    Next move = {Game.actions[next_move]}'
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
    return dbc.Alert(f' {text}', color='info' if info else 'success', dismissable=True, duration=5000,
                     className='admin-notification')


def params_line(e):
    data = params_dict[e]
    if data['element'] == 'input':
        if 'step' in data:
            return dbc.InputGroup([
                dbc.InputGroupText(e, className='par-input-text no-border'),
                dbc.Input(id=f'par_{e}', type=data['type'], value=data['value'], step=data['step'],
                          className='par-input-field no-border')], className='no-border')
        else:
            return dbc.InputGroup([
                dbc.InputGroupText(e, className='par-input-text no-border'),
                dbc.Input(id=f'par_{e}', type=data['type'], value=data['value'],
                          className='par-input-field no-border')], className='no-border')
    else:
        return dbc.InputGroup([
            dbc.InputGroupText(e, className='par-input-text no-border'),
            dbc.Select(id=f'par_{e}', options=opt_list(data['options']), value=data['value'],
                       className='par-select-field no-border')], className='no-border')


# App declaration and layout
app = DashProxy(__name__, transforms=[MultiplexerTransform()], title='RL Agent 2048', update_title=None,
                meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1'}])

app.layout = dbc.Container([
    dcc.Download(id="download_file"),
    dcc.Store(id='chain', storage_type='session'),
    dcc.Store(id='current_process', storage_type='session'),
    dcc.Interval(id='update_interval', n_intervals=0, disabled=True),
    dcc.Interval(id='logs_interval', interval=2000, n_intervals=0),
    dbc.Modal([
        dbc.ModalHeader('File Management'),
        dbc.ModalBody([
            dbc.DropdownMenu(id='choose_file_action', label='Action:',
                             children=[dbc.DropdownMenuItem(act_list[v], id=v, n_clicks=0) for v in act_list]),
            dbc.Select(id='choose_file', className='choose-file'),
            dbc.Button('action?', id='admin_go', n_clicks=0, className='admin-go'),
            dcc.Upload('Drag and Drop or Select Files', id='data_uploader', style={'display': 'none'},
                       className='upload-file')
        ], className='admin-page-body'),
        dbc.ModalFooter([
            html.Div(id='admin_notification'),
            dbc.Button('CLOSE', id='close_admin', n_clicks=0)
        ])
    ], id='admin_page', size='lg', centered=True, contentClassName='admin-page'),
    dbc.Modal([
        dbc.ModalHeader('Enter/adjust/confirm parameters for an Agent'),
        dbc.ModalBody([params_line(e) for e in params_list], className='params-page-body'),
        dbc.ModalFooter([
            dbc.Button('TRAIN', id='start_training', n_clicks=0, className='start-training'),
            html.Div(id='params_notification'),
            dbc.Button('CLOSE', id='close_params', n_clicks=0)
        ])
    ], id='params_page', size='lg', centered=True, contentClassName='params-page'),
    dbc.Row(html.H3('Reinforcement Learning 2048 Agent', className='header card-header')),
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.DropdownMenu(id='choose_option', label='Mode:', className='mode-choose',
                             children=[dbc.DropdownMenuItem(mode_list[v][0], id=v, n_clicks=0) for v in mode_list]),
            html.H6(id='mode_text', className='mode-text'),
            dbc.InputGroup([
                dbc.InputGroupText('Game:', className='input-text'),
                dbc.Select(id='choose_for_replay', className='input-field')
                ], id='input_group_game', style={'display': 'none'}, className='my-input-group',
            ),
            dbc.InputGroup([
                dbc.InputGroupText('Agent:', className='input-text'),
                dbc.Select(id='choose_stored_agent', className='input-field'),
                html.Div([
                    dbc.InputGroupText('depth:', className='lf-cell lf-text lf-depth'),
                    dbc.Input(id='choose_depth', type='number', min=0, max=4, value=0,
                              className='lf-cell lf-field lf-depth'),
                    dbc.InputGroupText('width:', className='lf-cell lf-text lf-width'),
                    dbc.Input(id='choose_width', type='number', min=1, max=4, value=1,
                              className='lf-cell lf-field lf-width'),
                    dbc.InputGroupText('since_empty:', className='lf-cell lf-text lf-empty'),
                    dbc.Input(id='choose_since_empty', type='number', min=0, max=8, value=6,
                              className='lf-cell lf-field lf-empty'),
                ], className='lf-params'),
                dbc.Button('LAUNCH!', id='replay_agent_button', disabled=True, className='launch-game')
                ], id='input_group_agent', style={'display': 'none'}, className='my-input-group',
            ),
            dbc.InputGroup([
                dbc.InputGroupText('Agent:', className='input-text'),
                dbc.Select(id='choose_train_agent', className='input-field'),
                dbc.InputGroupText('Config:', className='input-text second-line'),
                dbc.Select(id='choose_config', disabled=True, className='input-field second-line'),
                dbc.Button('Confirm parameters', id='go_to_params', disabled=True, className='go-to-params'),
                ], id='input_group_train', style={'display': 'none'}, className='my-input-group',
            ),
            html.Div([
                    html.H6('Logs', className='logs-header card-header'),
                    dbc.Button('Stop', id='stop_agent', n_clicks=0, className='logs-button stop-agent',
                               style={'display': 'none'}),
                    dbc.Button('Download', id='download_logs', n_clicks=0, className='logs-button logs-download'),
                    dbc.Button('Clear', id='clear_logs', n_clicks=0, className='logs-button logs-clear'),
                    html.Div(id='logs_display', className='logs-display')
                ], className='logs-window'),
            ], className='log-box'),

        ),
        dbc.Col([
            dbc.Card([
                dbc.CardBody(id='game_card'),
                daq.Gauge(id='gauge', className='gauge',
                          color={"gradient": True, "ranges": {"green": [0, 6], "yellow": [6, 8], "red": [8, 10]}}),
                html.H6('Speed', className='speed-header'),
                dcc.Slider(id='gauge-slider', min=0, max=10, value=6, marks={v: str(v) for v in range(11)},
                           step=0.1, className='slider'),
                html.Div([
                    dbc.Button('PAUSE', id='pause_game', n_clicks=0, className='one-button pause-button'),
                    dbc.Button('RESUME', id='resume_game', n_clicks=0, className='one-button resume-button'),
                ], className='button-line'),
            ], className='game-box align-items-center'),
        ])
    ])
])


# admin page callbacks
@app.callback(
    Output('admin_page', 'is_open'),
    Input('open_admin', 'n_clicks'), Input('close_admin', 'n_clicks'),
    State('admin_page', 'is_open'),
)
def toggle_admin_page(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@app.callback(
    Output('choose_file', 'options'), Output('choose_file', 'value'), Output('admin_go', 'children'),
    [Input(v, 'n_clicks') for v in act_list]
)
def act_process(*args):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    idx = ctx.triggered[0]['prop_id'].split('.')[0]
    files = ['config file', 'game', 'agent'] if idx == 'upload' else list_names_s3()
    value = 'config file' if idx == 'upload' else None
    return [{'label': v, 'value': v} for v in files], value, act_list[idx]


@app.callback(
    Output('admin_notification', 'children'), Output('download_file', 'data'),
    Input('admin_go', 'n_clicks'),
    State('admin_go', 'children'), State('choose_file', 'value')
)
def admin_act(n, act, name):
    if n:
        if act == 'Delete':
            if name:
                delete_s3(name)
                return my_alert(f'{name} deleted'), NUP
            else:
                return my_alert(f'Choose file to delete!', info=True), NUP
        elif act == 'Download':
            if name:
                temp = name[2:]
                s3_bucket.download_file(name, temp)
                to_send = dcc.send_file(temp)
                os.remove(temp)
                return NUP, to_send
            else:
                return my_alert(f'Choose file for download!', info=True), NUP
        raise PreventUpdate
    else:
        raise PreventUpdate


@app.callback(
    Output('data_uploader', 'style'),
    Input('admin_go', 'children')
)
def show_upload(act):
    return {'display': 'block' if act == 'Upload' else 'none'}


@app.callback(
    Output('admin_notification', 'children'),
    Input('data_uploader', 'contents'), Input('data_uploader', 'filename'),
    State('choose_file', 'value')
)
def upload_process(content, name, kind):
    if name:
        file_data = content.encode("utf8").split(b";base64,")[1]
        with open(name, "wb") as f:
            f.write(base64.decodebytes(file_data))
        prefix = 'c/' if kind == 'config file' else ('g/' if kind == 'game' else '/a')
        s3_bucket.upload_file(name, prefix + name)
        os.remove(name)
        return my_alert(f'Uploaded {name} as new {kind}')
    else:
        raise PreventUpdate


# Control Panel callbacks
@app.callback(
    Output('mode_text', 'children'), Output('input_group_agent', 'style'), Output('input_group_game', 'style'),
    Output('input_group_train', 'style'),
    [Input(v, 'n_clicks') for v in mode_list]
)
def mode_process(*args):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    idx = ctx.triggered[0]['prop_id'].split('.')[0]
    to_see_games = 'block' if idx == 'replay_button' else 'none'
    to_see_agents = 'block' if idx in ('watch_agent_button', 'agent_stat_button') else 'none'
    to_train_agent = 'block' if idx == 'train_agent_button' else 'none'
    return mode_list[idx][1], {'display': to_see_agents}, {'display': to_see_games}, {'display': to_train_agent}


# Game Replay callbacks
@app.callback(
    Output('choose_for_replay', 'options'),
    Input('input_group_game', 'style')
)
def find_games(style):
    if style['display'] == 'none':
        raise PreventUpdate
    games = [v for v in list_names_s3() if v[:2] == 'g/']
    return [{'label': v[2:-4], 'value': v} for v in games]


@app.callback(
    Output('chain', 'data'), Output('update_interval', 'disabled'),
    Input('choose_for_replay', 'value'),
    State('chain', 'data')
)
def replay_game(name, previous):
    if name:
        if previous:
            del globals()[previous]
        chain = f'game{random.randrange(100000)}'
        game = load_s3(name)
        globals()[chain] = {
            'games': game.replay(verbose=False),
            'step': 0
        }
        return chain, False
    else:
        raise PreventUpdate


@app.callback(
    Output('game_card', 'children'), Output('update_interval', 'disabled'),
    Input('update_interval', 'n_intervals'),
    State('chain', 'data')
)
def refresh_board(n, chain):
    if n and chain:
        point = globals().get(chain, None)
        if not point:
            raise PreventUpdate
        step = point['step']
        if step == -1:
            return NUP, True
        row, score, next_move = point['games'][step]
        to_show = display_table(row, score, step, next_move)
        point['step'] = -1 if next_move == -1 else point['step'] + 1
        return to_show, NUP
    else:
        raise PreventUpdate


# Agent Replay and Test callbacks
@app.callback(
    Output('choose_stored_agent', 'options'),
    Input('input_group_agent', 'style')
)
def find_agents(style):
    if style['display'] == 'none':
        raise PreventUpdate
    agents = [v for v in list_names_s3() if v[:2] == 'a/']
    return [{'label': v[2:-4], 'value': v} for v in agents]


@app.callback(
    Output('replay_agent_button', 'disabled'),
    Input('choose_stored_agent', 'value')
)
def enable_agent_play_button(name):
    if name:
        return False
    else:
        raise PreventUpdate


@app.callback(
    Output('chain', 'data'), Output('update_interval', 'disabled'),
    Input('replay_agent_button', 'n_clicks'),
    State('chain', 'data'), State('choose_stored_agent', 'value'),
    State('choose_depth', 'value'), State('choose_width', 'value'), State('choose_since_empty', 'value')
)
def start_agent_play(n, previous, agent_file, depth, width, empty):
    if n:
        if previous:
            del globals()[previous]
        agent = load_s3(agent_file)
        evaluator = agent.evaluate
        chain = f'game{random.randrange(100000)}'
        return chain, False
    else:
        raise PreventUpdate


# Agent Train callbacks
@app.callback(
    Output('choose_train_agent', 'options'), Output('choose_config', 'options'),
    Input('input_group_train', 'style')
)
def find_agents(style):
    if style['display'] == 'none':
        raise PreventUpdate
    agents = [v for v in list_names_s3() if v[:2] == 'a/']
    configs = [v for v in list_names_s3() if v[:2] == 'c/']
    agent_options = [{'label': v[2:-4], 'value': v} for v in agents] + [{'label': 'New agent', 'value': 'New agent'}]
    conf_options = [{'label': v[2:-5], 'value': v} for v in configs] + [{'label': 'New config', 'value': 'New config'}]
    return agent_options, conf_options


@app.callback(
    Output('choose_config', 'value'), Output('choose_config', 'disabled'),
    Input('choose_train_agent', 'value')
)
def open_train_params(agent):
    if agent == 'New agent':
        return None, False
    elif agent:
        return NUP, True
    else:
        raise PreventUpdate


@app.callback(
    Output('go_to_params', 'disabled'),
    Input('choose_train_agent', 'value'),
    Input('choose_config', 'value')
)
def open_train_params(agent, config):
    return not agent or (agent == 'New agent' and not config)


@app.callback(
    Output('params_page', 'is_open'),
    Input('go_to_params', 'n_clicks'), Input('close_params', 'n_clicks'),
    State('params_page', 'is_open'),
)
def toggle_params_page(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@app.callback(
    [Output(f'par_{e}', 'disabled') for e in params_list] + [Output(f'par_{e}', 'value') for e in params_list],
    Input('params_page', 'is_open'),
    State('choose_train_agent', 'value'), State('choose_config', 'value'),
)
def fill_params(is_open, agent_name, config_name):
    if is_open:
        if agent_name != 'New agent':
            agent = load_s3(agent_name)
            dis = [params_dict[e]['disable'] for e in params_list]
            ui_params = [getattr(agent, e) for e in params_list]
        elif config_name != 'New config':
            config = load_s3(config_name)
            dis = [False for e in params_list]
            ui_params = [config.get(e, params_dict[e]['value']) for e in params_list]
        else:
            dis = [False for e in params_list]
            ui_params = [params_dict[e]['value'] for e in params_list]
        return dis + ui_params
    else:
        raise PreventUpdate


@app.callback(
    Output('params_notification', 'children'), Output('stop_agent', 'style'), Output('current_process', 'data'),
    Input('start_training', 'n_clicks'),
    [State(f'par_{e}', 'value') for e in params_list] +
    [State('choose_train_agent', 'value'), State('current_process', 'data')]
)
def start_training(*args):
    if args[0]:
        message = NUP
        new_name, new_agent_file, current_process = args[1], args[-2], args[-1]
        ui_params = {e: args[i + 2] for i, e in enumerate(params_list[1:])}
        if current_process:
            globals()[current_process].terminate()
            globals()[current_process].join()
        if new_name == 'generate random':
            new_name = f'agent_{random.randrange(100000)}'
        else:
            new_name = ''.join(x for x in new_name if x.isalnum())
        if new_agent_file == 'New agent':
            new_config_file = f'c/config_{new_name}.json'
            save_s3(ui_params, new_config_file)
            message = my_alert(f'new config file {new_config_file[2:]} saved')
            current = Q_agent(name=new_name, config_file=new_config_file, storage='s3', console='web')
        else:
            current = load_s3(new_agent_file)
            if current.name != new_name:
                current.name = new_name
                current.file = current.name + '.pkl'
                current.game_file = 'best_of_' + current.file
            for e in ui_params:
                setattr(current, e, ui_params[e])
        current.logs = ''
        current.save_agent()
        proc = f'p_{random.randrange(100000)}'
        LOGS.clear()
        globals()[proc] = Process(target=current.train_run, daemon=True)
        globals()[proc].start()
        return message, {'display': 'block'}, proc
    else:
        raise PreventUpdate


@app.callback(
    Output('current_process', 'data'), Output('stop_agent', 'style'),
    Input('stop_agent', 'n_clicks'),
    State('current_process', 'data'),
)
def stop_agent(n, current_process):
    if n and current_process:
        globals()[current_process].terminate()
        globals()[current_process].join()
        LOGS.add('Training terminated')
        return None, {'display': 'none'}
    else:
        raise PreventUpdate


# Log window callbacks
@app.callback(
    Output('logs_display', 'children'),
    Input('logs_interval', 'n_intervals')
)
def update_logs(n):
    if n:
        return LOGS.get()
    else:
        raise PreventUpdate


# Game Board callbacks
@app.callback(
    Output('gauge', 'value'), Output('update_interval', 'interval'),
    Input('gauge-slider', 'value')
)
def update_output(value):
    return value, value * 200 + 50


@app.callback(
    Output('update_interval', 'disabled'),
    Input('pause_game', 'n_clicks')
)
def pause_game(n):
    return bool(n)


@app.callback(
    Output('update_interval', 'disabled'),
    Input('resume_game', 'n_clicks')
)
def resume_game(n):
    return not bool(n)


if __name__ == '__main__':

    app.run_server(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)), debug=(LOCAL == 'local'), use_reloader=False)
