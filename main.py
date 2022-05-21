import dash
from dash import no_update as NUP
from dash import dcc, html
from dash.dependencies import ClientsideFunction
import dash_daq as daq
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from dash_extensions.enrich import DashProxy, MultiplexerTransform, Output, Input, State
from dash_extensions import Keyboard
import plotly.express as px

from game2048.r_learning import *
from game2048 import game_logic

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
params_list = ['name', 'reward', 'decay_model', 'n', 'alpha', 'decay', 'decay_step',
               'low_alpha_limit', 'Training episodes']
params_dict = {
    'name': {'element': 'input', 'type': 'text', 'value': 'test_agent', 'disable': False},
    'reward': {'element': 'select', 'value': 'basic', 'options': ['basic', 'log'], 'disable': True},
    'decay_model': {'element': 'select', 'value': 'simple', 'options': ['simple', 'scaled'], 'disable': True},
    'n': {'element': 'select', 'value': 4, 'options': [2, 3, 4, 5], 'disable': True},
    'alpha': {'element': 'input', 'type': 'number', 'value': 0.25, 'step': 0.01, 'disable': False},
    'decay': {'element': 'input', 'type': 'number', 'value': 0.75, 'step': 0.01, 'disable': False},
    'decay_step': {'element': 'input', 'type': 'number', 'value': 10000, 'step': 1000, 'disable': False},
    'low_alpha_limit': {'element': 'input', 'type': 'number', 'value': 0.01, 'step': 0.0025, 'disable': False},
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


def while_loading(id, top):
    return dcc.Loading(html.Div(id=id), type='cube', color='#77b300', className='loader', style={'top': f'{top}px'})


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


def kill_process(proc_name):
    if proc_name and proc_name in globals():
        proc = globals()[proc_name]
        proc.terminate()
        proc.join()
        del proc


def kill_chain(chain_name):
    if chain_name and chain_name in globals():
        del globals()[chain_name]
        if chain_name in game_logic.__dict__:
            del game_logic.__dict__[chain_name]


# App declaration and layout
app = DashProxy(__name__, transforms=[MultiplexerTransform()], title='RL Agent 2048', update_title=None,
                meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1'}])

app.layout = dbc.Container([
    dcc.Download(id='download_file'),
    Keyboard(id='keyboard'),
    dcc.Store(id='chain', storage_type='session'),
    dcc.Store(id='current_process', storage_type='session'),
    dcc.Store(id='agent_for_chart', storage_type='session'),
    dcc.Interval(id='update_interval', n_intervals=0, disabled=True),
    dcc.Interval(id='logs_interval', interval=1000, n_intervals=0),
    dbc.Modal([
        while_loading('uploading', 25),
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
        while_loading('fill_loading', 125),
        while_loading('loading', 125),
        dbc.ModalHeader('Enter/adjust/confirm parameters for an Agent'),
        dbc.ModalBody([params_line(e) for e in params_list], className='params-page-body'),
        dbc.ModalFooter([
            dbc.Button('TRAIN', id='start_training', n_clicks=0, className='start-training'),
            html.Div(id='params_notification'),
            dbc.Button('CLOSE', id='close_params', n_clicks=0)
        ])
    ], id='params_page', size='lg', centered=True, contentClassName='params-page'),
    dbc.Modal([
        while_loading('chart_loading', 0),
        dbc.ModalHeader(id='chart_header'),
        dbc.ModalBody(id='chart'),
        dbc.ModalFooter(dbc.Button('CLOSE', id='close_chart', n_clicks=0))
    ], id='chart_page', size='xl', centered=True, contentClassName='chart-page'),
    dbc.Row(html.H4(['Reinforcement Learning 2048 Agent ',
                     dcc.Link('\u00A9abachurin', href='https://www.abachurin.com', target='blank')],
                    className='card-header my-header')),
    dbc.Row([
        dbc.Col(dbc.Card([
            html.H6('Choose:', id='mode_text', className='mode-text'),
            dbc.DropdownMenu(id='choose_option', label='MODE ?', color='success', className='mode-choose',
                             children=[dbc.DropdownMenuItem(mode_list[v][0], id=v, n_clicks=0) for v in mode_list]),
            dbc.Button(id='chart_button', className='chart-button', style={'display': 'none'}),
            dbc.InputGroup([
                dbc.InputGroupText('Game:', className='input-text'),
                dbc.Select(id='choose_for_replay', className='input-field'),
                dbc.Button('REPLAY', id='replay_game_button', disabled=True, className='replay-game')
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
                    dbc.InputGroupText('empty:', className='lf-cell lf-text lf-empty'),
                    dbc.Input(id='choose_since_empty', type='number', min=0, max=8, value=6,
                              className='lf-cell lf-field lf-empty'),
                ], className='lf-params'),
                dbc.Button('LAUNCH!', id='replay_agent_button', disabled=True, className='launch-game'),
                html.Div([
                    dbc.InputGroupText('N:', className='num-eps-text'),
                    dbc.Input(id='choose_num_eps', type='number', min=10, value=100, step=10, className='num-eps-field')
                    ], id='num_eps', style={'display': 'none'}, className='num-eps')
                ], id='input_group_agent', style={'display': 'none'}, className='my-input-group',
            ),
            dbc.InputGroup([
                dbc.InputGroupText('Agent:', className='input-text'),
                dbc.Select(id='choose_train_agent', className='input-field'),
                dbc.InputGroupText('Config:', className='input-text config-text'),
                dbc.Select(id='choose_config', disabled=True, className='input-field config-input'),
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
                dbc.Toast('Use buttons below or keyboard!\n'
                          'When two equal tiles collide, they combine to give you one '
                          'greater tile that displays their sum. The more you do this, obviously, the higher the '
                          'tiles get and the more crowded the board becomes. Your objective is to reach highest '
                          'possible score before the board fills up', header='Game instructions  ',
                          headerClassName='inst-header', id='play_instructions', dismissable=True, is_open=False),
                dbc.CardBody(id='game_card'),
                html.Div([
                    daq.Gauge(id='gauge', className='gauge',
                              color={"gradient": True, "ranges": {"blue": [0, 6], "yellow": [6, 8], "red": [8, 10]}}),
                    html.H6('DELAY', className='speed-header'),
                    dcc.Slider(id='gauge-slider', min=0, max=10, value=3, marks={v: str(v) for v in range(11)},
                               step=0.1, className='slider'),
                    html.Div([
                        dbc.Button('PAUSE', id='pause_game', n_clicks=0, className='one-button pause-button'),
                        dbc.Button('RESUME', id='resume_game', n_clicks=0, className='one-button resume-button'),
                    ], className='button-line')
                ], id='gauge_group', className='gauge-group'),
                html.Div([
                    dbc.Button('\u2190', id='move_0', className='move-button move-left'),
                    dbc.Button('\u2191', id='move_1', className='move-button move-up'),
                    dbc.Button('\u2192', id='move_2', className='move-button move-right'),
                    dbc.Button('\u2193', id='move_3', className='move-button move-down'),
                    dbc.Button('RESTART', id='restart_play', className='restart-play'),
                ], id='play-yourself-group', className='gauge-group', style={'display': 'none'}),
            ], className='log-box align-items-center'),
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
    files = ['config file', 'game', 'agent'] if idx == 'upload' else [v for v in list_names_s3() if v != 'logs.txt']
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
                temp, ext = temp_name(name)
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
    Output('admin_notification', 'children'), Output('uploading', 'className'),
    Input('data_uploader', 'filename'), State('data_uploader', 'contents'),
    State('choose_file', 'value')
)
def upload_process(name, content, kind):
    if name:
        file_data = content.encode("utf8").split(b";base64,")[1]
        with open(name, "wb") as f:
            f.write(base64.decodebytes(file_data))
        prefix = 'c/' if kind == 'config file' else ('g/' if kind == 'game' else 'a/')
        s3_bucket.upload_file(name, prefix + name)
        os.remove(name)
        return my_alert(f'Uploaded {name} as new {kind}'), NUP
    else:
        raise PreventUpdate


# Control Panel callbacks
@app.callback(
    Output('mode_text', 'children'), Output('input_group_agent', 'style'), Output('input_group_game', 'style'),
    Output('input_group_train', 'style'), Output('num_eps', 'style'),
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
    to_test_agent = 'block' if idx == 'agent_stat_button' else 'none'
    return mode_list[idx][1], {'display': to_see_agents}, {'display': to_see_games}, \
        {'display': to_train_agent}, {'display': to_test_agent}


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
    Output('replay_game_button', 'disabled'),
    Input('choose_for_replay', 'value')
)
def enable_replay_game_button(name):
    return not bool(name)


@app.callback(
    Output('chain', 'data'), Output('update_interval', 'disabled'), Output('choose_for_replay', 'value'),
    Input('replay_game_button', 'n_clicks'),
    State('choose_for_replay', 'value'), State('chain', 'data')
)
def replay_game(name, game_file, previous_chain):
    if name:
        kill_chain(previous_chain)
        chain = f'g{random.randrange(100000)}'
        game = load_s3(game_file)
        globals()[chain] = {
            'type': 'game',
            'games': game.replay(verbose=False),
            'step': 0
        }
        return chain, False, None
    else:
        raise PreventUpdate


# Board Refresh for "Game Replay" and "Agent Play" functions
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

        # Game Replay
        if point['type'] == 'game':
            step = point['step']
            if step == -1:
                return NUP, True
            row, score, next_move = point['games'][step]
            to_show = display_table(row, score, step, next_move)
            point['step'] = -1 if next_move == -1 else point['step'] + 1
            return to_show, NUP

        # Agent Play
        elif point['type'] == 'agent':
            step, game = point['step'], point['game']
            if point['step'] >= game.odometer and not game.game_over(game.row):
                return NUP, NUP
            if step == -1:
                return NUP, True
            row, score, next_move = game.history[step]
            to_show = display_table(row, score, step, next_move)
            point['step'] = -1 if next_move == -1 else point['step'] + 1
            return to_show, NUP

        # Play Yourself
        else:
            raise PreventUpdate
    else:
        raise PreventUpdate


# Agent Play callbacks
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
    State('mode_text', 'children'), State('chain', 'data'), State('choose_stored_agent', 'value'),
    State('choose_depth', 'value'), State('choose_width', 'value'), State('choose_since_empty', 'value')
)
def start_agent_play(n, mode, previous_chain, agent_file, depth, width, empty):
    if n and mode == 'Watch agent':
        kill_chain(previous_chain)
        chain = f'a{random.randrange(100000)}'
        game_logic.__dict__[chain] = True
        agent = load_s3(agent_file)
        estimator = agent.evaluate
        game = Game()
        globals()[chain] = {
            'type': 'agent',
            'game': game,
            'step': 0,
        }
        game.thread_trial(estimator, depth=depth, width=width, since_empty=empty, stopper=chain)
        return chain, False
    else:
        raise PreventUpdate


# Agent Test callbacks
@app.callback(
    Output('stop_agent', 'style'), Output('current_process', 'data'),
    Input('replay_agent_button', 'n_clicks'),
    State('mode_text', 'children'),  State('current_process', 'data'), State('choose_stored_agent', 'value'),
    State('choose_depth', 'value'), State('choose_width', 'value'), State('choose_since_empty', 'value'),
    State('choose_num_eps', 'value')
)
def start_agent_test(n, mode, previous_proc, agent_file, depth, width, empty, num_eps):
    if n and mode == 'Test agent':
        kill_process(previous_proc)
        agent = load_s3(agent_file)
        estimator = agent.evaluate
        params = {'depth': depth, 'width': width, 'since_empty': empty, 'num': num_eps,
                  'console': 'web', 'game_file': 'g/best_of_last_trial.pkl'}
        proc = f'p_{random.randrange(100000)}'
        LOGS.clear(start=f'Trial run for {num_eps} games, Agent = {agent.name}')
        globals()[proc] = Process(target=Game.trial, args=(estimator,), kwargs=params, daemon=True)
        globals()[proc].start()
        return {'display': 'block'}, proc
    else:
        raise PreventUpdate


# Agent Train callbacks
@app.callback(
    Output('choose_train_agent', 'options'),
    Output('choose_config', 'options'), Output('choose_config', 'value'),
    Input('input_group_train', 'style')
)
def find_agents(style):
    if style['display'] == 'none':
        raise PreventUpdate
    agents = [v for v in list_names_s3() if v[:2] == 'a/']
    configs = [v for v in list_names_s3() if v[:2] == 'c/']
    agent_options = [{'label': v[2:-4], 'value': v} for v in agents] + [{'label': 'New agent', 'value': 'New agent'}]
    conf_options = [{'label': v[2:-5], 'value': v} for v in configs] + [{'label': 'New config', 'value': 'New config'}]
    return agent_options, conf_options, None


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
    [Output(f'par_{e}', 'disabled') for e in params_list] + [Output(f'par_{e}', 'value') for e in params_list] +
    [Output('fill_loading', 'className')],
    Input('params_page', 'is_open'),
    State('choose_train_agent', 'value'), State('choose_config', 'value'),
)
def fill_params(is_open, agent_name, config_name):
    if is_open:
        if agent_name != 'New agent':
            agent = load_s3(agent_name)
            dis = [params_dict[e]['disable'] for e in params_list]
            ui_params = [getattr(agent, e) for e in params_list[:-1]] + [params_dict['Training episodes']['value']]
        elif config_name != 'New config':
            config = load_s3(config_name)
            dis = [False for e in params_list]
            ui_params = [config.get(e, params_dict[e]['value']) for e in params_list]
        else:
            dis = [False for e in params_list]
            ui_params = [params_dict[e]['value'] for e in params_list]
        return dis + ui_params + [NUP]
    else:
        raise PreventUpdate


@app.callback(
    Output('params_notification', 'children'), Output('current_process', 'data'),
    Output('choose_train_agent', 'options'), Output('choose_train_agent', 'value'), Output('loading', 'className'),
    Output('mode_text', 'children'), Output('input_group_train', 'style'),
    Input('start_training', 'n_clicks'),
    [State(f'par_{e}', 'value') for e in params_list] +
    [State('choose_train_agent', 'value'), State('current_process', 'data')]
)
def start_training(*args):
    if args[0]:
        message = NUP
        new_name, new_agent_file, current_process = args[1], args[-2], args[-1]
        ui_params = {e: args[i + 2] for i, e in enumerate(params_list[1:])}
        ui_params['n'] = int(ui_params['n'])
        bad_inputs = [e for e in ui_params if ui_params[e] is None]
        if bad_inputs:
            return my_alert(f'Parameters {bad_inputs} unacceptable', info=True), NUP, NUP, NUP, NUP, NUP, NUP
        kill_process(current_process)
        num_eps = ui_params.pop('Training episodes')
        name = ''.join(x for x in new_name if (x.isalnum() or x in ('_', '.')))
        if new_agent_file == 'New agent':
            new_config_file = f'c/config_{name}.json'
            save_s3(ui_params, new_config_file)
            message = my_alert(f'new config file {new_config_file[2:]} saved')
            current = Q_agent(name=name, config_file=new_config_file, storage='s3', console='web')
        else:
            current = load_s3(new_agent_file)
            if current.name != name:
                current.name = name
                current.file = current.name + '.pkl'
                current.game_file = 'best_of_' + current.file
            for e in ui_params:
                setattr(current, e, ui_params[e])
        current.logs = ''
        current.print = LOGS.add
        current.save_agent()
        proc = f'p_{random.randrange(100000)}'
        LOGS.clear(start='')
        globals()[proc] = Process(target=current.train_run, kwargs={'num_eps': num_eps}, daemon=True)
        globals()[proc].start()
        if name != new_name:
            agents = [v for v in list_names_s3() if v[:2] == 'a/']
            opts = [{'label': v[2:-4], 'value': v} for v in agents] + [{'label': 'New agent', 'value': 'New agent'}]
        else:
            opts = NUP
        return message, proc, opts, f'a/{current.file}', NUP, 'Choose:', {'display': 'none'}
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


@app.callback(
    Output('logs_display', 'children'),
    Input('clear_logs', 'n_clicks')
)
def clear_logs(n):
    if n:
        LOGS.clear()
        return NUP
    else:
        raise PreventUpdate


@app.callback(
    Output('download_file', 'data'),
    Input('download_logs', 'n_clicks')
)
def download_logs(n):
    if n:
        return dcc.send_file(LOGS.file)
    else:
        raise PreventUpdate


@app.callback(
    Output('stop_agent', 'style'),
    Input('current_process', 'data'),
)
def stop_agent(current_process):
    return {'display': 'block' if current_process else 'none'}


@app.callback(
    Output('current_process', 'data'), Output('stop_agent', 'style'),
    Input('stop_agent', 'n_clicks'),
    State('current_process', 'data'),
)
def stop_agent(n, current_process):
    if n:
        if current_process:
            kill_process(current_process)
            LOGS.add('Process terminated by user')
        return None, {'display': 'none'}
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


# Chart callbacks
@app.callback(
    Output('chart_button', 'style'), Output('chart_button', 'children'), Output('agent_for_chart', 'data'),
    Input('choose_train_agent', 'value'), Input('choose_stored_agent', 'value')
)
def enable_chart_button(*args):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    agent = ctx.triggered[0]['value']
    if agent == 'New agent':
        raise PreventUpdate
    return {'display': 'block'}, f'{agent[2: -4]} train history chart', agent


@app.callback(
    Output('chart_page', 'is_open'),
    Input('chart_button', 'n_clicks'), Input('close_chart', 'n_clicks'),
    State('chart_page', 'is_open'),
)
def toggle_chart_page(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@app.callback(
    Output('chart_header', 'children'), Output('chart', 'children'), Output('chart_loading', 'className'),
    Input('chart_page', 'is_open'),
    State('agent_for_chart', 'data')
)
def make_chart(is_open, agent_file):
    if is_open:
        agent = load_s3(agent_file)
        if agent is None:
            return '', f'No Agent with this name in storage', NUP
        history = agent.train_history
        header = f'Training history of {agent.name}'
        if not history:
            return header, 'No history yet', NUP
        x = np.array([v * 100 for v in range(1, len(history) + 1)])
        fig = px.line(x=x, y=history, labels={'x': 'number of episodes', 'y': 'Average score of last 100 games'})
        return header, dcc.Graph(figure=fig, style={'width': '100%', 'height': '100%'}), NUP
    else:
        raise PreventUpdate


# Play Yourself callbacks
@app.callback(
    Output('play_instructions', 'is_open'), Output('gauge_group', 'style'), Output('play-yourself-group', 'style'),
    Output('chain', 'data'), Output('update_interval', 'disabled'), Output('game_card', 'children'),
    Input('mode_text', 'children'),
    State('chain', 'data')
)
def play_yourself_start(mode, previous_chain):
    if mode:
        if mode == 'Play':
            kill_chain(previous_chain)
            chain = f'g{random.randrange(100000)}'
            game = Game()
            globals()[chain] = {
                'type': 'play',
                'game': game,
            }
            to_show = display_table(game.row, game.score, game.odometer, 0, self_play=True)
            return True, {'display': 'none'}, {'display': 'block'}, chain, True, to_show
        else:
            return False, {'display': 'block'}, {'display': 'none'}, NUP, NUP, NUP


@app.callback(
    Output('game_card', 'children'),
    Input('keyboard', 'n_keydowns'),
    State('keyboard', 'keydown'), State('mode_text', 'children'), State('chain', 'data')
)
def keyboard_play(n, event, mode, chain):
    if n and mode == 'Play':
        key = json.dumps(event).split('"')[3]
        if key.startswith('Arrow'):
            move = keyboard_dict[key[5:]]
            game = globals()[chain]['game']
            new_row, new_score, change = game.pre_move(game.row, game.score, move)
            if not change:
                raise PreventUpdate
            game.odometer += 1
            game.row, game.score = new_row, new_score
            game.new_tile()
            next_move = -1 if game.game_over(game.row) else 0
            return display_table(game.row, game.score, game.odometer, next_move, self_play=True)
        else:
            raise PreventUpdate
    else:
        raise PreventUpdate


@app.callback(
    Output('game_card', 'children'),
    [Input(f'move_{i}', 'n_clicks') for i in range(4)],
    State('chain', 'data')
)
def button_play(*args):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    move = int(ctx.triggered[0]['prop_id'].split('.')[0][-1])
    game = globals()[args[-1]]['game']
    new_row, new_score, change = game.pre_move(game.row, game.score, move)
    if not change:
        raise PreventUpdate
    game.odometer += 1
    game.row, game.score = new_row, new_score
    game.new_tile()
    next_move = -1 if game.game_over(game.row) else 0
    return display_table(game.row, game.score, game.odometer, next_move, self_play=True)


@app.callback(
    Output('game_card', 'children'),
    Input('restart_play', 'n_clicks'),
    State('chain', 'data')
)
def restart_play(n, chain):
    if n:
        game = Game()
        globals()[chain]['game'] = game
        return display_table(game.row, game.score, game.odometer, 0, self_play=True)
    else:
        raise PreventUpdate


app.clientside_callback(
    ClientsideFunction(namespace='clientside', function_name='make_draggable'),
    Output('play_instructions', 'className'),
    State('play_instructions', 'id'), Input('play_instructions', 'is_open')
)


if __name__ == '__main__':

    LOGS.clear()
    app.run_server(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)), debug=(LOCAL == 'local'), use_reloader=False)
