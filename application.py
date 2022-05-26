from game2048.dash_utils import *


# App declaration and layout
app = DashProxy(__name__, transforms=[MultiplexerTransform()], title='RL Agent 2048', update_title=None,
                meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1'}])
application = app.server

app.layout = dbc.Container([
    dcc.Interval(id='refresh_status', interval=dash_intervals['refresh']),
    dcc.Store(id='session_tags', storage_type='session'),
    dcc.Interval(id='initiate_logs', interval=dash_intervals['initiate_logs'], n_intervals=0),
    dcc.Store(id='log_file', data=None, storage_type='session'),
    dcc.Store(id='running_now', storage_type='session'),
    dcc.Download(id='download_file'),
    Keyboard(id='keyboard'),
    dcc.Store(id='chain', storage_type='session'),
    dcc.Store(id='current_process', storage_type='session'),
    dcc.Store(id='agent_for_chart', storage_type='session'),
    dcc.Interval(id='update_interval', n_intervals=0, disabled=True),
    dcc.Interval(id='logs_interval', interval=dash_intervals['logs'], n_intervals=0),
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
                while_loading('test_loading', 225),
                while_loading('agent_play_loading', 225),
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
                    html.Div(id='logs_display', className='logs-display'),
                    html.Div(id='log_footer', className='card-footer log-footer')
                ], className='logs-window'),
            ], className='log-box'),
        ),
        dbc.Col([
            dbc.Card([
                dbc.Toast('Use buttons below or keyboard!\n'
                          'When two equal tiles collide, they combine to give you one '
                          'greater tile that displays their sum. The more you do this, obviously, the higher the '
                          'tiles get and the more crowded the board becomes. Your objective is to reach highest '
                          'possible score before the board fills up', header='Game instructions',
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


# refresh status, to keep parallel processes from closing down while the app is open in the browser,
# script "vacuum_cleaner" is killing them afterwards
@app.callback(
    Output('refresh_status', 'disabled'),
    Input('refresh_status', 'n_intervals'),
    State('session_tags', 'data')
)
def refresh_status(n, tags):
    if n and tags:
        status = load_s3('status.json')
        for key in status:
            value = tags[key]
            if value in status[key]:
                status[key][value]['finish'] = next_time()
        save_s3(status, 'status.json')
    raise PreventUpdate


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
    if idx == 'upload':
        files = ['config file', 'game', 'agent']
    elif idx == 'download':
        files = list_names_s3()
    else:
        files = [v for v in list_names_s3() if v != 'status.json']
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
                return NUP, dash_send(name)
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
        chain = f'g{time_suffix()}'
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
        # Agent Play
        elif point['type'] == 'agent':
            step, game = point['step'], point['game']
            if point['step'] >= game.odometer and not game.game_over(game.row):
                return NUP, NUP
            if step == -1:
                return NUP, True
            row, score, next_move = game.history[step]
        # Play Yourself
        else:
            raise PreventUpdate

        to_show = display_table(row, score, step, next_move)
        point['step'] = -1 if next_move == -1 else point['step'] + 1
        return to_show, NUP
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
    Output('chain', 'data'), Output('update_interval', 'disabled'), Output('agent_play_loading', 'className'),
    Input('replay_agent_button', 'n_clicks'),
    State('mode_text', 'children'), State('chain', 'data'), State('choose_stored_agent', 'value'),
    State('choose_depth', 'value'), State('choose_width', 'value'), State('choose_since_empty', 'value')
)
def start_agent_play(n, mode, previous_chain, agent_file, depth, width, empty):
    if n and mode == 'Watch agent':
        kill_chain(previous_chain)
        chain = f'a{time_suffix()}'
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
        return chain, False, NUP
    else:
        raise PreventUpdate


# Agent Test callbacks
@app.callback(
    Output('stop_agent', 'style'), Output('current_process', 'data'), Output('running_now', 'data'),
    Output('session_tags', 'data'), Output('test_loading', 'className'),
    Input('replay_agent_button', 'n_clicks'),
    State('mode_text', 'children'),  State('current_process', 'data'), State('choose_stored_agent', 'value'),
    State('choose_depth', 'value'), State('choose_width', 'value'), State('choose_since_empty', 'value'),
    State('choose_num_eps', 'value'), State('log_file', 'data'), State('session_tags', 'data')
)
def start_agent_test(n, mode, previous_proc, agent_file, depth, width, empty, num_eps, log_file, tags):
    if n and mode == 'Test agent':
        kill_process(previous_proc)
        agent = load_s3(agent_file)
        estimator = agent.evaluate
        params = {'depth': depth, 'width': width, 'since_empty': empty, 'num': num_eps, 'console': 'web',
                  'log_file': log_file, 'game_file': 'g/best_of_last_trial.pkl'}
        save_s3(f'Trial run for {num_eps} games, Agent = {agent.name}', log_file)
        proc = Process(target=Game.trial, args=(estimator,), kwargs=params, daemon=True)
        proc.start()
        pid = str(proc.pid)
        add_status('proc', pid)
        tags['proc'] = pid
        return {'display': 'block'}, {'pid': pid}, 'testing', tags, NUP
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
    Output('params_page', 'is_open'), Output('start_training', 'disabled'),
    Input('go_to_params', 'n_clicks')
)
def open_params_page(n):
    if n:
        return True, True
    else:
        raise PreventUpdate


@app.callback(
    Output('params_page', 'is_open'),
    Input('close_params', 'n_clicks'),
)
def close_params_page(n):
    if n:
        return False
    else:
        raise PreventUpdate


@app.callback(
    [Output(f'par_{e}', 'disabled') for e in params_list] + [Output(f'par_{e}', 'value') for e in params_list] +
    [Output('start_training', 'disabled'), Output('fill_loading', 'className')],
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
        return dis + ui_params + [False, NUP]
    else:
        raise PreventUpdate


@app.callback(
    Output('params_notification', 'children'), Output('current_process', 'data'),
    Output('choose_train_agent', 'options'), Output('choose_train_agent', 'value'),
    Output('start_training', 'disabled'), Output('loading', 'className'),
    Output('mode_text', 'children'), Output('input_group_train', 'style'),
    Output('running_now', 'data'), Output('session_tags', 'data'),
    Input('start_training', 'n_clicks'),
    [State(f'par_{e}', 'value') for e in params_list] +
    [State('choose_train_agent', 'value'), State('current_process', 'data'),
     State('log_file', 'data'), State('session_tags', 'data')]
)
def start_training(*args):
    if args[0]:
        message = NUP
        new_name, new_agent_file, current_process, log_file, tags = args[1], args[-4], args[-3], args[-2], args[-1]
        ui_params = {e: args[i + 2] for i, e in enumerate(params_list[1:])}
        ui_params['n'] = int(ui_params['n'])
        bad_inputs = [e for e in ui_params if ui_params[e] is None]
        if bad_inputs:
            return [my_alert(f'Parameters {bad_inputs} unacceptable', info=True)] + [NUP] * 9
        name = ''.join(x for x in new_name if (x.isalnum() or x in ('_', '.')))
        if name == 'test_agent':
            name = f'test_{time_suffix()}'
        num_eps = ui_params.pop('Training episodes')
        if new_agent_file == 'New agent':
            if f'a/{name}.pkl' in list_names_s3():
                return [my_alert(f'Agent with {name} already exists!', info=True)] + [NUP] * 9
            new_config_file = f'c/config_{name}.json'
            save_s3(ui_params, new_config_file)
            message = my_alert(f'new config file {new_config_file[2:]} saved')
            current = Q_agent(name=name, config_file=new_config_file, storage='s3', console='web')
        else:
            current = load_s3(new_agent_file)
            if current.name != name:
                if f'a/{name}.pkl' in list_names_s3():
                    return [my_alert(f'Agent with {name} already exists!', info=True)] + [NUP] * 9
                current.name = name
                current.file = current.name + '.pkl'
                current.game_file = 'best_of_' + current.file
            else:
                if name in load_s3('status.json')['agent']:
                    return [my_alert(f'Agent {name} is being trained by another user', info=True)] + [NUP] * 9
            for e in ui_params:
                setattr(current, e, ui_params[e])
        kill_process(current_process)
        current.log_file = log_file
        current.print = Logger(log_file=log_file).add
        save_s3('', log_file)
        add_status('agent', name)
        tags['agent'] = name
        current.save_agent()
        proc = Process(target=current.train_run, kwargs={'num_eps': num_eps}, daemon=True)
        proc.start()
        pid = str(proc.pid)
        add_status('proc', pid)
        tags['proc'] = pid
        if name != new_name:
            agents = [v for v in list_names_s3() if v[:2] == 'a/']
            opts = [{'label': v[2:-4], 'value': v} for v in agents] + [{'label': 'New agent', 'value': 'New agent'}]
        else:
            opts = NUP
        return message, {'train': name, 'pid': pid}, opts, f'a/{current.file}', True, NUP, 'Choose:',\
            {'display': 'none'}, 'training', tags
    else:
        raise PreventUpdate


# Game Board callbacks
@app.callback(
    Output('gauge', 'value'), Output('update_interval', 'interval'),
    Input('gauge-slider', 'value')
)
def update_output(value):
    return value, value * 200 + LOWEST_SPEED


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
            chain = f'g{time_suffix()}'
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


# Log window callbacks
@app.callback(
    Output('log_file', 'data'), Output('session_tags', 'data'), Output('initiate_logs', 'disabled'),
    Input('initiate_logs', 'n_intervals'),
    State('session_tags', 'data'),
)
def assign_log_file(n, tags):
    if n:
        log_file = f'l/logs_{time_suffix()}.txt'
        add_status('logs', log_file)
        tags = {'logs': log_file, 'proc': 0, 'agent': 0}
        parent = str(os.getpid())
        Process(target=vacuum_cleaner, args=(parent,), daemon=True).start()
        return log_file, tags, True
    else:
        raise PreventUpdate


@app.callback(
    Output('log_footer', 'children'),
    Input('running_now', 'data')
)
def populate_log_footer(data):
    if data:
        return Logger.msg[data]
    else:
        return Logger.msg['welcome']


@app.callback(
    Output('logs_display', 'children'),
    Input('logs_interval', 'n_intervals'),
    State('log_file', 'data')
)
def update_logs(n, log_file):
    if n:
        return load_s3(log_file)
    else:
        raise PreventUpdate


@app.callback(
    Output('logs_display', 'children'),
    Input('clear_logs', 'n_clicks'),
    State('log_file', 'data')
)
def clear_logs(n, log_file):
    if n:
        save_s3('', log_file)
        return None
    else:
        raise PreventUpdate


@app.callback(
    Output('download_file', 'data'),
    Input('download_logs', 'n_clicks'),
    State('logs_display', 'children'),
)
def download_logs(n, current_logs):
    if n and current_logs:
        temp = f'temp{time_suffix()}.txt'
        with open(temp, 'w') as f:
            f.write(current_logs)
        to_send = dcc.send_file(temp)
        os.remove(temp)
        return to_send
    else:
        raise PreventUpdate


@app.callback(
    Output('stop_agent', 'style'),
    Input('current_process', 'data'),
)
def enable_stop_agent_button(current_process):
    return {'display': 'block' if current_process else 'none'}


@app.callback(
    Output('current_process', 'data'), Output('stop_agent', 'style'), Output('running_now', 'data'),
    Input('stop_agent', 'n_clicks'),
    State('current_process', 'data'), State('log_file', 'data')
)
def stop_agent(n, current_process, log_file):
    if n:
        kill_process(current_process)
        if current_process and 'train' in current_process:
            delete_status('agent', current_process['train'])
        now = load_s3(log_file) or ''
        save_s3(now + '\n' + Logger.msg['stop'], log_file)
        return None, {'display': 'none'}, None
    else:
        raise PreventUpdate


if __name__ == '__main__':

    # make_empty_status(); sys.exit()
    # app.run_server(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)), debug=(LOCAL == 'local'), use_reloader=False)
    application.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)), debug=(LOCAL == 'local'), use_reloader=False)
