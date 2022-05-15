from game2048.show import *

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

cell_size = 100
x_position = {i: f'{i * cell_size}px' for i in range(4)}
y_position = {i: f'{i * cell_size + 35}px' for i in range(4)}
numbers = {i: str(1 << i) if i else '' for i in range(16)}
colors = load_s3('config.json')['colors']
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


def my_alert(text, info=False):
    return dbc.Alert(f' {text}', color='info' if info else 'success', dismissable=True, duration=5000,
                     className='admin-notification')


image_directory = os.path.dirname(os.path.realpath(__file__)) + '/image/'
app = DashProxy(__name__, transforms=[MultiplexerTransform()], title='RL Agent 2048', update_title=None,
                meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1'}])

app.layout = dbc.Container([
    dcc.Download(id="download_file"),
    dcc.Store(id='chain', storage_type='session'),
    dcc.Interval(id='update_interval', n_intervals=0, disabled=True),
    dbc.Row(html.H3('Reinforcement Learning 2048 Agent', className='header card-header')),
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.DropdownMenu(id='choose_option', label='Mode:',
                             children=[dbc.DropdownMenuItem(mode_list[v][0], id=v, n_clicks=0) for v in mode_list]),
            html.H6(id='mode_text', className='mode-text'),
            dbc.Select(id='choose_stored_agent', style={'display': 'none'}, className='choose-from-s3'),
            dbc.Select(id='choose_for_replay', style={'display': 'none'}, className='choose-from-s3'),
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
            ], id='admin_page', size='lg', centered=True, contentClassName='admin-page')
            ], className='log-box')
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


@app.callback(
    Output('admin_page', 'is_open'),
    Input('open_admin', 'n_clicks'), Input('close_admin', 'n_clicks'),
    State('admin_page', 'is_open'),
)
def toggle_view_results_table(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@app.callback(
    [Input(v, 'n_clicks') for v in act_list],
    Output('choose_file', 'options'), Output('choose_file', 'value'), Output('admin_go', 'children')
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
            delete_s3(name)
            return my_alert(f'{name} deleted'), NUP
        elif act == 'Download':
            if name:
                s3_bucket.download_file(name, name)
                to_send = dcc.send_file(name)
                os.remove(name)
                return NUP, to_send
            else:
                return my_alert(f'Choose file for download!', info=True), NUP
        elif act == 'Upload':
            return NUP, NUP
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
    else:
        raise PreventUpdate


@app.callback(
    [Input(v, 'n_clicks') for v in mode_list],
    Output('mode_text', 'children'), Output('choose_stored_agent', 'style'), Output('choose_for_replay', 'style')
)
def mode_process(*args):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    idx = ctx.triggered[0]['prop_id'].split('.')[0]
    to_see_games = 'block' if idx == 'replay_button' else 'none'
    to_see_agents = 'block' if idx in ('watch_agent_button', 'agent_stat_button') else 'none'
    return mode_list[idx][1], {'display': to_see_agents}, {'display': to_see_games}


@app.callback(
    Input('choose_for_replay', 'style'),
    Output('choose_for_replay', 'options')
)
def find_games(style):
    if style['display'] == 'none':
        raise PreventUpdate
    games = [v for v in list_names_s3() if v[:2] == 'g/']
    return [{'label': v[2:-4], 'value': v} for v in games]


@app.callback(
    Input('choose_stored_agent', 'style'),
    Output('choose_stored_agent', 'options')
)
def find_agents(style):
    if style['display'] == 'none':
        raise PreventUpdate
    agents = [v for v in list_names_s3() if v[:2] == 'a/']
    return [{'label': v[2:-4], 'value': v} for v in agents]


@app.callback(
    Input('gauge-slider', 'value'),
    Output('gauge', 'value'), Output('update_interval', 'interval'),
)
def update_output(value):
    return value, value * 200 + 50


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
