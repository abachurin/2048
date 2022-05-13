from game2048.show import *


image_directory = os.path.dirname(os.path.realpath(__file__)) + '/image/'
app = DashProxy(__name__, transforms=[MultiplexerTransform()],
                meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1'}])

app.title = 'RL Agent 2048'
app.layout = dbc.Container([
    dcc.Store(id='next_move', storage_type='session'),
    dcc.Interval(id='update_moves', interval=2000, n_intervals=0),
    dbc.Row(html.H2('Reinforcement Learning 2048 Agent', className='header card-header')),
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.DropdownMenu(id='choose_option', label='Choose option:',
                             children=[
                                 dbc.DropdownMenuItem('Train Agent', id='train_agent_button', n_clicks=0),
                                 dbc.DropdownMenuItem('Watch Agent play', id='watch_agent_button', n_clicks=0),
                                 dbc.DropdownMenuItem('Collect Agent statistics', id='agent_stat_button', n_clicks=0),
                                 dbc.DropdownMenuItem('Replay best game', id='replay_best_button', n_clicks=0),
                                 dbc.DropdownMenuItem('Play', id='play_button', n_clicks=0),
                             ])
            ], className='log-box')
        ),
        dbc.Col(
            dbc.Card(
                dbc.CardBody(id='game_card'), className='game-box align-items-center')
        )
    ])
])


@app.callback(
    Input('replay_best_button', 'n_clicks'),
    Output('game_card', 'children')
)
def start_play(n):
    if n:
        game = load_s3('best_game.pkl')
        game.trial_run(estimator=score_eval)
        return display_table(game)
    else:
        raise PreventUpdate


if __name__ == '__main__':

    app.run_server(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)), debug=(LOCAL == 'local'), use_reloader=False)
