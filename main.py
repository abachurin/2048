import html.entities

from game2048.show import *


image_directory = os.path.dirname(os.path.realpath(__file__)) + '/image/'
app = DashProxy(__name__, transforms=[MultiplexerTransform()],
                meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1'}])

app.title = 'RL Agent 2048'
app.layout = html.Div([
    html.H1('Reinforcement Learning 2048 Agent', className='header card-header'),
    dbc.Row([
        dbc.Col(dbc.Card(
            dbc.CardHeader('Logs')
        )),
        dbc.Col(dbc.Card(
            dbc.CardHeader('Game')
        )),
    ])
])



if __name__ == '__main__':

    app.run_server(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)), debug=(LOCAL == 'local'), use_reloader=False)
