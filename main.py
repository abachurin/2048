from game2048.show import *


image_directory = os.path.dirname(os.path.realpath(__file__)) + '/image/'
app = DashProxy(__name__, transforms=[MultiplexerTransform()],
                meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1'}])

app.layout = html.Div([
    dbc.Row(html.H1('Ola!', className='card-header')),
])



if __name__ == '__main__':

    app.run_server(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)), debug=(LOCAL == 'local'), use_reloader=False)
