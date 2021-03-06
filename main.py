from game2048.show import *
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import random

external_stylesheets = ['look.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.H3("What is your name dear?"),
    html.Div(["Come on, no need to be so shy: ",
              dcc.Input(id='my-input', value='?', type='text')]),
    html.Br(),
    html.Div(id='my-output'),
])


def generate_compliment(name):
    compliments = [f"Take the rest of the day off, {name}! You clearly need it",
                   f"Good afternoon, {name}! Thank you for showing up",
                   f"Great input, {name}! Now make yourself a cup of tea and let adults work",
                   f"Thank you, captain {name}! Now we know what to do yesterday"]
    return random.choice(compliments)


@app.callback(
    Output(component_id='my-output', component_property='children'),
    Input(component_id='my-input', component_property='value')
)
def update_output_div(name):
    if name == "?":
        return ""
    if name == "ok":
        agent = Q_agent.load_agent("best_agent.npy")
        est = agent.evaluate
        results = Game.trial(estimator=est, num=100)
    return generate_compliment(name)


if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8050, debug=True)
