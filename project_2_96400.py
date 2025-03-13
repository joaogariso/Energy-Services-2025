import pandas as pd
import numpy as np
import pickle

import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn import  metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor, 
    BaggingRegressor, 
    GradientBoostingRegressor
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

print('\n\nYou can find the online implementation of this dashboard on https://energy-services-2025-dashboard.onrender.com/.\n')



# == DATA ==
# Load full dataset
df_all      = pd.read_csv('data.17.18.19.csv', index_col=0, parse_dates=True)
date_start  = df_all.index.min()
date_end    = df_all.index.max()

# Split the data
train_test_cutoff   = '2019-01-01'
df_17_18            = df_all.loc[df_all.index <  train_test_cutoff]
df_19               = df_all.loc[df_all.index >= train_test_cutoff]
df_19.to_csv('df_19.csv', encoding='utf-8', index=True)

# Create predictor DataFrames 
df_17_18_predictors = df_17_18.drop('Power (kW)', axis=1)
df_19_predictors    = df_19.drop('Power (kW)', axis=1)

# Create new DataFrames for the Power (kW) column
df_17_18_power  = df_17_18[['Power (kW)']]
df_19_power     = df_19[['Power (kW)']]



final_fig = px.line(df_19, x=df_19.index, y='Power (kW)', labels={'x': 'Time', 'y': 'Power (kW)'})
final_fig.update_layout(
    title="Actual vs Forecasted Power Consumption",  # Main title
    title_x=0.5,  # Centers the title
    xaxis_title="Time",  # X-axis label
    yaxis_title="Power (kW)",  # Y-axis label
    legend_title="Legend",  # Legend title
    template="plotly_white"  # Makes the plot visually cleaner
)

# Global variables 
X_train = X_test = y_train = y_test = None
y_pred_list = []
y_pred_2019 = []

# == AUXILIARY FUNCTIONS ======
def generate_table(dataframe, max_rows=10):
    table_style = {
        'borderCollapse': 'collapse',
        'borderSpacing': '0',
        'width': '100%',
        'border': '1px solid #ddd',
        'fontFamily': 'Arial, sans-serif',
        'fontSize': '14px'
    }

    th_style = {
        'border': '1px solid #ddd',
        'padding': '8px',
        'textAlign': 'center',
        'backgroundColor': '#f2f2f2',
        'fontWeight': 'bold',
        'color': '#333'
    }

    td_style = {
        'border': '1px solid #ddd',
        'padding': '8px',
        'textAlign': 'center'
    }

    # Remove the index references and only display dataframe columns:
    return html.Table(
        style=table_style,
        children=[
            html.Thead(
                html.Tr([
                    html.Th(col, style=th_style) for col in dataframe.columns
                ])
            ),
            html.Tbody([
                html.Tr([
                    html.Td(dataframe.iloc[i][col], style=td_style) for col in dataframe.columns
                ]) for i in range(min(len(dataframe), max_rows))
            ])
        ]
    )

def generate_graph(df, columns, start_date, end_date):
    """
    Generates a Plotly figure for the selected columns over a specified date range.

    Parameters:
    - df: Pandas DataFrame with a DateTime index.
    - columns: List of column names to include in the graph.
    - start_date: Start date for filtering the DataFrame.
    - end_date: End date for filtering the DataFrame.

    Returns:
    - fig: A Plotly figure with multiple line plots and properly configured axes.
    """

     # Filter the df based on the date range and selected columnsm, and initialize the figure
    filtered_df = df.loc[start_date:end_date, columns]
    fig = go.Figure()

    # Dynamic multiple y-axis configuration
    y_axes = {}

    for i, column in enumerate(columns):
        axis_name       = f"y{i+1}" if i > 0 else "y"
        y_axis_config   = dict(
            title       =column,
            overlaying  ="y" if i > 0 else None,
            side        ="rigth" if i > 0 else None,
            poisiton    =0.05 * i if i > 0 else None
        )

        # Add the y-axis config to the dictionary
        y_axes[axis_name] = y_axis_config

        # Add trace to figure
        fig.add_trace(go.Scatter(x=filtered_df.index, y=filtered_df[column], name=column, yaxis=axis_name))

    # Apply layout updates
    fig.update_layout(
        title       =", ".join(columns),
        xaxis_title = 'Date',
        **y_axes
    )
    
    return fig


# == DASHBOARD ORGANIZATION =====
app     = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server  = app.server

app.layout = html.Div(style={'backgroundColor':'white'}, children=[
    html.H1('Energy Services Project 2 - North Tower Energy Consumption Forecast'),
    html.H3('João Gariso - 96400'),
    # html.Div(id='df_17_18', children=df_17_18.to_json(orient='split'), style={'display': 'none'}),
    dcc.Tabs(id='tabs', value='dataset_explorer_tab', children=[

        dcc.Tab(label='Dataset Explorer', value='dataset_explorer_tab', children=[
            html.Div([
                html.H2('Dataset Explorer'),
                html.P('Select the features you would like to explore, along with the date range for which you want to analyze the data.'),
                html.Div([
                    html.Div([
                        dcc.Dropdown(
                            id          ='raw_features_dropdown',
                            options     =[{'label': ft, 'value': ft} for ft in df_17_18.columns],
                            value       =[df_17_18.columns[0]],
                            multi       = True
                        )
                    ],style={
                        'flex':'2',
                        'min-width':'60%'
                    }),
                    # html.P('Set the time period:'),
                    html.Div([
                        dcc.DatePickerRange(
                            id                  ='raw_data_date_picker',
                            min_date_allowed    =df_17_18.index.min(),
                            start_date          =df_17_18.index.min(),
                            max_date_allowed    =df_17_18.index.max(),
                            end_date            =df_17_18.index.max()
                        )
                    ],style={
                        'flex':'1',
                        'max-width':'18%',
                        'min-width':'15%'
                    })
                ], style={
                    'display':'flex',
                    'gap':'20px',
                    'alignItems':'center',
                }),
                html.H4('Time Plot'),
                dcc.Graph(
                    id='raw_data_graph'
                ),
                html.H4('Box Plots'),
                dcc.Graph(
                    id='raw_data_box_plot'
                )                                           
            ])
        ]),
        dcc.Tab(
            label='Feature & Model Selection', 
            value='feature_model_selection_tab', 
            children=[
                html.Div(
                    style={
                        'display': 'flex',
                        'gap': '40px', 
                        'alignItems': 'flex-start'
                        }, 
                    children=[
                        # Left section: Paragraph, Dropdowns, and Button
                        html.Div(
                            style={
                                'flex': '2', 
                                'max-width': '40%',
                                'min-width': '300px',                                
                                },  
                            children=[
                                html.H2('Feature & Model Selection'),
                                html.P('Select the features for training, choose a model type, and click "Train" to start.'),
                                dcc.Dropdown(
                                    id='selection_features_dropdown',
                                    options=[{'label': ft, 'value': ft} for ft in df_17_18_predictors.columns],
                                    value=[df_17_18.columns[0]],
                                    multi=True,
                                    style={'margin-bottom': '10px'}
                                ),
                                dcc.Dropdown(
                                    id='model_dropdown',
                                    options=[
                                        {'label': 'Linear Regression', 'value': 'linear'},
                                        {'label': 'Gradient Boosting', 'value': 'gradint_boosting'},
                                        {'label': 'Random Forests', 'value': 'random_forests'},
                                        {'label': 'Bootstrapping', 'value': 'bootstrapping'},
                                        {'label': 'Decision Tree Regressor', 'value': 'decision_trees'},
                                        {'label': 'Neural Networks', 'value': 'neural_networks'}
                                    ],
                                    value='linear',
                                    style={'margin-bottom': '10px'}
                                ),
                                html.Button('Train', id='feature_submission'),
                            ]
                        ),
                        
                        # Right section: Plot
                        html.Div(
                            style={
                                'flex': '2',
                                'min-width': '500px',
                                },  # Take up more space for the graph
                            children=[
                                dcc.Loading(
                                    id="loading-1",
                                    children=[dcc.Graph(id="predicted_actual_plot")]
                                )
                            ]
                        )
                    ]
                )
            ]
        ),
        

        dcc.Tab(
            label='Model Predictions', 
            value='model_predictions_tab', 
            children=[
                html.H2('Model Predictions'),
                html.P("Click the button below to run the model and generate predictions for the 2019 power consumption, allowing you to compare the estimated values with the actual data. Don't forget to train the model on the previous tab"),
                html.Button('Run Model', id='button_run_model'),
                html.Div(
                    style={
                        'display':'flex',
                        'gap':'40px',
                        'alignItems':'center',
                        'justifyContent': 'center'
                    },
                    children=[
                        html.Div(
                            style={
                                'flex':'2',
                                'max-width':'75%',
                                'min-width':'400px',
                                'alignItems': 'center',
                                'justifyContent': 'center',
                                'border-right': '2px solid #ccc',
                                'padding-right': '15px'
                            }, 
                            children=[
                                dcc.Graph(id='final_plot', figure=final_fig)
                            ]
                        ),
                        html.Div(
                            style={
                                'flex':'1',
                                'min-width':'150px',
                                'alignItems': 'center',
                                'justifyContent': 'center',
                                'padding-right': '15px'
                            },
                            children=[
                                html.Div(id='error_table')
                            ]
                        )
                    ]
                )



                # html.Div([
                #     html.H2('Model Predictions'),
                #     html.P('Click the button below to run the model and generate predictions for the 2019 power consumption, allowing you to compare the estimated values with the actual data.'),
                #     html.Button('Run Model', id='button_run_model'),
                #     dcc.Graph(id='final_plot', figure=final_fig),
                #     html.Div(id='error_table')                
                #     ]
                # )
            ]
        )
    ]),
    html.Div(id='tabs_content')
])


# == CALLBACKS =====
@app.callback(
        Output('raw_data_graph', 'figure'),
        Input('raw_features_dropdown','value'),
        Input('raw_data_date_picker','start_date'),
        Input('raw_data_date_picker','end_date')
              )
def update_fig(columns, start_date, end_date):
    # Filter DataFrame based on date range
    filtered_df = df_17_18.loc[start_date:end_date, columns]

    y_axes =[]

    for i, column in enumerate(columns):
        y_axes.append({
            'overlaying':'y',
            'side':'rigth',
            'position': 1 - i * 0.1
        })

    data = [{
        'x'     :filtered_df.index,
        'y'     :filtered_df[column],
        'type'  :'line',
        'name'  :column
        } for column in filtered_df.columns]
    layout={
        'title':{
            'text':', '.join(columns)
        },
        'xaxis':{
            'title':'Date'
        }
    }

    layout.update({
        'yaxis{}'.format(i+1):y_axes[i] for i in range(len(y_axes))
    })

    fig ={
        'data'  :data,
        'layout':layout
    }

    return fig


@app.callback(
    Output('raw_data_box_plot', 'figure'),
    [
        Input('raw_features_dropdown', 'value'),
        Input('raw_data_date_picker', 'start_date'),
        Input('raw_data_date_picker', 'end_date')
    ]
)
def update_box_plots(columns, start_date, end_date):
    """
    Generates a figure with multiple box plots (subplots), 
    one for each selected feature over the specified date range.
    """
    
    # 1. Filter the DataFrame by date range and columns
    filtered_df = df_17_18.loc[start_date:end_date, columns]

    # 2. Create a subplot figure with one column per feature
    #    rows=1, cols=len(columns)
    fig = make_subplots(
        rows=1,
        cols=len(columns),
        subplot_titles=[col for col in columns],  # Subplot titles
        shared_yaxes=False  # Each subplot can have its own scale if you prefer
    )

    # 3. Add one Box trace per feature
    for i, col in enumerate(columns):
        fig.add_trace(
            go.Box(y=filtered_df[col], name=col),
            row=1,
            col=i+1
        )

    # 4. Update layout with a main title and optional styling
    fig.update_layout(
        title_text="Box Plots for Selected Features",
        title_x=0.5,  # Center the title
        template="plotly_white"
    )

    return fig




@app.callback(
    Output('predicted_actual_plot', 'figure'),
    Input('feature_submission', 'n_clicks'),   # Single button
    State('model_dropdown', 'value'),          # Model choice
    State('selection_features_dropdown','value')
)
def train_and_predict(n_clicks, model_type, selected_features):

    # global X_train, X_test, y_train, y_test
    global y_pred_list, y_pred_2019

    # If the button hasn't been clicked, do nothing
    if not n_clicks:
        return dash.no_update
    
    
    # 1. Prepare Data
    X = df_17_18_predictors[selected_features]
    y = df_17_18_power['Power (kW)']

    X_2019 = df_19_predictors[selected_features]

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)


    # 2. Select and train model
    if model_type == 'linear':
        model = LinearRegression()

    elif model_type == 'gradint_boosting':
        model = GradientBoostingRegressor()

    elif model_type == 'random_forests':
        # Example parameters, feel free to adjust
        parameters = {
            'bootstrap': True,
            'min_samples_leaf': 3,
            'n_estimators': 200, 
            'min_samples_split': 15,
            'max_features': 'sqrt',
            'max_depth': 20
        }
        model = RandomForestRegressor(**parameters)

    elif model_type == 'bootstrapping':
        model = BaggingRegressor()

    elif model_type == 'decision_trees':
        model = DecisionTreeRegressor()

    elif model_type == 'neural_networks':
        # Basic MLP, feel free to specify hidden_layer_sizes, etc.
        model = MLPRegressor(max_iter=1000)

    else:
        # Fallback model
        model = LinearRegression()

    model.fit(X_train, y_train)

    # 3. Make Predictions
    y_pred = model.predict(X_test)
    y_pred_list.append(y_pred)  # Store predictions for other uses if needed

    # Predict on 2019
    y_pred_2019_vals = model.predict(X_2019)
    y_pred_2019 = y_pred_2019_vals  

    # 4. Save Trained Model
    with open('model.pkl','wb') as file:
        pickle.dump(model, file)

    # 5. Create Plotly figure
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=y_test.values.flatten(),
        y=y_pred,
        mode='markers',
        name='Predictions'
    ))
        
    fig.update_layout(
        title=f"{model_type.capitalize()} Model: Actual vs Predicted",
        xaxis_title='Actual',
        yaxis_title='Predicted'
    )

    return fig


@app.callback(
    Output('final_plot', 'figure'),
    Output('error_table', 'children'),
    Input('button_run_model', 'n_clicks')
)
def run_model(n_clicks):

    if not n_clicks:
        raise PreventUpdate  # Do nothing if button hasn't been clicked

    # If your df_19 is already indexed by a DateTimeIndex, there's no need to re-set the index:
    df_real = df_19.copy()  
    # Ensure you have the real values in a column named "Power (kW)"
    # The predictions are stored in global y_pred_2019 from your training callback
    
    # 1) Generate the figure
    fig = go.Figure(layout=go.Layout(title='Real vs Predicted Power Consumption (2019)'))
    fig.add_scatter(x=df_real.index, y=df_real['Power (kW)'], name='Real Power (kW)')
    fig.add_scatter(x=df_real.index, y=y_pred_2019, name='Predicted Power (kW)')

    # 2) Calculate model performance metrics
    mae = round(metrics.mean_absolute_error(df_real['Power (kW)'], y_pred_2019), 3)
    mbe = round(np.mean(df_real['Power (kW)'] - y_pred_2019), 3)
    mse = round(metrics.mean_squared_error(df_real['Power (kW)'], y_pred_2019), 3)
    rmse = round(np.sqrt(mse), 3)
    cvrmse = round(rmse / np.mean(df_real['Power (kW)']), 3)
    nmbe = round(mbe / np.mean(df_real['Power (kW)']), 3)  # Fixed closing parenthesis

    # Convert cvRMSE and NMBE to percentages
    cvRMSE_perc = "{:.2f}%".format(cvrmse * 100)
    NMBE_perc   = "{:.2f}%".format(nmbe * 100)

    # 3) Create the transposed table of metrics
    df_metrics = pd.DataFrame({
        "Metric": ["MAE", "MBE", "MSE", "RMSE", "cvRMSE", "NMBE"],
        "Value": [mae, mbe, mse, rmse, cvRMSE_perc, NMBE_perc]
    })

    # Ensure index is removed
    df_metrics = df_metrics.loc[:, ["Metric", "Value"]]  # Select only required columns

    # Generate the table
    table = generate_table(df_metrics)

    return fig, table

if __name__ == '__main__':
    app.run_server()