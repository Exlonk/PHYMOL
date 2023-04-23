
# By Exlonk Gil
import os
import pandas as pd
import plotly.express as px
import pandas as pd
import plotly.express as px
from dash import dash_table, html, dcc,  Input, Dash
from dash import Output
import plotly.express as px
import dash_bootstrap_components as dbc
from jupyter_dash import JupyterDash
from importlib.machinery import SourceFileLoader
ds = SourceFileLoader("add",os.path.join(os.path.dirname(__file__),
                                         'data_science.py')).load_module()  
from dash_bootstrap_templates import load_figure_template                     
from dash import DiskcacheManager
import dash 

# Universals

# Cache 

import diskcache
cache = diskcache.Cache("./cache")
background_callback_manager = DiskcacheManager(cache)          

# Functions

path_models= os.path.join(os.path.dirname(__file__),'models')
path_validation_curves = os.path.join(os.path.dirname(__file__),
                                      'validation_curves')
path_figures = os.path.join(os.path.dirname(__file__),'figures')
path_dataframes = os.path.join(os.path.dirname(__file__),'dataframes')

def load_plot_json(path,name):
    import plotly
    import json 
    import os
    with open(os.path.join(path,name+'.json'),'r+',encoding="utf-8") as f:
        read_ = f.read()
    figure = plotly.io.from_json(read_)
    return figure

# Layout style

load_figure_template("cosmo")

discrete_color_graph = px.colors.sequential.Plasma
background_color = '#1B2B35'
font_color = "#fff"

# ----------------------------- DATA INSIGHTS -------------------------------- #

# Pure Raw Dataframe 

df = pd.read_csv(os.path.join(os.path.dirname(__file__),'train.csv'))

table = ds.table(df,bgcolor=background_color,textcolor=font_color,bgheader='#40505c') 

shape = load_plot_json(path_figures,'shape') 
shape.update_layout(title_text="Shape <br><sup>With target</sup>",showlegend=False)
shape.update_layout(paper_bgcolor=background_color,plot_bgcolor=background_color,
                    font={'color':font_color})

info_fig = load_plot_json(path_figures,'info_fig') 
info_fig.update_layout(title_text="Data Type <br><sup>With target</sup>",showlegend=True)
info_fig.update_layout(paper_bgcolor=background_color,plot_bgcolor=background_color,
                       font={'color':font_color})

# Duplicates and Missing values

duplicates = load_plot_json(path_figures,'duplicates') 
duplicates.update_layout(paper_bgcolor=background_color,plot_bgcolor=background_color,
                         font={'color':font_color})

missing_values = load_plot_json(path_figures,'missing_values')
missing_values.update_layout(paper_bgcolor=background_color,
                             plot_bgcolor=background_color,font={'color':font_color})

# Histograms

histograms = []
for i in range(0,3):
    histograms.append(load_plot_json(path_figures,'histogram_'+str(i)))
    histograms[i].update_layout(paper_bgcolor=background_color,
                                plot_bgcolor=background_color,font={'color':font_color})
    histograms[i].update_annotations(font_color=font_color)

# Outliers 

box_plot_pure = load_plot_json(path_figures,'box_plot_pure')
box_plot_pure.update_layout(paper_bgcolor=background_color
                            ,plot_bgcolor=background_color,font={'color':font_color})
box_plot_lof = ds.load_plot_json(path_figures,'box_plot_ifo')
box_plot_lof.update_layout(paper_bgcolor=background_color,
                           plot_bgcolor=background_color,font={'color':font_color})
box_plot_ifo = ds.load_plot_json(path_figures,'box_plot_lof')
box_plot_ifo.update_layout(paper_bgcolor=background_color,
                           plot_bgcolor=background_color,font={'color':font_color})

# # ---------------------------- FEATURE SELECTION --------------------------- #

pearson_correlation_fig = ds.load_plot_json(path_figures,'pearson_correlation_fig')
pearson_correlation_fig_std = ds.load_plot_json(path_figures,'pearson_correlation_fig_std')
pearson_correlation_fig_ifo = ds.load_plot_json(path_figures,'pearson_correlation_fig_ifo')
pearson_correlation_fig_lof = ds.load_plot_json(path_figures,'pearson_correlation_fig_lof')
pearson_correlation_fig.update_layout(paper_bgcolor=background_color,
                                      plot_bgcolor=background_color,font={'color':font_color})
pearson_correlation_fig_std.update_layout(paper_bgcolor=background_color,
                                          plot_bgcolor=background_color,
                                          font={'color':font_color})
pearson_correlation_fig_ifo.update_layout(paper_bgcolor=background_color,
                                          plot_bgcolor=background_color,
                                          font={'color':font_color})
pearson_correlation_fig_lof.update_layout(paper_bgcolor=background_color,
                                          plot_bgcolor=background_color,
                                          font={'color':font_color})

spearman_correlation_fig = ds.load_plot_json(path_figures,'spearman_correlation_fig')
spearman_correlation_fig_std = ds.load_plot_json(path_figures,'spearman_correlation_fig_std')
spearman_correlation_fig_ifo = ds.load_plot_json(path_figures,'spearman_correlation_fig_ifo')
spearman_correlation_fig_lof = ds.load_plot_json(path_figures,'spearman_correlation_fig_lof')
spearman_correlation_fig.update_layout(paper_bgcolor=background_color,
                                       plot_bgcolor=background_color,font={'color':font_color})
spearman_correlation_fig_std.update_layout(paper_bgcolor=background_color,
                                           plot_bgcolor=background_color,
                                           font={'color':font_color})
spearman_correlation_fig_ifo.update_layout(paper_bgcolor=background_color,
                                           plot_bgcolor=background_color,
                                           font={'color':font_color})
spearman_correlation_fig_lof.update_layout(paper_bgcolor=background_color,
                                           plot_bgcolor=background_color,
                                           font={'color':font_color})

mutual_correlation_fig = ds.load_plot_json(path_figures,'mutual_correlation_fig')
mutual_correlation_fig.update_layout(paper_bgcolor=background_color,
                                     plot_bgcolor=background_color,
                                     font={'color':font_color})

df_feature = pd.read_csv(os.path.join((os.path.join(os.path.dirname(__file__),
                                                    'feature_selection')),
                                                    'feature_selection.csv'))
table_feature = ds.table(df_feature,bgcolor=background_color,textcolor=font_color,
                         bgheader='#40505c')

# <------------------------- MACHINE LEARNING MODELS ------------------------> #

# Baseline

baseline_accuracy_train = ds.load_plot_json(path_figures,'baseline_accuracy_train')
baseline_accuracy_cv = ds.load_plot_json(path_figures,'baseline_accuracy_cv')
baseline_accuracy_test = ds.load_plot_json(path_figures,'baseline_accuracy_test')
baseline_ind_train = ds.load_plot_json(path_figures,'baseline_ind_train')
baseline_ind_cv = ds.load_plot_json(path_figures,'baseline_ind_cv')
baseline_ind_test = ds.load_plot_json(path_figures,'baseline_ind_test')
baseline_bias_variance = ds.load_plot_json(path_figures,'baseline_bias_variance')
baseline_bias_variance.update_layout(paper_bgcolor=background_color,
                                     plot_bgcolor=background_color,font={'color':font_color})
baseline_bias_variance.update_traces(marker_line_width=0)
for k in ['train','cv','test']:
        vars()['baseline_accuracy_'+k].update_layout(paper_bgcolor=background_color,
                                                     plot_bgcolor=background_color,font={'color':font_color})
        vars()['baseline_accuracy_'+k].update_annotations(font_color=font_color)
        vars()['baseline_ind_'+k].update_layout(paper_bgcolor=background_color,
                                                plot_bgcolor=background_color,font={'color':font_color})
        vars()['baseline_ind_'+k].update_traces(marker_line_width=0)

# Linear Regression

linear_accuracy_train = ds.load_plot_json(path_figures,'linear_accuracy_train')
linear_accuracy_cv = ds.load_plot_json(path_figures,'linear_accuracy_cv')
linear_accuracy_test = ds.load_plot_json(path_figures,'linear_accuracy_test')
linear_ind_train = ds.load_plot_json(path_figures,'linear_ind_train')
linear_ind_cv = ds.load_plot_json(path_figures,'linear_ind_cv')
linear_ind_test = ds.load_plot_json(path_figures,'linear_ind_test')
linear_bias_variance = ds.load_plot_json(path_figures,'linear_bias_variance')
linear_bias_variance.update_layout(paper_bgcolor=background_color,
                                   plot_bgcolor=background_color,font={'color':font_color})
linear_bias_variance.update_traces(marker_line_width=0)
for k in ['train','cv','test']:
        vars()['linear_accuracy_'+k].update_layout(paper_bgcolor=background_color,
                                                   plot_bgcolor=background_color,font={'color':font_color})
        vars()['linear_accuracy_'+k].update_annotations(font_color=font_color)
        vars()['linear_ind_'+k].update_layout(paper_bgcolor=background_color,
                                              plot_bgcolor=background_color,font={'color':font_color})
        vars()['linear_ind_'+k].update_traces(marker_line_width=0)

# Polynomial

r2_train_cv_poly_graph = ds.load_plot_json(path_figures,'r2_train_cv_poly_graph')
mse_train_cv_poly_graph = ds.load_plot_json(path_figures, 'mse_train_cv_poly_graph')
r2_train_cv_poly_graph.update_layout(paper_bgcolor=background_color,
                                     plot_bgcolor=background_color,font={'color':font_color})
mse_train_cv_poly_graph.update_layout(paper_bgcolor=background_color,
                                      plot_bgcolor=background_color,font={'color':font_color})

poly_accuracy_train = ds.load_plot_json(path_figures,'poly_accuracy_train')
poly_accuracy_train.update_layout(paper_bgcolor=background_color,
                                  plot_bgcolor=background_color,font={'color':font_color})
poly_accuracy_train.update_annotations(font_color=font_color)
poly_ind_train = ds.load_plot_json(path_figures,'poly_ind_train')
poly_ind_train.update_traces(marker_line_width=0)
poly_ind_train.update_layout(paper_bgcolor=background_color,
                             plot_bgcolor=background_color,font={'color':font_color})

poly_accuracy_cv = ds.load_plot_json(path_figures,'poly_accuracy_cv')
poly_accuracy_cv.update_annotations(font_color=font_color)
poly_accuracy_cv.update_layout(paper_bgcolor=background_color,
                               plot_bgcolor=background_color,font={'color':font_color})
poly_ind_cv = ds.load_plot_json(path_figures,'poly_ind_cv')
poly_ind_cv.update_traces(marker_line_width=0)
poly_ind_cv.update_layout(paper_bgcolor=background_color,
                          plot_bgcolor=background_color,font={'color':font_color})

poly_accuracy_test = ds.load_plot_json(path_figures,'poly_accuracy_test')
poly_accuracy_test.update_annotations(font_color=font_color)
poly_accuracy_test.update_layout(paper_bgcolor=background_color,
                                 plot_bgcolor=background_color,font={'color':font_color})
poly_ind_test = ds.load_plot_json(path_figures,'poly_ind_test')
poly_ind_test.update_traces(marker_line_width=0)
poly_ind_test.update_layout(paper_bgcolor=background_color,plot_bgcolor=background_color,
                            font={'color':font_color})

poly_bias_variance = ds.load_plot_json(path_figures,'poly_bias_variance')
poly_bias_variance.update_layout(paper_bgcolor=background_color,
                                 plot_bgcolor=background_color,font={'color':font_color})
poly_bias_variance.update_traces(marker_line_width=0)

# Learning curve 

learning_curve = ds.load_plot_json(path_figures,'learning_curve')
learning_curve.update_layout(paper_bgcolor=background_color,
                             plot_bgcolor=background_color,font={'color':font_color})

# Kneighbors Model

validation_curve_1 = ds.load_plot_json(os.path.join(path_validation_curves,
                                                    'KNeighborsRegressor'),'validation_curve_1')
validation_curve_1.update_layout(paper_bgcolor=background_color,
                                 plot_bgcolor=background_color,font={'color':font_color})
validation_curve_2 = ds.load_plot_json(os.path.join(path_validation_curves,
                                                    'KNeighborsRegressor'),'validation_curve_2')
validation_curve_2.update_layout(paper_bgcolor=background_color,plot_bgcolor=background_color,font={'color':font_color})
validation_curve_3 = ds.load_plot_json(os.path.join(path_validation_curves,
                                                    'KNeighborsRegressor'),'validation_curve_3')
validation_curve_3.update_layout(paper_bgcolor=background_color,plot_bgcolor=background_color,font={'color':font_color})
validation_curve_4 = ds.load_plot_json(os.path.join(path_validation_curves,
                                                    'KNeighborsRegressor'),'validation_curve_4')
validation_curve_4.update_layout(paper_bgcolor=background_color,plot_bgcolor=background_color,font={'color':font_color})

knn_accuracy_train = ds.load_plot_json(path_figures,'knn_accuracy_train')
knn_accuracy_cv = ds.load_plot_json(path_figures,'knn_accuracy_cv')
knn_accuracy_test = ds.load_plot_json(path_figures,'knn_accuracy_test')
knn_ind_train = ds.load_plot_json(path_figures,'knn_ind_train')
knn_ind_cv = ds.load_plot_json(path_figures,'knn_ind_cv')
knn_ind_test = ds.load_plot_json(path_figures,'knn_ind_test')
knn_bias_variance = ds.load_plot_json(path_figures,'knn_bias_variance')
knn_bias_variance.update_layout(paper_bgcolor=background_color,
                                plot_bgcolor=background_color,font={'color':font_color})
knn_bias_variance.update_traces(marker_line_width=0)
for k in ['train','cv','test']:
        vars()['knn_accuracy_'+k].update_layout(paper_bgcolor=background_color,
                                                plot_bgcolor=background_color,
                                                font={'color':font_color})
        vars()['knn_accuracy_'+k].update_annotations(font_color=font_color)
        vars()['knn_ind_'+k].update_layout(paper_bgcolor=background_color,
                                           plot_bgcolor=background_color,
                                           font={'color':font_color})
        vars()['knn_ind_'+k].update_traces(marker_line_width=0)

# Neural Network

network_accuracy_train = ds.load_plot_json(path_figures,'network_accuracy_train')
network_accuracy_cv = ds.load_plot_json(path_figures,'network_accuracy_cv')
network_accuracy_test = ds.load_plot_json(path_figures,'network_accuracy_test')
network_ind_train = ds.load_plot_json(path_figures,'network_ind_train')
network_ind_cv = ds.load_plot_json(path_figures,'network_ind_cv')
network_ind_test = ds.load_plot_json(path_figures,'network_ind_test')
network_bias_variance = ds.load_plot_json(path_figures,'network_bias_variance')
network_bias_variance.update_layout(paper_bgcolor=background_color,
                                    plot_bgcolor=background_color,
                                    font={'color':font_color})
network_bias_variance.update_traces(marker_line_width=0)
for k in ['train','cv','test']:
        vars()['network_accuracy_'+k].update_layout(paper_bgcolor=background_color,
                                                    plot_bgcolor=background_color,
                                                    font={'color':font_color})
        vars()['network_accuracy_'+k].update_annotations(font_color=font_color)
        vars()['network_ind_'+k].update_layout(paper_bgcolor=background_color,
                                               plot_bgcolor=background_color,
                                               font={'color':font_color})
        vars()['network_ind_'+k].update_traces(marker_line_width=0)

neural_graph = ds.load_plot_json(path_figures,'neural_graph')
neural_graph.update_layout(paper_bgcolor=background_color,
                           plot_bgcolor=background_color,font={'color':font_color})
neural_graph.update_traces(marker_line_width=0)

# Deploy

def deploy_transform(X_predict):
    from pickle import load

    def name_columns(transformer_names,transformer):
        columns_name = []
        for i in transformer.get_feature_names_out():
            for k in transformer_names:
                if k in i:
                    name = i.replace(k,'')
                    columns_name.append(name)
        return columns_name

    imputer = load(open(os.path.join(path_models,'imputer.pkl'),'rb'))
    X_predict = imputer.transform(X_predict)
    columns_name = name_columns(['numerical__','categorical__','remainder__'],imputer)
    X_predict = pd.DataFrame(X_predict,columns=columns_name)          

    # Remove Outliers

    outlier = load(open(os.path.join(path_models,'outlier.pkl'),'rb'))
    df_predict = pd.concat([X_predict,pd.DataFrame({'target':[0.255]})],axis=1)
    X_bool_outliers = outlier.predict(df_predict)
    mask = X_bool_outliers != -1
    X_predict = df_predict.drop(columns=['target']).iloc[mask,:]

    # Numerical Selection

    num = load(open(os.path.join(path_models,'num_feature.pkl'),'rb'))

    X_predict = num.transform(X_predict)

    columns_name = name_columns(['num_feature__','remainder__'],num)
    X_predict = pd.DataFrame(X_predict,columns=columns_name)
    
    return X_predict

columns = ['1','2','3','4','5','6','7']
input_table = dash_table.DataTable(id='table-editing-simple',
        columns = [{"name": i, "id": i, "deletable": False, "selectable": False, "hideable": False} for i in columns], 
        data = [{'1':0.12, '2':0.255, '3':0.255, '4':0.255, '5':0.255, '6':0.255, '7':0.255}],
        editable=True,            
        style_data={                
            'whiteSpace': 'normal',
            'height': 'auto',
            'color': 'red',
            'backgroundColor': "#0f2537"
        },
        style_header={
            'backgroundColor': '#07121B',
             'color': '#fff',
          },
        )

# <----------------------------- Dash Layout --------------------------------> #

app = Dash(__name__,external_stylesheets=[dbc.themes.COSMO],title='Phosphate',
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, \
                             initial-scale=1.0'}]) # SOLAR, LUX

server = app.server

draw_figure_buttons = {'modeBarButtonsToAdd':['drawline',
                                        'drawopenpath',
                                        'drawclosedpath',
                                        'drawcircle',
                                        'drawrect',
                                        'eraseshape'
                                       ]}

app.layout = dbc.Container([ 

    dbc.Row(dbc.Col([html.H1('Phosphate DataFrame')],width=6,className="title")),
    dbc.Row(dbc.Col([html.H4('by Exlonk Gil')],width=12)),

    # <------------------------------DATA INSIGHTS --------------------------> #

    dbc.Row(dbc.Col([html.H2('Raw Data Insights')],width=12,    
    style={"padding-top":"1rem","padding-bottom":"1rem","textAlign":"center"})),

    dbc.Row(dbc.Col([html.P("This dataset has data of the amount of concentration \
        of phosphate ions (polyphosphates) in river water. There are 8 consecutive \
        stations of the state water monitoring system. It's should predict the value \
        in the eighth station by the first seven stations. The numbering of stations \
        in the dataset is done from the target station upstream, i.e. closest to it - first, \
        upstream - second, etc.")])),

    dbc.Row(dbc.Col([html.P("Data are average monthly. The number of observations \
        on stations is different (from 4 to about 20 years). Concentration of phosphate \
        ions (polyphosphates) (PxOy) is measured in mg/cub. dm (ie milligrams in \
        the cubic decimeter).")])),

    dbc.Row(dbc.Col(html.Br(),width=12)), 

    dbc.Row(dbc.Col([table],width=12)),

    dbc.Row(dbc.Col(html.Br(),width=12)), 

    dbc.Row(dbc.Col([html.P("The shape figure illustrates the size of the data set, \
        this data set in specific is tiny but relevant. The data type graph reveal, \
        like as expected, that all the data is numerical.")])),   

    dbc.Row(dbc.Col(html.Br(),width=12)),

    dbc.Row([dbc.Col([dcc.Graph(figure=shape)],width=5),dbc.Col
           ([dcc.Graph(figure=info_fig)],width=7)]),

    dbc.Row(dbc.Col([html.H2('Data Cleaning')],width=12,
    className="title",style={"textAlign": "center"})),
    
    # Duplicates and missing values

    dbc.Row(dbc.Col([html.H3('Duplicates and Missing Values')],width=12,
                            className="subtitle",style={"textAlign": "left"})),

    dbc.Row(dbc.Col([html.P('This section shows if there is some duplicated\
                              rows in the dataframe ' 
                              ' and the number of missing data per feature')])),

    dbc.Row([dbc.Col([dcc.Graph(figure=duplicates)],width=3),\
             dbc.Col([dcc.Graph(figure=missing_values)],width=9)]),
    
    dbc.Row(dbc.Col([html.P('The figures indicate that there is no duplicate data, \
        but there are plenty of missing values, from station three to seven most  \
        of the data is missing. Due to this, they were dropped from the set. \
        This decision will be supported later.')])),

    # Histograms

    dbc.Row(dbc.Col([html.H3('Histograms')],width=12,
                            className="subtitle",style={"textAlign": "left"})),

    dbc.Row(dbc.Col([html.P('The distribution type on the data is important \
        for multiple machine learning models. The following histograms show how \
        this distribution looks for this set.')])),

    dbc.Row([dbc.Col([dcc.Graph(figure=histograms[0])],width=4),\
             dbc.Col([dcc.Graph(figure=histograms[1])],width=4),\
             dbc.Col([dcc.Graph(figure=histograms[2])],width=4)]),
             
    dbc.Row(dbc.Col([html.P('It is evident from the three histograms that neither \
        has a normal distribution: all of them are biased, besides the Shapiro and \
        Kolmogorov index are below 0.05.')])),

    # <---------------------------DATA PREPARATION --------------------------> #

    dbc.Row(dbc.Col([html.H3('Outlier Identification')],width=12,
                            className="subtitle", style={"textAlign": "left"})),

    dbc.Row(dbc.Col([html.P("Outlier detection aims to remove the points that are \
    truly outliers, in order to build a model that performs well on unseen test \
    data and cleaning the data.")])),

    dbc.Row([dbc.Col([dcc.Graph(figure=box_plot_pure,config=draw_figure_buttons)],width=4),
    dbc.Col([dcc.Graph(figure=box_plot_lof,config=draw_figure_buttons)],width=4),
    dbc.Col([dcc.Graph(figure=box_plot_ifo,config=draw_figure_buttons)],width=4)]),

    dbc.Row(dbc.Col([html.P('All outlier models label one particular point (at 2.39) \
    as an outlier. Applying any outlier models gives a box plot with all inner points, \
    though, the data is still biased. Based on the values and amount of outlier data, \
    it appears that the outliers are an important characteristic for this data set.')])),

    # <------------------------- FEATURE SELECTION --------------------------> #

    dbc.Row(dbc.Col([html.H2('Feature Selection',style={"textAlign": "center"})])),

    dbc.Row(dbc.Col([html.H3('Numerical Correlation')])),

    dbc.Row(dbc.Col([html.P('The Pearson correlation allows to find some linear \
    correlation between the feature and the target. The outlier models were tested \
    against this correlation method.')])),

    dbc.Row([dbc.Col([dcc.Graph(figure=pearson_correlation_fig)],width=6),\
             dbc.Col([dcc.Graph(figure=pearson_correlation_fig_std)],width=6)]),
    
    dbc.Row([dbc.Col([dcc.Graph(figure=pearson_correlation_fig_lof)],width=6),\
             dbc.Col([dcc.Graph(figure=pearson_correlation_fig_ifo)],width=6)]),


    dbc.Row(dbc.Col([html.P('With the LOF model, the correlation is 0.83 and \
    0.768 for station one and two, respectively, whereas without it, it is 0.699 \
    and 0.807 for each one. The next graphs are the spearman correlation, which \
    is used to find non-linear relationships between numerical features and the \
    target.')])),
    
    dbc.Row([dbc.Col([dcc.Graph(figure=spearman_correlation_fig)],width=6),\
             dbc.Col([dcc.Graph(figure=spearman_correlation_fig_std)],width=6)]),
    
    dbc.Row([dbc.Col([dcc.Graph(figure=spearman_correlation_fig_lof)],width=6),\
             dbc.Col([dcc.Graph(figure=spearman_correlation_fig_ifo)],width=6)]),
    
    dbc.Row(dbc.Col([html.P('Since the values are the same that for the pearson \
    correlation, it appears that there is no non-linear correlation between the \
    target and the features.')])),

    dbc.Row(dbc.Col([html.P('The mutual information is another correlation method \
    used to find some correlation between the features. It is equal to zero if and \
    only if two random variables are independent. Higher values mean higher dependency. \
    The set without outliers given by the local outlier factor algorithm is used \
    in this case.')])),

    dbc.Row([dbc.Col([],width=3),dbc.Col([dcc.Graph(figure=mutual_correlation_fig)],width=6),
            dbc.Col([],width=3)]),

    dbc.Row(dbc.Col([html.P('The first station is more correlated with the target, \
    it was what was expected, since the distance from the station one with the \
    target station is less.')])),

    dbc.Row(dbc.Col([html.H3('Feature Selection Algorithm')])),

    dbc.Row(dbc.Col([html.P('An algorithm was made to find the best feature \
    selection pipeline on all the data. The equipment for doing this search is very limited, \
    so the algorithm uses a simple linear regression model as the estimator and \
    the following transformers:')])),

    dbc.Row(dbc.Col([html.P('Numerical imputers: SimpleImputer(), KNNImputer(), \
    IterativeImputer(random_state=0)')])),

    dbc.Row(dbc.Col([html.P('Categorical imputer: SimpleImputer()')])),

    dbc.Row(dbc.Col([html.P('Encoder: HotEncoder()')])),

    dbc.Row(dbc.Col([html.P('Scaler: RobustScaler(), MinMaxScaler(), \
    MaxAbsScaler(), StandardScaler()')])),

    dbc.Row(dbc.Col([html.P('Outlier: LOF and ISO')])),

    dbc.Row(dbc.Col([html.P('Feature selection and reduction of dimensionality: \
    LinearDiscriminantAnalysis(), PCA(), \
    SelectKBest(score_func = r_regression), SelectKBest(score_func = chi2) ,\
    SelectKBest(score_func = mutual_info_classif)')])),

    dbc.Row(dbc.Col([html.P('Distribution: QuantileTransformer(output_distribution="normal"), \
    PowerTransformer()')])),

    dbc.Row(dbc.Col([html.P('The algorithm searches for the best combination of \
    these transformers at different stages for this data set.')])),

    dbc.Row(dbc.Col([table_feature],width=12)),

    dbc.Row(dbc.Col(html.Br(),width=12)),

    dbc.Row(dbc.Col([html.P('The best combination of transformers was chosen \
    between 7920 combinations and three repetitions of these combinations \
    (it was done because of the stochastic nature of the ML algorithms)')])),

    dbc.Row(dbc.Col([html.P("In conclusion, the pipeline that was chosen for this \
    project's preprocessing step is: KNNImputer (Its selection was done with a similar \
    process that the previous one) for missing data, not making a change in the distribution \
    nature of the data nor scaling, use the local outlier factor for detecting the outlier \
    data (the n_neighbors is change to 10 for improving the underfitting in the cv set), \
    and finally a feature statistical selection, using the pearson correlation.")])),

    dbc.Row(dbc.Col([html.P("It can be noted that between all the columns, only \
    one was chosen by this algorithm, which supports the initial assumption that\
    only take two features to make the predictions.")])),

    # <---------------------- MACHINE LEARNING ------------------------------> #

    dbc.Row(dbc.Col([html.H2('Machine Learning Models',style={"textAlign": "center"})])),

    dbc.Row(dbc.Col([html.H3('Baseline Model With Outliers')])),

    dbc.Row(dbc.Col([html.P('A baseline model is a simple model that gives us answers \
        we can compare against. Complex models should perform better than the baseline model. \
        The simplest model for a regression problem is to take the mean or median value of \
        the target and use it as a constant for any instance.')])),


    dbc.Row(dbc.Col([html.P('The following graphs illustrate the accuracy of the model in \
        the training and validation set, the "accuracy graph" is a figure with adjusted \
        curves made applying a "lowess" adjustment, meanwhile the "Individual error" \
        is a graph that shows the absolute error for each target data, whose size and \
        color is an indicator of error size.')])),

    dbc.Row([dbc.Col([dcc.Graph(figure=baseline_accuracy_train)],width=6),\
             dbc.Col([dcc.Graph(figure=baseline_ind_train)],width=6)]),

    dbc.Row([dbc.Col([dcc.Graph(figure=baseline_accuracy_cv)],width=6),\
             dbc.Col([dcc.Graph(figure=baseline_ind_cv)],width=6)]),

    dbc.Row([dbc.Col([dcc.Graph(figure=baseline_accuracy_test)],width=6),\
             dbc.Col([dcc.Graph(figure=baseline_ind_test)],width=6)]),

    dbc.Row([dbc.Col([dcc.Graph(figure=baseline_bias_variance)],width=12)]),

    dbc.Row(dbc.Col([html.P('As expected, there is very little adjustment of this \
        approximation to the data. The R-squared is zero, and the max mean absolute error \
        is thirty, which is 15% of the max value in the target set, which is very high value.')])),

    dbc.Row(dbc.Col([html.H3('Simple Linear Model With Outliers')])),

    dbc.Row(dbc.Col([html.P('The following model is used to assess the impact \
    of outliers on the models performance.')])),

    dbc.Row([dbc.Col([dcc.Graph(figure=linear_accuracy_train)],width=6),\
             dbc.Col([dcc.Graph(figure=linear_ind_train)],width=6)]),

    dbc.Row([dbc.Col([dcc.Graph(figure=linear_accuracy_cv)],width=6),\
             dbc.Col([dcc.Graph(figure=linear_ind_cv)],width=6)]),

    dbc.Row([dbc.Col([dcc.Graph(figure=linear_accuracy_test)],width=6),\
             dbc.Col([dcc.Graph(figure=linear_ind_test)],width=6)]),

    dbc.Row([dbc.Col([dcc.Graph(figure=linear_bias_variance)],width=12)]),

    dbc.Row(dbc.Col([html.P('In comparison with the baseline model the MAE value \
    is half on the train set, 2/3 on the validation set, and a third part \
    on the test, and it shows an acceptable R squared.')])),

    dbc.Row(dbc.Col([html.H3('Polynomial Model')])),

    dbc.Row(dbc.Col([html.P('In this case the preprocessed set is used, multiple \
    polynomial degrees were tested with the simplest linear regression model.')])),

    dbc.Row([dbc.Col([dcc.Graph(figure=mse_train_cv_poly_graph)],width=6),\
             dbc.Col([dcc.Graph(figure=r2_train_cv_poly_graph)],width=6)]),

    dbc.Row(dbc.Col([html.P('The best degree (with the best train/cv gap) for \
    the model is two (though the difference with degree one is very little), it \
    has mean accuracy, higher degrees improve the fitting in the training set, \
    but down the fitting in the validation set. More detailed graphs are shown:')])),

    dbc.Row([dbc.Col([dcc.Graph(figure=poly_accuracy_train)],width=6),\
             dbc.Col([dcc.Graph(figure=poly_ind_train)],width=6)]),

    dbc.Row([dbc.Col([dcc.Graph(figure=poly_accuracy_cv)],width=6),\
             dbc.Col([dcc.Graph(figure=poly_ind_cv)],width=6)]),

    dbc.Row([dbc.Col([dcc.Graph(figure=poly_accuracy_test)],width=6),\
             dbc.Col([dcc.Graph(figure=poly_ind_test)],width=6)]),

    dbc.Row([dbc.Col([dcc.Graph(figure=poly_bias_variance)],width=12)]),

    dbc.Row(dbc.Col([html.P("It can be seen that in this scenario, it is critical to identify outliers. \
    The accuracy in the training and the validation set improved (in the validation \
    set pass from 0.16 to 0.89), though in the test set down a little.\
    As a note, use Ridge model doesn't improve the accuracy.")])),

    dbc.Row(dbc.Col([html.H4("Learning Curve")])),

    dbc.Row(dbc.Col([html.P('This figure illustrates if the accuracy of the data \
    can be improved adding more data to the model')])),

    dbc.Row(dbc.Col([dcc.Graph(figure=learning_curve)],width=12)),

    dbc.Row(dbc.Col([html.P('This figure illustrates if the accuracy of the data \
    can be improved adding more data to the model')])),

    dbc.Row(dbc.Col([html.P("The graph can be divided into three sections from \
    left to right. In the first section (0 , 40) the model improve its accuracy \
    on the cv set at the expense of decreased it on the training set, just like \
    is seen in general on learning curves. The second section (40, 50) which is \
    a flat section, where the train/cv gap remains 'constant' and a third section \
    (50, 80) where de train/cv grap increase. From the graph's shape, it can be said \
    that the model doesn't get better by adding more instances because it enters the \
    flat zone and expanses from it.")])),

    dbc.Row(dbc.Col([html.H3("KNeighbor Model")])),

    dbc.Row(dbc.Col([html.H4("Validation Curves")])),

    dbc.Row(dbc.Col([html.P('The validation curves graphs illustrate the impact \
    of one hyperparameter or another on the accuracy metric. For the kneighbord model \
    four hyperparameters were used: n neighbors, weights, algorithm, and leaf size')])),

    dbc.Row([dbc.Col([dcc.Graph(figure=validation_curve_1)],width=6),\
             dbc.Col([dcc.Graph(figure=validation_curve_2)],width=6)]),

    dbc.Row([dbc.Col([dcc.Graph(figure=validation_curve_3)],width=6),\
             dbc.Col([dcc.Graph(figure=validation_curve_4)],width=6)]),

    dbc.Row(dbc.Col([html.P("It appears that the 'n neighbors' hyperparameter \
    is the parameter with the highest impact on the accuracy and the train/cv \
    gap, this result can be used for narrowed the hyperparameters search values, \
    though in this case, due to the size the data and the result itself, \
    it wasn't used.")])),

    dbc.Row([dbc.Col([dcc.Graph(figure=validation_curve_1)],width=6),\
             dbc.Col([dcc.Graph(figure=validation_curve_2)],width=6)]),

    dbc.Row([dbc.Col([dcc.Graph(figure=validation_curve_3)],width=6),\
             dbc.Col([dcc.Graph(figure=validation_curve_4)],width=6)]),

    dbc.Row(dbc.Col([html.P('A grid search was used to choose the best parameters.')])),

    dbc.Row([dbc.Col([dcc.Graph(figure=knn_accuracy_train)],width=6),\
             dbc.Col([dcc.Graph(figure=knn_ind_train)],width=6)]),

    dbc.Row([dbc.Col([dcc.Graph(figure=knn_accuracy_cv)],width=6),\
             dbc.Col([dcc.Graph(figure=knn_ind_cv)],width=6)]),

    dbc.Row([dbc.Col([dcc.Graph(figure=knn_accuracy_test)],width=6),\
             dbc.Col([dcc.Graph(figure=knn_ind_test)],width=6)]),

    dbc.Row([dbc.Col([dcc.Graph(figure=knn_bias_variance)],width=12)]),

    dbc.Row(dbc.Col([html.P('The performance of this model is similar \
    to the one found normally in literature, has a relative good fit \
    on the training set, and decreased on the cv and test set. As well \
    as for the polynomial model, the number of points with an error higher \
    tan 0.29 is three on the test set, which is a ten percent of the data.')])),

    dbc.Row(dbc.Col([html.H3('Neural Network')],width=12,
                            className="subtitle",style={"textAlign": "left"})),

    dbc.Row(dbc.Col([html.P('The next graph is a representation of the neural \
    network model used, it shows the number and relative size of the layers \
    as well as the relative number of connections')])),

    dbc.Row(dbc.Col([dcc.Graph(figure=neural_graph)],width=12)),

    # Model Evaluation

    dbc.Row([dbc.Col([dcc.Graph(figure=network_accuracy_train)],width=6),\
             dbc.Col([dcc.Graph(figure=network_ind_train)],width=6)]),

    dbc.Row([dbc.Col([dcc.Graph(figure=network_accuracy_cv)],width=6),\
             dbc.Col([dcc.Graph(figure=network_ind_cv)],width=6)]),

    dbc.Row([dbc.Col([dcc.Graph(figure=network_accuracy_test)],width=6),\
             dbc.Col([dcc.Graph(figure=network_ind_test)],width=6)]),

    dbc.Row([dbc.Col([dcc.Graph(figure=network_bias_variance)],width=12)]),

    dbc.Row(dbc.Col([html.P('It is well known that, for small data sets, \
    the use of a neural network does not improve significantly the accuracy, \
    if it does. This premise is corroborated in this case.')])),

    dbc.Row(dbc.Col([html.P("Multiple architectures have been proven with and \
    without regularization, in this model the relu function is used, and sixteen \
    layers with a range of neurons between fifty and seven hundred. The regularization \
    decreased the accuracy in the test set, a higher number of layer didn't improve \
    the model neither higher number of neurons. The bias and variance graph is \
    very similar to the polynomial one, it has sens by the fact that the neural \
    network is based in linear functions. It doesn't show any improvement compared \
    with the other models.")])),

    dbc.Row(dbc.Col([html.P('In conclusion, it is essential to preprocess \
    the data set against outlier values. A better performance can be \
    expected if the outlier values detector is refined. The simplest \
    linear model has good performance in this problem. The learning \
    curve indicates that it appears not to be needed to get more data \
    to address this problem.')])),

    dbc.Row(dbc.Col([html.H3('Deploy')],width=12,
                            className="subtitle",style={"textAlign": "left"})),

    dbc.Row(dbc.Col([html.P('You can try your own data inputting them in the next table and pressing Enter.')])),
    
    dbc.Row(dbc.Col([input_table],width=12)),
    dbc.Row(dbc.Col([ html.Progress(id="progress_bar", value="0")],width=2)),
    dbc.Row([dbc.Col([],width=3),dbc.Col(id='prediction',children=[],width=6),
            dbc.Col([],width=3)])
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                # Cierre del layout
    ],className="container")


@dash.callback(
    Output('prediction','children'),
    Input('table-editing-simple', 'data'),
    background=True,
    prevent_initial_call=False,
    manager = background_callback_manager,
     running=[
        (
            Output("progress_bar", "style"),
            {"visibility": "visible"},
            {"visibility": "hidden"},
        )
     ],
     progress=[Output("progress_bar", "value"), Output("progress_bar", "max")],
    )

def prepare_data(set_progress,data):
    from pickle import load
    import joblib
    from sklearn.preprocessing import PolynomialFeatures
    import os
    import time
    total = 5
    for i in range(total + 1):
        set_progress((str(i), str(total)))
        time.sleep(1)
    try:
        X_predict = data[0].copy()
        for k,v in X_predict.items():
            if k in ['1','2','3','4','5','6','7']:
                X_predict[k] = float(v)     
        X_predict = pd.DataFrame([X_predict]) 
        X_predict = deploy_transform(X_predict)
        poly_model = joblib.load(os.path.join(path_models,'LinearRegression_polynomial_regression_2.joblib'))
        poly_features = PolynomialFeatures(degree=2, include_bias=False)
        X_predict = poly_features.fit_transform(X_predict)
        y =  poly_model.predict(X_predict)
        figure = px.bar(x=["Polynomial Model"],title='Value Prediction',y=[float(y)],text=[float(y)],labels={'x':" ",'y':'Predicted Value mg/cub.dm'},color_discrete_sequence=['#24eca6'])
        figure.update_yaxes(range=(0,float(y)+0.15))
        figure.update_layout(paper_bgcolor=background_color,plot_bgcolor=background_color,font={'color':font_color})
        figure.update_traces(marker_line_width=0)
        figure.update_traces(texttemplate='%{text:.4}',textposition='outside',textfont_size=18)
        predict_graph = [dcc.Graph(figure=figure)]
        return predict_graph 
    except:
        predict_graph = [html.Br(),html.Div('The input data has an error or is taken by model like outlier data'),html.Br()]
        return predict_graph  

if __name__ == '__main__':
      app.run_server()
