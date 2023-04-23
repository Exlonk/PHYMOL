from matplotlib.figure import Figure
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import plotly.express as px
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import dash_table,callback,Dash, html, dcc,  Input, Output, no_update, ctx
import dash_bootstrap_components as dbc
import plotly.figure_factory as ff
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
import math
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
from dash import dash_table,callback,Dash, html, dcc,  Input
from dash import Output, no_update, ctx
from jupyter_dash import JupyterDash
import os

# <-------------------------- GENERAL PYTHON --------------------------------> #

def import_module(path):

    """
    Import a module at a specific location.
    """

    from importlib.machinery import SourceFileLoader
    ds = SourceFileLoader("add",path).load_module() 
    return ds
    
def fuc_arg(fuction):
    
    """ Return the arguments of a fuction """

    from inspect import signature
    
    return str(signature(fuction))

def df_to_numpy(X_train=None,X_cv=None,X_test=None,y_train=None,y_cv=None,
                y_test=None):

    """ Transform a pandas dataframe to a numpy array """

    try:
        X_train = X_train.to_numpy()
    except:
        pass
    try:
        X_cv = X_cv.to_numpy()
    except:
        pass
    try:
        X_test = X_test.to_numpy()
    except:
        pass
    try:
        y_train = y_train.to_numpy()
    except:
        pass
    try:
        y_cv = y_cv.to_numpy()
    except:
        pass
    try:
        y_test = y_test.to_numpy()
    except:
        pass
    
    return X_train,X_cv,X_test,y_train,y_cv,y_test

def save_plot_json(path,name,figure):

    """ Save a plotly plot as json"""

    import plotly
    import os

    with open(os.path.join(path ,name+'.json'),'w',encoding="utf-8") as f:
        f.write(plotly.io.to_json(figure))

def load_plot_json(path,name):

    """ Load a json plot in using plotly """

    import plotly
    import json 
    import os
    with open(os.path.join(path,name+'.json'),'r+',encoding="utf-8") as f:
        read_ = f.read()
    figure = plotly.io.from_json(read_)
    return figure

def sampling(df,save=False):   

    """
    Made a sample from data
    df must be a pandas dataframe and
    the column 'target' called 'target'
    """ 

    grouped = df.groupby(df['target'])

    # Define the size of the sample for each subgroup
    sample_size = int(len(df) * 0.1) # 10% of the data

    # Initialize an empty list to store the sample
    sample = []

    # For each subgroup, take a sample of the desired size
    for name, group in grouped:
        sample.append(group.sample(sample_size))

    # Concatenate all the subgroups into one sample
    sample = pd.concat(sample)

    # Shuffle the sample randomly
    sample = sample.sample(frac=1).reset_index(drop=True)

    # Save the sample to a new CSV file
    if save == True:
        sample.to_csv("representative_sample.csv", index=False)
    return sample

# <------------------------- DATA PREPARATION -------------------------------> #

# Data Raw Insights #

def table(df,bgcolor="#fff",textcolor="#fff",bgheader='#fff',columns_description=[None]):
    
    """ Fuction to do a interactive table in plotly"""

    from dash import dash_table
    table = dash_table.DataTable( 
        columns = [{"name": i, "id": i, "deletable": False, "selectable": False, "hideable": False} for i in df.columns], 
        data = df.round(decimals=4).to_dict('records'),
        editable=True,              # allow editing of data inside all cells
        filter_action="native",     # allow filtering of data by user ('native') or not ('none')
        sort_action="native",       # enables data to be sorted per-column by user or not ('none')
        sort_mode="single",         # sort across 'multi' or 'single' columns
        column_selectable="multi",  # allow users to select 'multi' or 'single' columns
        row_selectable=False,     # allow users to select 'multi' or 'single' rows
        row_deletable=False,         # choose if user can delete a row (True) or not (False)
        selected_columns=[],       # ids of columns that user selects
        selected_rows=[],           # indices of rows that user selects
        page_action="none",       # all data is passed to the table up-front or not ('none')
        page_current=0,             # page number that user is on
        page_size=7,                # number of rows visible per page
        style_cell={                # ensure adequate header width when text is shorter than cell's text
            'minWidth': '220px', 'width': '220px', 'maxWidth': '223px', "whiteSpace": "pre-line"
        }, # "whiteSpace": "pre-line": permite dividir la información de las celdas en más de una columna
        style_data={                # overflow cells' content into multiple lines
            'whiteSpace': 'normal',
            'height': 'auto',
            'color': textcolor,
            'backgroundColor': bgcolor
        },
        style_table={'overflow':'scroll','height':'20rem'},
        style_header={
            'backgroundColor': bgheader,
             'color': textcolor,
          },
        style_filter={
            'backgroundColor': bgheader,
             'color': textcolor,
          },
        fixed_rows={'headers': True, 'data': 0},
        css=[{"selector": ".show-hide", "rule": "display: none; "},
        {"selector":".dash-table-tooltip","rule":"background-color:{0}; color:{1};".format(bgheader,textcolor)}],
        tooltip_header={i: k for i,k in zip(df.columns,columns_description)},
        tooltip_delay=0,
        tooltip_duration=None
        #style_cell_conditional=[    # align text columns to left. By default they are aligned to right
        #    {
        #        'if': {'column_id': c},
        #        'textAlign': 'left'
        #    } for c in X[2:].columns
        #],
        )
    return table

def info(x):

    """ Return a dataframe with basic info in pandas format """ 

    info = pd.DataFrame({"name": x.columns, "non-nulls": \
    len(x)-x.isnull().sum().values, "nulls": x.isnull().sum().values, "type": x.dtypes.values.astype(str)})
    return info

def shape_figure(x,color1,color2):

    """ Return a shape plot with the rows and the columns of the dataframe """

    shape = make_subplots(rows=1, cols=2, shared_yaxes = False)
    shape.add_trace(
        go.Bar(y=[x.shape[0]],x = ["Rows"],text=str(x.shape[0]),marker_color=color1),
        row=1, col=1
    )

    shape.add_trace(
        go.Bar(y=[x.shape[1]], x = ["Columns"], text=str(x.shape[1]),marker_color=color2),
        row=1, col=2
    )

    shape.update_layout(title_text="Shape <br><sup>Without target</sup>",showlegend=False) # height=600, width=800, 
    shape.update_traces(marker_line_width=0,textfont_size=17, textangle=0, textposition="inside", cliponaxis=False)
    shape.update_yaxes(visible=False)
    shape.update_traces(hoverinfo='y')

    return shape

def datatype_graph_for_small_data(df,color):

    """ Return a plot for small data sets with the data types """

    infor = info(df)
    type_fig=go.Figure()
    type_fig.add_trace(go.Pie(labels=infor['name'], values=infor['non-nulls'],\
        text=infor["type"], textinfo='text+value', sort=False, direction='clockwise', marker_colors=color))
    type_fig.update_traces(marker_line_width=0,hoverinfo='label')
    type_fig.update_layout(title_text="Data Type <br><sup>Without target</sup>")
    return type_fig

def datatype_graph_for_large_data(df,color):

    """ Return a plot for large data sets with the data types """

    infor = info(df)
    type_fig=go.Figure()
    type_fig.add_trace(go.Pie(labels=infor['type'],\
        text=infor["type"], textinfo='text+value', sort=False, direction='clockwise',marker_colors=color))
    type_fig.update_traces(hoverinfo='label')
    type_fig.update_layout(title_text="Data Type <br><sup>Without target</sup>",showlegend=False)
    return type_fig

def histograms_(df,color=None,categorical_data = [None]):

    """ Return the histograms to look for normality """

    from scipy.stats import shapiro 
    from scipy.stats import kstest
    
    unique = pd.DataFrame(df.nunique()).transpose()
    container = []
    for i,k in zip(df.columns,range(0,df.shape[1])):
        if unique[i][0] <= 10:
            nbins= int(unique[i][0])
        else:
            nbins = 10
        figure = px.histogram(x=df[i],title=str(i).capitalize()+' Histogram',labels={'x':str(i).capitalize()},nbins=nbins)
        
        if color != None:
            figure.update_traces(marker_color=color[k])

        figure.update_layout(yaxis_title="Frequency") 

        if i not in categorical_data:
            
            figure.add_annotation(xref = 'x domain', yref = 'y domain',
                 text=f"Shapiro-Wilk p:  {round(shapiro(df[i])[1],5)} <br> Kol-Smir p:  {round(kstest(df[i],'norm')[1],5)}", 
                 showarrow=False,font=dict(family="arial",
                  size=12,
                 color="black"))
                # p values for both > 0.05 -> Normally distributed  
        container.append(figure)

    return container

def histogram_density(df,column,bins=30):

    """Generates a histogram of column "column" with a number of 
    divisions equal to bins """

    hist_data = [df[column]]
    group_labels = [column] # name of the dataset
    fig = ff.create_distplot(hist_data,group_labels,bin_size=(df[column].max()-df[column].min())/bins)
    return fig
    
# <------------------------ Data Cleaning ---------------------------> #

def unique_values(df, header=1, treshold=100,title='Number of Categories for each Categorical Feature',colors1=None,colors2=None):
    
    """ The bar graph shows us the number of different elements in each column.
        The treemap which values are repeated and in what quantity

    Args:
        df: Dataframe
        header (int, None): Si el archivo no posee cabeceras de 
        columna poner None. Defaults to 1.
        porcentaje: Porcentaje (menor o igual) que desea imprimirse
    """
    
    column, categories, values, count = [], [], [], []
    for i in df:
        value = df.nunique()[i]
        if value <= treshold:
            column += [i]*df.nunique()[i]
            categories += list(df[i].unique())
            values.append(value)
            a = dict(df.pivot_table(columns=[i], aggfunc='size'))
            for k in list(df[i].unique()):
                count.append(a[k])

    fig_1 =make_subplots(rows=1, cols=1, shared_yaxes = False)

    fig_1.add_trace(
        go.Bar(y=values,x = list(dict.fromkeys(column)),text=[str(i)+' Categories' for i in values],marker_color=colors1),row=1, col=1
        )

    fig_1.update_layout(title_text=title+"<br><sup>Without target</sup>",showlegend=False) # height=600, width=800, 
    fig_1.update_traces(marker_line_width=0,textfont_size=17, textangle=0, textposition="inside", cliponaxis=False)
    fig_1.update_yaxes(visible=False)
    fig_1.update_traces(hoverinfo='y')
    
    fig_2 = px.treemap(path=[column,categories], values=count,color_discrete_sequence=colors2)
    fig_2.update_layout(title_text="Categories Insights <br><sup>Without target</sup>",showlegend=False) # height=600, width=800, 

    """
    # Código DASH para esta función que es para categorias #

    dbc.Row(dbc.Col([html.H3('Unique Values (different)')],
    width=12,className="subtitle",style={"textAlign": "left"})),

    # En este poner df.shape del dataframe final
    dbc.Row(dbc.Col([dcc.Slider(id='slider', min=1, max=df.shape[0], step=1,
         value=int(df.nunique().min()),marks={x: str(x) for x in 
                                  range(1,int(df.shape[0]),int(df.shape[0]/4))},
         tooltip={"placement": "bottom", "always_visible": True})])),

    dbc.Row([dbc.Col([dcc.Graph(id='unique_values_1')],width=5),dbc.Col
                                  ([dcc.Graph(id='unique_values_2')],width=7)]),

    @app.callback(Output(component_id='unique_values_1', component_property='figure'),
              Output(component_id='unique_values_2', component_property='figure'),
              Input(component_id="slider", component_property="value"))

    def unique(value): 
        fig_1_unique_values, fig_2_unique_values= ds.unique_values(df,treshold=value)
        return  fig_1_unique_values, fig_2_unique_values
    """
    return fig_1 , fig_2

# Missing and Duplicate Data

def duplicates_missing(X,drop=False,colors1=None,colors2=None):

    """ Return a plot with the missing data and the duplicates """

    duplicates = X.shape[0]- X.drop_duplicates().shape[0]
    figure_1 = px.bar(x=['All Dataframe'],y=[duplicates],text=[duplicates],
                        color_discrete_sequence = colors1, labels=dict([('x',''),('y','Number of duplicates')]))
    figure_1.update_traces(texttemplate='%{text:.2s}',textposition='outside')
    missing = X.isna().sum()
    figure_1.update_layout(title_text="Number of Duplicates <br><sup>With target</sup>")
    figure_2 = px.bar(x=missing.index,y=missing,range_y=[0,missing.max()],text=missing
    ,labels=dict([('x',''),('y','Number of missing values')]),color_discrete_sequence = colors2)
    figure_2.update_traces(marker_line_width=0,texttemplate='%{text:.2s}',textposition='outside')
    figure_2.update_layout(title_text="Missing Values <br><sup>With target</sup>")
    return figure_1, figure_2

def missing_knn(df,**arg):
    
    """
    KNN model to replace missing values (NaN) in a dataframe
   
    df (pandas.DataFrame): Dataframe type pandas

    **arg (arguments): Arguments of the KNNImputer estimates
    """
    from sklearn.impute import KNNImputer
    # Parámetros principales = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')
    imputer = KNNImputer(**arg).fit_transform(df.to_numpy())
    imputer = pd.DataFrame(imputer)
    fig1, fig2 = duplicates_missing(imputer)
    return imputer, fig2 
    
def missing_iterative(df,**arg):

    """ Iterative model to replace missing values (NaN) in a dataframe """

    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    # Parámetros IterativeImputer(estimator=BayesianRidge(), 
    #             n_nearest_features=None, imputation_order='ascending') 
    
    imputer = IterativeImputer(**arg).fit_transform(df.to_numpy())
    imputer = pd.DataFrame(imputer)
    fig1, fig2 = duplicates_missing(imputer)
    return imputer, fig2 

# Outlier detection

def box_plot(df,color=None):

    """ Box plot of all numerical columns """

    box_plot = px.box(df, points="all",labels={'variable':'','value':''},color_discrete_sequence=color )
    #  box_plot.update_xaxes(range=[-1,9.5]) Modifica el ancho por defecto de los datos que se visualizan
    box_plot.update_layout(title_text="Pure Box Plot <br><sup>With target</sup>")
    box_plot.update_traces(hovertemplate='value:%{y}')
    # box_plot.update_layout(width=1000)
    return box_plot

def outliers_std(X,y, RIC_number=1.5, column_names = None, all=True, print_outliers=False):

    """
    Remove outliers using quantiles

    X : Dataframe, si se desea un conjunto en especifico de columnas se debe especificar
    y : Target
    RIC_number : Número de veces por el rango intercuartil
    all: True or False, si es false se debe especificar las columnas a las cuales se le hara la transformación
    """

    if all == True:
        data = pd.concat([X,y],axis=1)
        df = pd.concat([X,y],axis=1)
        df_ = pd.concat([X,y],axis=1)  
        for column in df.columns:
            Q1= np.quantile(data[column],0.25)
            Q3= np.quantile(data[column],0.75)
            IQR = Q3-Q1
            Low = Q1 - RIC_number*(IQR)
            High = Q3 + RIC_number*(IQR)
            df = df[df[column].between(Low, High)]
        X = df.iloc[:,:-1]
        y = df.iloc[:,-1:]
    else:
        data = pd.concat([X,y],axis=1)
        df = pd.concat([X,y],axis=1)
        df_ = pd.concat([X,y],axis=1) 
        df = df[column_names]
        df_ = df_[column_names]
        for column in df.columns:
            Q1= np.quantile(data[column],0.25)
            Q3= np.quantile(data[column],0.75)
            IQR = Q3-Q1
            Low = Q1 - RIC_number*(IQR)
            High = Q3 + RIC_number*(IQR)
            df = df[df[column].between(Low, High)]
        X = df.iloc[:,:-1]
        y = df.iloc[:,-1:]
    if print_outliers:
        print('Number of outlier data: ',np.array(df_.shape[0]) - np.array(df.shape[0]))
    return X,y

def outliers_models(df, model_name, numerical_data=None,color=None, all_dataframe = True,**arg_model):

    """ Detect the outliers of the dataset using LOF or the IFO model
      
        df (pandas.dataframe): Dataframe que incluye todos dos datos inclusive
                               el target, y la misma debe ser identficada como
                               'target'
      
        model_name: 'LOF' or 'IFO'

        all_dataframe: Si se quiere analizar todo el dataframe y tener en cuenta el target para determinar
                       outliers se deja True, si no se pone False, y la variable 
                       numerical_data no debe tener la palabra 'target'.
        
        numerical_data: Es una lista que contiene los predictores númericos a 
                        analizar
                        
        **arg_model = Es un diccionario cons los parametros que utilizara el modelo """

    if model_name == "LOF":
        if all_dataframe == True:
            lof = LocalOutlierFactor(**arg_model)
            df_bool_outliers = lof.fit_predict(df)
            mask = df_bool_outliers != -1
            df_pred = df.iloc[mask,:]
            outlier_data = df.iloc[mask==False,:]
            # Figure
            out = pd.concat([df,pd.DataFrame(df_bool_outliers)],axis=1)
            color_ = out[0].copy()
            color_ = color_.astype(str)
            color_[color_=='1']='non-outlier'
            color_[color_=='-1']='outlier'  
            box_plot = px.box(out.drop(columns=[0]), points="all",labels={'variable':'','value':''},
                              color=color_,color_discrete_sequence=color)
            # box_plot.update_xaxes(range=[-1,9.5])
            box_plot.update_layout(title_text="Box Plot Outliers by  LocalOutlierFactor <br><sup> With target</sup>",
                                  showlegend=False)
            box_plot.update_traces(hovertemplate='value:%{y}')
            return df_pred, outlier_data, box_plot
        else:
            lof = LocalOutlierFactor(**arg_model)
            X_bool_outliers = lof.fit_predict(df[numerical_data])
            mask = X_bool_outliers != -1
            X, y = df.drop(columns=['target']).iloc[mask,:], pd.DataFrame(df['target'][mask])
            outlier_data = df.iloc[mask==False,:]
            out = pd.concat([df.drop(columns=['target']),pd.DataFrame(X_bool_outliers)],axis=1)
            color_ = out[0].copy()
            color_ = color_.astype(str)
            color_[color_=='1']='non-outlier'
            color_[color_=='-1']='outlier'  
            box_plot = px.box(out[numerical_data], points="all",labels={'variable':'','value':''},color=color_,color_discrete_sequence=color)
            # box_plot.update_xaxes(range=[-1,9.5])
            box_plot.update_layout(title_text="Box Plot Outliers by  LocalOutlierFactor <br><sup> Without target</sup>",
                                showlegend=False)
            box_plot.update_traces(hovertemplate='value:%{y}')            
            return X,y, mask, box_plot

    if model_name == "IFO":
        if all_dataframe == True:
            ifo = IsolationForest(**arg_model)
            df_bool_outliers = ifo.fit_predict(df.to_numpy())
            mask = df_bool_outliers != -1
            df_pred = df.iloc[mask,:]
            outlier_data = df.iloc[mask==False,:]

            # Figure
            out = pd.concat([df,pd.DataFrame(df_bool_outliers)],axis=1)
            color_ = out[0].copy()
            color_ = color_.astype(str)
            color_[color_=='1']='non-outlier'
            color_[color_=='-1']='outlier'   
            box_plot = px.box(out.drop(columns=[0]), points="all",labels={'variable':'','value':''},color=color_,
                                color_discrete_sequence=color)
            # box_plot.update_xaxes(range=[-1,9.5])
            box_plot.update_layout(title_text="Box Plot Outliers by IsolationForest <br><sup> With target</sup>",
                                  showlegend=False)
            box_plot.update_traces(hovertemplate='value:%{y}')
            return df_pred, outlier_data, box_plot
        else:
            ifo = IsolationForest(**arg_model)
            X_bool_outliers = ifo.fit_predict(df[numerical_data])
            mask = X_bool_outliers != -1
            X, y = df.drop(columns=['target']).iloc[mask,:], pd.DataFrame(df['target'][mask])
            outlier_data = df.iloc[mask==False,:]
            out = pd.concat([df.drop(columns=['target']),pd.DataFrame(X_bool_outliers)],axis=1)
            color_ = out[0].copy()
            color_ = color_.astype(str)
            color_[color_=='1']='non-outlier'
            color_[color_=='-1']='outlier'  
            box_plot = px.box(out[numerical_data], points="all",labels={'variable':'','value':''},color=color_,
                                color_discrete_sequence=color)
            # box_plot.update_xaxes(range=[-1,9.5])
            box_plot.update_layout(title_text="Box Plot Outliers by IsolationForest <br><sup> Without target</sup>",
                                showlegend=False)
            box_plot.update_traces(hovertemplate='value:%{y}')
            return X,y, mask, box_plot

def delete_unique_value (df,header=1):

    """ Remove columns with a single value

    Args:
        ruta (str): Ruta del archivo a eliminar
        header (int or None): Si el conjunto de datos 
                               no lleva cabecera poner None
    """

    df = pd.read_csv(df, header=header)
    print("Initial value Row x Columns: ", df.shape)
    to_del = [df.columns[i] for i,v in enumerate(df.nunique()) if v == 1]
    df.drop(to_del, axis=1, inplace=True)
    print("Final value Row x Columns: ", df.shape)

class OneHotEncoder(SklearnOneHotEncoder):
    
    """ Class to convert categorical data to numeric while keeping the labels """

    from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder
    def __init__(self, **kwargs):
        super(OneHotEncoder, self).__init__(**kwargs)
        self.fit_flag = False

    def fit(self, X, **kwargs):
        out = super().fit(X)
        self.fit_flag = True
        return out

    def transform(self, X, **kwargs):
        sparse_matrix = super(OneHotEncoder, self).transform(X)
        new_columns = self.get_new_columns(X=X)
        d_out = pd.DataFrame(sparse_matrix.toarray(), columns=new_columns, index=X.index)
        return d_out

    def fit_transform(self, X, **kwargs):
        self.fit(X)
        return self.transform(X)

    def get_new_columns(self, X):
        new_columns = []
        for i, column in enumerate(X.columns):
            j = 0
            while j < len(self.categories_[i]):
                new_columns.append(f'{column}_{self.categories_[i][j]}')
                j += 1
        return new_columns
        
# <---------------------------- FEATURE SELECTION ---------------------------> #

def numerical_correlation(df=None,treshold=0.8,corr_meth='pearson',    
    X_train=None,y_train=None,X_cv=None, X_test=None,numerical_data=None,\
        categorical_data = None, return_df=False,k='all',estimator = None, color_heat=None,
        color_bar=None):

    """ Determines the correlation between variables for numerical data
    df (pandas.dataframe): Dataframe completo para person, kendall, spearman
    treshold: Porcentaje de correlación para pearson, kendall, spearman 
    X_tran, X_cv, X_test = pandas.dataframe
    numerical_data = list   
    return df = True si se quiere devolver X_train,X_cv 
    k = número de predictores que se mantienen """

    import pandas as pd
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import f_classif
    from sklearn.feature_selection import f_regression
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.feature_selection import mutual_info_regression

    if corr_meth in ['pearson','kendall','spearman']:
        """ Pearson, Spearman for regression output, kendall for categorical output """
        corr = df.corr()[(abs(df.corr(method=corr_meth,numeric_only=False)) > treshold) & (df.corr() < 1)]
    
        fig = go.Figure()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        fig.add_trace(
            go.Heatmap(
                x = corr.columns,
                y = corr.index,
                z = corr.mask(mask),
                texttemplate="%{z:.3f}",
                zmin=-1,
                zmax=1,
                colorscale=color_heat
            )
        )
        fig.update_layout(title_text=corr_meth.capitalize() +' Correlation',showlegend=False)
        return corr, fig
    
    elif corr_meth == 'anova':
        """ For categorical output """
        fs = SelectKBest(score_func=f_classif, k='all')
        fs.fit(df[numerical_data], df_['target'])
        df_fs = fs.transform(df[numerical_data])
        figure = px.bar(x=numerical_data,y=fs.scores_,title= 'Numerical Correlation Using Anova',
                        color=numerical_data,text=fs.scores_,labels={'x':'Features','y':'Anova Value'},color_discrete_sequence=color_bar)      
        figure.update_traces(marker_line_width=0,texttemplate='%{text:.2f}',textposition='outside',showlegend=False,hovertemplate='value:%{y}')

        if return_df == True:
            return figure, df_fs
        else:
            return figure
    
    elif corr_meth == 'f_regression':
        """ For categorical output """
        fs = SelectKBest(score_func=f_regression, k=k)
        fs.fit(X_train[numerical_data], y_train)
        X_train_fs = fs.transform(X_train[numerical_data])
        X_cv_fs = fs.transform(X_cv[numerical_data])
        X_test_fs = fs.transform(X_test[numerical_data])
        fs.scores_
        figure = px.bar(x=numerical_data,y=fs.scores_,title= 'Numerical Correlation Using Anova',
                color=numerical_data,text=fs.scores_,labels={'x':'Features','y':'Anova Value'},color_discrete_sequence=color_bar)      
        figure.update_traces(marker_line_width=0,texttemplate='%{text:.2f}',textposition='outside',showlegend=False,hovertemplate='value:%{y}')
        if return_df == True:
            return figure, X_train_fs, X_cv_fs, X_test_fs
        else:
            return figure
    
    elif corr_meth == 'mutual_info_classif':
        """for categorical output """
        fs = SelectKBest(score_func=mutual_info_classif, k='all')
        fs.fit(df[numerical_data], df['target'].astype(float))
        df = fs.transform(df[numerical_data])
        mutual_classification_correlation_num = px.bar(x=numerical_data,y=fs.scores_,title= 'Numerical Correlation Using Mutual Info Classification',
                        color=numerical_data,text=fs.scores_,labels={'x':'','y':'Mutual Value'},color_discrete_sequence=color_bar)      
        mutual_classification_correlation_num.update_traces(marker_line_width=0,texttemplate='%{text:.2f}',textposition='outside',showlegend=False,hovertemplate='value:%{y}')

        if return_df == True:
            return figure, df
        else:
            return figure
    
    elif corr_meth == 'mutual_info_regression':
        """for numerical output """
        fs = SelectKBest(score_func=mutual_info_regression, k=k)
        fs.fit(X_train[numerical_data], y_train)
        X_train_fs = fs.transform(X_train[numerical_data])
        X_cv_fs = fs.transform(X_cv[numerical_data])
        X_test_fs = fs.transform(X_test[numerical_data])
        fs.scores_
        figure = px.bar(x=numerical_data,y=fs.scores_,title= "Numerical Correlation Using Mutual Info Regression",
                            color=numerical_data,text=fs.scores_,
                            labels={'x':'','y':'Mutual Information Value'},color_discrete_sequence=color_bar)
        figure.update_traces(marker_line_width=0,texttemplate='%{text:.2f}',textposition='outside',showlegend=False,hovertemplate='value:%{y}')
        if return_df == True:
            return figure, X_train_fs, X_cv_fs, X_test_fs
        else:
            return figure
    
    elif corr_meth == 'AIC':
        """For numerical output 
           Use numerical and categorical data
           Form: (estimator=Ridge(),X_train=X_train,y_train=y_train) """
        import dmba as dmba
        model = estimator
        def train_model(variables):
            if len(variables) == 0:
                return None  
            model.fit(X_train[variables], y_train)
            return model
        def score_model(model, variables):
            if len(variables) == 0:
                return dmba.AIC_score(y_train, [y_train.mean()] * len(y_train), model, df=1)
            return dmba.AIC_score(y_train, model.predict(X_train[variables]), model) 
        best_model, best_variables = dmba.stepwise_selection(X_train.columns, train_model,
        score_model, verbose=True)
        # print(f'Intercept: {float(best_model.intercept_):.3f}')
        # print('Coefficients:')
        features = []
        values = []
        for name, coef in zip(best_variables,best_model.coef_.tolist()[0]):
            features.append(name)
            values.append(coef)

        X_train_fs  = pd.DataFrame(X_train,columns=features)
        X_cv_fs  = pd.DataFrame(X_train,columns=features)
        X_test_fs = pd.DataFrame(X_train,columns=features)

        for i in X_train.columns:
            if i not in features:
                features.append(i)
                values.append(0)

        figure = px.bar(x=features,y=values,title= 'Numerical Correlation Using AIC',
                                   color=features, text=values,labels={'x':'','y':'Values'},color_discrete_sequence=color_bar)      
        figure.update_traces(marker_line_width=0,texttemplate='%{text:.2f}',textposition='outside',showlegend=False,hovertemplate='value:%{y}')

        if return_df == True:
            return figure, X_train_fs, X_cv_fs, X_test_fs
        else:
            return figure
    
def categorical_correlation(X_train,y_train,X_cv, X_test,categorical_data,corr_meth='chi2',return_df=False,k='all',
                            color_heat=None, color_bar=None):
    
    """
    Determines the correlation between variables for ctegorical data
    categorical_data = lista de columnas tipo categoria

    Return: Devuelve dataframes que solo contienen las columnas categoricas, elegidas
    """

    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.feature_selection import mutual_info_regression
    if corr_meth == 'chi2':
        """For categorical output"""
        fs = SelectKBest(score_func=chi2, k='all')
        fs.fit(df[categorical_data], df['target'].astype(float))
        df_fs = fs.transform(df[categorical_data])
        figure = px.bar(x=categorical_data,y=fs.scores_,title= 'Categorical Correlation Using Chi2',
                        color=categorical_data,text=fs.scores_,
                        labels={'x':'Features','y':'Chi2 Value'},color_discrete_sequence=color_bar)      
        figure.update_traces(marker_line_width=0,texttemplate='%{text:.2f}',textposition='outside',showlegend=False,hovertemplate='value:%{y}')
        if return_df == True:
            return figure, df_fs
        else:
            return figure

    elif corr_meth == 'mutual_info_classif':
        """for categorical output """
        fs = SelectKBest(score_func=mutual_info_classif, k='all')
        fs.fit(df[categorical_data], df['target'].astype(float))
        df_fs = fs.transform(df[categorical_data])
        figure = px.bar(x=categorical_data,y=fs.scores_,title= "Categorical Correlation Using Mutual Info Classification",
                        text=fs.scores_,labels={'x':'','y':'Mutual Info Value'},color=categorical_data,color_discrete_sequence=color_bar)
        figure.update_traces(marker_line_width=0,texttemplate='%{text:.2f}',textposition='outside',showlegend=False,hovertemplate='value:%{y}')
        if return_df == True:
            return figure, df_fs
        else:
            return figure
    
    elif corr_meth == 'mutual_info_regression':
        """for numerical output """
        fs = SelectKBest(score_func=mutual_info_regression, k=k)
        fs.fit(X_train[categorical_data], y_train)
        X_train_fs = fs.transform(X_train[categorical_data])
        X_cv_fs = fs.transform(X_cv[categorical_data])
        X_test_fs = fs.transform(X_test[categorical_data])
        fs.scores_
        figure = px.bar(x=categorical_data,y=fs.scores_,title= "Categorical Correlation Using Mutual Info Regression",
                     text=fs.scores_,labels={'x':'','y':'Mutual Information Value'},
                     color=categorical_data,color_discrete_sequence=color_bar)
        figure.update_traces(marker_line_width=0,texttemplate='%{text:.2f}',textposition='outside',showlegend=False,hovertemplate='value:%{y}')
        if return_df == True:
            return figure, X_train_fs, X_cv_fs, X_test_fs
        else:
            return figure

def feature_selection_stat(X_train,y_train,estimator,corr_meth,scoring,arg_fold={'n_splits':10}):
    
    """ Returns the best number of predictors with their score """

    from sklearn.model_selection import KFold
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import mutual_info_regression
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GridSearchCV

    cv = KFold(**arg_fold)
    fs = SelectKBest(score_func=corr_meth)
    pipeline = Pipeline(steps=[('fs',fs), ('estimator', estimator)])
    grid = dict()
    grid['fs__k'] = [i+1 for i in range(X_train.shape[1])]
    search = GridSearchCV(pipeline, grid, scoring=scoring, n_jobs=-1, cv=cv)
    results = search.fit(X_train, y_train)
    return results.best_params_, results.best_score_

def dim_red(X_train,X_cv,y_train,y_cv,estimator,dim_red_mod,scoring,number_components,cv=5):

    """ To check dimensional the reduction effect """

    from sklearn.model_selection import cross_val_score
    from numpy import mean
    X_train, X_cv, X_test, y_train, y_cv, y_test = df_to_numpy(X_train=X_train,X_cv=X_cv,y_train = y_train, y_cv = y_cv)
    estimator = estimator
    scores_ = []
    for i in range(1,number_components):
        dim_red = dim_red_mod.set_params(n_components=i)
        X_train_red = dim_red.fit_transform(X_train,y_train)
        X_cv_red = dim_red.transform(X_cv)
        score = cross_val_score(estimator=estimator,X=np.concatenate([X_train_red,X_cv_red]),y= np.concatenate([y_train,y_cv]),scoring=scoring,cv=cv, n_jobs=-1)
        print('i: ',i,' mean: ',mean(score))
        scores_.append(mean(score))
    max_score = np.max(scores_)
    print('Best i: ',int(scores_.index(max_score))+1,'; Score: ',max_score)

# <----------------------- MACHINE LEARNING MODELS -------------------------> #

def validation_curves(model,X_train,y_train,X_cv,y_cv,scoring,path_,cv=5,
                    original_or_positive='original',parameters_range={},save=False,
                    color_line=None):
    
    """ Return validation curves """

    from sklearn.model_selection import validation_curve
    import json 
    import plotly
    import os
    import numpy as np
    import pandas as pd
    import plotly.express as px

    if os.path.exists(os.path.join(path_,str(model.__class__.__name__))):
        path_ = os.path.join(path_,str(model.__class__.__name__))
    else:
        os.mkdir(os.path.join(path_,str(model.__class__.__name__)))
        path_= os.path.join(path_,str(model.__class__.__name__))

    """
    Dataframes in numpy form
    parameters_range: a dictionarie with key = str, and range = list
    """

    if 'numpy' in str(type(X_train)):
        X = np.concatenate((X_train,X_cv),axis=0)
        y = np.concatenate((y_train,y_cv),axis=0)       
    else:
        X=pd.concat([X_train,X_cv],axis=0)
        y=pd.concat([y_train,y_cv],axis=0)

    memory_ = 0
    memory_file = []

    if save == True:
        try:       
            conter = np.load(os.path.join(path_,'conter.npy'))
            fig_conter = np.load(os.path.join(path_,'fig_conter.npy'))

        except:
            conter = np.array(1)
            fig_conter = np.array(0)
            np.save('conter',conter)
            with open(os.path.join(path_,'validation_curves.json'),'a',encoding="utf-8") as f:
                f.write(json.dumps([parameters_range])) #json file that contain the curves made
            with open(os.path.join(path_,'validation_curves_report.txt'),'a',encoding="utf-8") as f:
                j = 1
                for i,k in parameters_range.items():
                        f.write(str(i)+ " : "+str(k)+ "-> figure name: "+'validation_curve_'+str(j)+f'\n')
                        j+=1

        if conter != 1: 
            with open(os.path.join(path_,'validation_curves.json'),'r+',encoding="utf-8") as f:
                read_ = f.read()

            data = json.loads(read_).copy()

            for k in data:
                for h,t in k.items():
                    try:
                        if parameters_range[h] == t: # quita las figuras que se repetirian, es decir las ya realizadas
                            memory_ +=1 # cuenta el numero de figuras ya realizadas
                            memory_file.append(h + ' : ' + str(t))
                            del parameters_range[h]
                    except:
                        pass

            data.append(parameters_range)

            with open(os.path.join(path_,'validation_curves.json'),'r+',encoding="utf-8") as f:
                f.write(json.dumps(data))

            with open(os.path.join(path_,'validation_curves_report.txt'),'a',encoding="utf-8") as f:
                j = 1
                for i,k in parameters_range.items():
                        f.write(str(i) + " : " + str(k) + "-> figure name: " + 'validation_curve_'+str(fig_conter+j)+f'\n')
                        j+=1

    else:
        try: 
            fig_conter = np.load(os.path.join(path_,'fig_conter.npy'))
            with open(os.path.join(path_,'validation_curves.json'),'r+',encoding="utf-8") as f:
                read_ = f.read()

            data = json.loads(read_).copy()

            for k in data:
                for h,t in k.items():
                    try:
                        if parameters_range[h] == t: # quita las figuras que se repetirian, es decir las ya realizadas
                            memory_ +=1 # cuenta el numero de figuras ya realizadas
                            memory_file.append(h + ' : ' + str(t))
                            del parameters_range[h]
                    except:
                        pass
        except:
            pass 

    fig = []

    if  memory_ != 0:

        for i in range(1,memory_+1):
                with open(os.path.join(path_,'validation_curves_report.txt'),'r',encoding="utf-8") as f:
                    l = f.readlines()
                    for i in range(0,len(l)):
                        a,b = l[i].split('-> ')
                        for i in memory_file:                          
                            if i == a:
                                curve = b[13:-1]
                                fig.append(plotly.io.read_json((os.path.join(path_,curve+'.json'))))

    for i,k in parameters_range.items():
        if save == True:
            fig_conter+=1
            conter+=1
            np.save(os.path.join(path_,'conter'),conter)
            
        train_scores, cv_scores = validation_curve(model, X, y.values.ravel(), param_name=i, param_range=k,cv=cv,scoring=scoring,n_jobs=-1)
        if original_or_positive == 'positive':
            train_scores_mean = np.abs(np.mean(train_scores, axis=1))
            train_scores_std = np.abs(np.std(train_scores, axis=1))
            cv_scores_mean = np.abs(np.mean(cv_scores, axis=1))
            cv_scores_std = np.abs(np.std(cv_scores, axis=1))
        else:
            train_scores_mean = (np.mean(train_scores, axis=1))
            train_scores_std = (np.std(train_scores, axis=1))
            cv_scores_mean = (np.mean(cv_scores, axis=1))
            cv_scores_std = (np.std(cv_scores, axis=1))

        mae_train_cv_poly_graph = px.line(y=train_scores_mean, x=k, 
            title = i.replace('_',' ').capitalize() +' Validation Curve',
            labels={'x':i.replace('_',' ').capitalize(),
                'y':scoring[4:].replace('_',' ').capitalize()},markers=True) 
        mae_train_cv_poly_graph.update_traces(showlegend=True,name='Train',line_color=color_line[0],hovertemplate=None)
        mae_train_cv_poly_graph.add_scatter(y=cv_scores_mean,x=k,name='Cv',line_color=color_line[1])
        mae_train_cv_poly_graph.update_layout(hovermode="x")

        if save == True:
            with open(os.path.join(path_,'validation_curve_'+str(fig_conter)+'.json'),'w',encoding="utf-8") as f:
                f.write(plotly.io.to_json(mae_train_cv_poly_graph))
            
            np.save(os.path.join(path_,'fig_conter'),fig_conter)

        fig.append(mae_train_cv_poly_graph)

    return fig # Figures list
    
def single_polynomial_regression(model,X_train,y_train,X_cv,y_cv,path_,numerical_data,categorical_data,degree=10,
                        mode='train',save=True,estimator_grid={},grid_par={},color_line=[None,None],color_bar=None):

    """ This function applies a simple polynomial regression for a single 
    target data set (Dataframes in numpy format) """

    from sklearn.linear_model import SGDRegressor
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    import joblib
    mae_train_poly = []
    mae_cv_poly = []
    r2_train_poly = []
    r2_cv_poly = []
    grid_par['estimator']=model
    grid_par['param_grid']=estimator_grid
    degree = degree+1
    results = {}
    if mode == 'train':
        
        for poly_grade in range(1,degree):
            poly_features = PolynomialFeatures(degree=poly_grade, include_bias=False)
            X_train_poly = poly_features.fit_transform(X_train[numerical_data])
            X_train_poly = pd.DataFrame(X_train_poly,columns=poly_features.get_feature_names_out())
            X_train_poly =pd.concat([X_train_poly,X_train[categorical_data]],axis=1)
            X_cv_poly = poly_features.fit_transform(X_cv[numerical_data])
            X_cv_poly = pd.DataFrame(X_cv_poly,columns=poly_features.get_feature_names_out())
            X_cv_poly =pd.concat([X_cv_poly,X_cv[categorical_data]],axis=1)
            poly_reg = GridSearchCV(**grid_par)
            poly_reg.fit(X_train_poly,y_train)
            y_pred_train_poly = poly_reg.predict(X_train_poly)
            y_pred_cv_poly = poly_reg.predict(X_cv_poly)
            mae_train_poly.append(mean_absolute_error(y_train,y_pred_train_poly))
            mae_cv_poly.append(mean_absolute_error(y_cv,y_pred_cv_poly))
            r2_train_poly.append(r2_score(y_train,y_pred_train_poly))
            r2_cv_poly.append(r2_score(y_cv,y_pred_cv_poly))
            if save == True:
                dir_ = os.path.join(path_, str(model.__class__.__name__)+'_polynomial_regression'+'_'+str(poly_grade)+'.joblib')
                print(dir_)
                joblib.dump(poly_reg,dir_)
            results[str(poly_grade)] = poly_reg.best_params_

    elif mode == 'design':
        for poly_grade in range(1,degree):
            poly_features = PolynomialFeatures(degree=poly_grade, include_bias=False)
            X_train_poly = poly_features.fit_transform(X_train[numerical_data])
            X_train_poly = pd.DataFrame(X_train_poly,columns=poly_features.get_feature_names_out())
            X_train_poly =pd.concat([X_train_poly,X_train[categorical_data]],axis=1)
            X_cv_poly = poly_features.fit_transform(X_cv[numerical_data])
            X_cv_poly = pd.DataFrame(X_cv_poly,columns=poly_features.get_feature_names_out())
            X_cv_poly =pd.concat([X_cv_poly,X_cv[categorical_data]],axis=1)
            poly_reg = joblib.load(os.path.join(path_, str(model.__class__.__name__)+'_polynomial_regression'+'_'+str(poly_grade)+'.joblib'))
            y_pred_train_poly = poly_reg.predict(X_train_poly)
            y_pred_cv_poly = poly_reg.predict(X_cv_poly)
            mae_train_poly.append(mean_absolute_error(y_train,y_pred_train_poly))
            mae_cv_poly.append(mean_absolute_error(y_cv,y_pred_cv_poly))
            r2_train_poly.append(r2_score(y_train,y_pred_train_poly))
            r2_cv_poly.append(r2_score(y_cv,y_pred_cv_poly))

    "two lines graph"
    mae_train_cv_poly_graph = px.line(y=mae_train_poly, x=list(range(1,degree)), title = 'Polynomial Regression Degrees',labels={'x':'Degree','y':'Mean Absolute Error'},markers=True) 
    mae_train_cv_poly_graph.update_traces(showlegend=True,name='Train',line_color=color_line[0])
    mae_train_cv_poly_graph.add_scatter(y=mae_cv_poly,x=list(range(1,degree)),name='Cv',line_color=color_line[1])   
    optimal_degree = np.argmin(abs(np.array(mae_cv_poly)-np.array(mae_train_poly)))+1
    mae_train_cv_poly_graph.add_shape(type='line',x0=optimal_degree ,y0=0,x1=optimal_degree,y1=np.array(mae_train_poly+mae_cv_poly).max(),
                            line=dict(color='MediumPurple',dash='dot'))
    mae_train_cv_poly_graph.update_layout(hovermode="x")
    r2 = pd.DataFrame({'train':r2_train_poly,'cv':r2_cv_poly})
    r2_train_cv_poly_graph = px.bar(r2,y=['train','cv'], x=list(range(1,degree)), 
                                title = 'R-Squared Training/Cv Set',
                                labels={'x':'Degree','value':'R2'},barmode='overlay',
                                color_discrete_sequence=color_bar)
    r2_train_cv_poly_graph.update_yaxes(range=(0,1))
    
    if mode == 'train':
         return mae_train_cv_poly_graph, r2_train_cv_poly_graph, results
    else:
        return mae_train_cv_poly_graph, r2_train_cv_poly_graph, None

def single_polynomial_classification(model,X_train,y_train,X_cv,y_cv,path_,degree=10,
                        mode='train',save=True,estimator_grid={},grid_par={},color_line=[None,None],
                        color_bar=None):

    """ This function applies a simple polynomial classification for a binary 
    target data set (Dataframes in numpy format) """

    from sklearn.linear_model import LogisticRegression
    from sklearn.linear_model import SGDRegressor
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics import f1_score  
    from sklearn.metrics import accuracy_score
    import joblib
    f1_train_poly = []
    f1_cv_poly = []
    acc_train_poly = []
    acc_cv_poly = []
    grid_par['estimator']=model
    grid_par['param_grid']=estimator_grid
    results = {}

    if mode == 'train':
        for poly_grade in range(1,degree):
            poly_features = PolynomialFeatures(degree=poly_grade, include_bias=False)
            X_train_poly = poly_features.fit_transform(X_train)
            X_cv_poly = poly_features.fit_transform(X_cv)
            poly_cla = GridSearchCV(**grid_par)
            poly_cla.fit(X_train_poly,y_train)
            y_pred_train_poly = poly_cla.predict(X_train_poly)
            y_pred_cv_poly = poly_cla.predict(X_cv_poly)
            f1_train_poly.append(f1_score(y_train,y_pred_train_poly))
            f1_cv_poly.append(f1_score(y_cv,y_pred_cv_poly))
            acc_train_poly.append(accuracy_score(y_train,y_pred_train_poly))
            acc_cv_poly.append(accuracy_score(y_cv,y_pred_cv_poly))
            if save == True:
                dir_ = os.path.join(path_,str(model.__class__.__name__)+'_polynomial_classification'+'_'+str(poly_grade)+'.joblib')
                joblib.dump(poly_cla,dir_)
            results[str(poly_grade)] = poly_cla.best_params_
    elif mode == 'design':
        for poly_grade in range(1,degree):
            poly_features = PolynomialFeatures(degree=poly_grade, include_bias=False)
            X_train_poly = poly_features.fit_transform(X_train)
            X_cv_poly = poly_features.fit_transform(X_cv)
            poly_cla = joblib.load(os.path.join(path_,str(model.__class__.__name__)+'_polynomial_classification'+'_'+str(poly_grade)+'.joblib'))
            y_pred_train_poly = poly_cla.predict(X_train_poly)
            y_pred_cv_poly = poly_cla.predict(X_cv_poly)
            f1_train_poly.append(f1_score(y_train,y_pred_train_poly))
            f1_cv_poly.append(f1_score(y_cv,y_pred_cv_poly))
            acc_train_poly.append(accuracy_score(y_train,y_pred_train_poly))
            acc_cv_poly.append(accuracy_score(y_cv,y_pred_cv_poly))

    "two lines graph"
    f1_train_cv_poly_graph = px.line(y=f1_train_poly, x=list(range(1,degree)), title = 'Polynomial Classification Degrees',
        labels={'x':'Degree','y':'F1 Score'},markers=True) 
    f1_train_cv_poly_graph.update_traces(showlegend=True,name='Train',line_color=color_line[0])
    f1_train_cv_poly_graph.add_scatter(y=f1_cv_poly,x=list(range(1,degree)),name='Cv',line_color=color_line[1])   
    optimal_degree = np.argmax(f1_cv_poly)+1
    f1_train_cv_poly_graph.add_shape(type='line',x0=optimal_degree ,y0=0,x1=optimal_degree,y1=np.array(f1_train_poly+f1_cv_poly).max(),
                            line=dict(color='MediumPurple',dash='dot'))
    f1_train_cv_poly_graph.update_layout(hovermode="x")
    acc = pd.DataFrame({'train':acc_train_poly,'cv':acc_cv_poly})
    acc_train_cv_poly_graph = px.bar(acc,y=['train','cv'], x=list(range(1,degree)),
         title = 'Accuracy Training/Cv Set',labels={'x':'Degree','value':'Accuracy','variable':''},barmode='overlay',
         color_discrete_sequence=color_bar)
    acc_train_cv_poly_graph.update_yaxes(range=(0,1))
    if mode == 'train':
         return f1_train_cv_poly_graph, acc_train_cv_poly_graph, results
    else:
        return f1_train_cv_poly_graph, acc_train_cv_poly_graph, None

def grid_reg(estimator,X_train,y_train,X_test,y_test, parameters,cv, scoring, refit=True):

    """ A short algorithm to do a grid search regression with some model and get 
    some plots and metrics """

    parameters_grid = [parameters]
    estimator = estimator
    grid_search = GridSearchCV(estimator,parameters_grid,cv=cv,scoring=scoring,\
                return_train_score=True,refit=refit)
    grid_search.fit(X_train,y_train)
    print("Best parameters: ",grid_search.best_params_)
    y_predicted = pd.DataFrame(pd.Series(grid_search.predict(X_test),name='predicted'))
    y_test = pd.DataFrame(y_test)
    y_test = y_test.rename(columns={y_test.columns[0]:'target'})
    y_test = y_test.reset_index(drop=True)
    # Evaluation
    print("R²: ",r2_score(y_test,y_predicted))
    print("MSE: ",mean_squared_error(y_test, y_predicted))
    print("RMSE: ", math.sqrt(mean_squared_error(y_test, y_predicted)))
    # Graph
    graph = pd.concat([y_test, y_predicted], axis=1)
    fig = px.bar(graph,y=[graph['target'],graph['predicted']])
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    fig.show()
    # Graph
    perc_err = abs((y_test.values-y_predicted.values)/y_test.values)*100
    perc_err = pd.DataFrame(perc_err)
    perc_err = perc_err.rename(columns={perc_err.columns[0]:'Error %'})
    fig = px.bar(perc_err,y=perc_err['Error %'])
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    fig.show()
    # Graph
    graph = pd.concat([y_test, y_predicted], axis=1)
    fig = px.scatter(graph,x='target',y='predicted')
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    fig.show()
    # Graph
    return y_predicted

def random_reg(estimator,X_train,y_train,X_test,y_test, parameters,cv,n_iter, scoring):

    """ A short algorithm to do a random search regression with some model and get 
    some plots and metrics """

    parameters_random = [parameters]
    random_search = RandomizedSearchCV(
        estimator, param_distributions=parameters_random,cv=cv, n_iter=n_iter,scoring=scoring,return_train_score=True
    )
    random_search.fit(X_train,y_train)
    print("Best Parameters: ",random_search.best_params_)
    y_predicted = pd.DataFrame(pd.Series(random_search.predict(X_test),name='predicted'))
    y_test = pd.DataFrame(y_test)
    y_test = y_test.rename(columns={y_test.columns[0]:'target'})
    # Evaluation
    print("R²: ",r2_score(y_test,y_predicted))
    print("MSE: ",mean_squared_error(y_test, y_predicted))
    print("RMSE: ", math.sqrt(mean_squared_error(y_test, y_predicted)))
    # Gráfica 
    graph = pd.concat([y_test.reset_index(drop=True), y_predicted], axis=1)
    fig = px.bar(graph,y=[graph['target'],graph['predicted']])
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    fig.show()
    # Graph
    perc_err = abs((y_test.values-y_predicted.values)/y_test.values)*100
    perc_err = pd.DataFrame(perc_err)
    perc_err = perc_err.rename(columns={perc_err.columns[0]:'Error %'})
    fig = px.bar(perc_err,y=perc_err['Error %'])
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    fig.show()
    return y_predicted

# <---------------------------- MODEL EVALUATION --------------------------> #

def baseline_model(y_train,y_cv,y_test,color_line=None,method='mean',color_bar=[None,None],
                                        color_error=px.colors.sequential.Rainbow):
    
    """ Algorithm to implement a baseline model for regression, 'y' must have a column called 
    'target' """
    
    import plotly.express as px
    from sklearn.metrics import r2_score
    import numpy as np
    import pandas as pd
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    from sklearn.metrics import mean_absolute_error

    if method == 'mean':
        y_pred_train = pd.DataFrame({'target':[float(y_train.mean())]*len(y_train)}).to_numpy()
        y_pred_cv = pd.DataFrame({'target':[float(y_cv.mean())]*len(y_cv)}).to_numpy()
        y_pred_test = pd.DataFrame({'target':[float(y_test.mean())]*len(y_test)}).to_numpy()
    elif method == "median":
        y_pred_train = pd.DataFrame({'target':[float(y_train.median())]*len(y_train)}).to_numpy()
        y_pred_cv = pd.DataFrame({'target':[float(y_cv.median())]*len(y_cv)}).to_numpy()
        y_pred_test = pd.DataFrame({'target':[float(y_test.median())]*len(y_test)}).to_numpy() 

    y_train = y_train.to_numpy()
    y_cv = y_cv.to_numpy()
    y_test = y_test.to_numpy()

    y_train,y_cv,y_test,y_pred_train,y_pred_cv,y_pred_test = \
        y_train.reshape(-1,1),y_cv.reshape(-1,1),y_test.reshape(-1,1),y_pred_train.reshape(-1,1),y_pred_cv.reshape(-1,1),y_pred_test.reshape(-1,1)

    mae = []
    r_2 = []
    set_ =  ['train','cv','test']
    figures = ['fig_1','fig_2','fig_3']
    individual_error = ['fig_4','fig_5','fig_6']
    for i,figure,ind in zip(set_,figures,individual_error):
        vars()['r2_'+i] = r2_score(vars()['y_'+i],vars()['y_pred_'+i])
        r_2.append(vars()['r2_'+i])
        vars()['mae_'+i] = mean_absolute_error(vars()['y_'+i],vars()['y_pred_'+i])
        mae.append(vars()['mae_'+i])

        y_p = pd.DataFrame(vars()['y_pred_'+i],columns=['Predicted Target'])
        y_o = pd.DataFrame(vars()['y_'+i],columns=['Original Target'])
        df = pd.concat([y_p,y_o],axis=1)
        vars()[figure] = px.scatter(df,y=['Original Target','Predicted Target'],trendline='lowess',labels={'value':'Lowess Target Value','index':'Instance'},
                                            color_discrete_sequence=color_line)
        vars()[figure].data = [t for t in vars()[figure].data if t.mode == "lines"]

        vars()[figure].add_annotation(xref = 'x domain', yref = 'y domain',
         text=f"R²:  {vars()['r2_'+i]:.3f}",showarrow=False,font=dict(family="arial",
                size=15,
                color="black"
            ))

        vars()[figure].update_traces(hovertemplate=None)
        vars()[figure].update_layout(title_text=f"Accuracy Graph in the {i.capitalize()} Set",hovermode="x")
        vars()[figure].update_traces(showlegend=True)

        # mae and r_2 graphs
        bias_var = make_subplots(rows=1, cols=2, shared_yaxes = False,subplot_titles=("MAE <br> ","R-Squared <br> "))
        bias_var.add_trace(
            go.Bar(y=mae,x = ['train','cv','test'],text=[f'{i:.2f}'for i in mae],marker_color=color_bar[0]),
            row=1, col=1
        )

        bias_var.add_trace(
            go.Bar(y=r_2, x = ['train','cv','test'], text=[f'{i:.2f}'for i in r_2],marker_color=color_bar[1]),
            row=1, col=2
        )

        bias_var.update_layout(title_text="Bias and Variance",showlegend=False) # height=600, width=800, 
        bias_var.update_traces(textfont_size=14, textangle=0, textposition="outside", cliponaxis=False)
        bias_var.update_yaxes(visible=True)
        bias_var.update_traces(hoverinfo='y')

        # Individual Graph

        # Loss graph
        umae = np.abs(vars()['y_'+i]-vars()['y_pred_'+i])
        array = np.concatenate([vars()['y_'+i],vars()['y_pred_'+i],umae],axis=1)
        e = pd.DataFrame(array,columns=['y_true','y_pred','error'])
        
        vars()[ind] = px.scatter(e,y='y_true',x=[x for x in range(0,e.shape[0])],color='error',
                color_continuous_scale=color_error,size='error',
                labels={'x':'Instance','y_true':'True Target','error':'Absolute Error'}, 
                title=f"Individual Error in the {i.capitalize()} Set")

    return vars()['fig_1'], vars()['fig_2'],vars()['fig_3'],vars()['fig_4'], vars()['fig_5'],vars()['fig_6'],bias_var

def single_regression_model_evaluation(y_train,y_pred_train,y_cv,y_pred_cv,y_test,
                                        y_pred_test,color_line=None,color_bar=[None,None],
                                        color_error=px.colors.sequential.Rainbow):
    
    """
        Regression model evaluation function
    """
    import plotly.express as px
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_absolute_error
    import numpy as np
    import pandas as pd
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    y_train,y_cv,y_test,y_pred_train,y_pred_cv,y_pred_test = \
          df_to_numpy(y_train,y_cv,y_test,y_pred_train,y_pred_cv,y_pred_test)

    y_train,y_cv,y_test,y_pred_train,y_pred_cv,y_pred_test = \
    y_train.reshape(-1,1),y_cv.reshape(-1,1),y_test.reshape(-1,1),y_pred_train.reshape(-1,1),y_pred_cv.reshape(-1,1),y_pred_test.reshape(-1,1)
    mae = []
    r_2 = []
    set_ =  ['train','cv','test']
    figures = ['fig_1','fig_2','fig_3']
    individual_error = ['fig_4','fig_5','fig_6']
    for i,figure,ind in zip(set_,figures,individual_error):
        vars()['r2_'+i] = r2_score(vars()['y_'+i],vars()['y_pred_'+i])
        r_2.append(vars()['r2_'+i])
        vars()['mae_'+i] = mean_absolute_error(vars()['y_'+i],vars()['y_pred_'+i])
        mae.append(vars()['mae_'+i])
        vars()['rmae_'+i]= np.sqrt(mean_squared_error(vars()['y_'+i],vars()['y_pred_'+i]))
        y_p = pd.DataFrame(vars()['y_pred_'+i],columns=['Predicted Target'])
        y_o = pd.DataFrame(vars()['y_'+i],columns=['Original Target'])
        df = pd.concat([y_p,y_o],axis=1)
        vars()[figure] = px.scatter(df,y=['Original Target','Predicted Target'],trendline='lowess',labels={'value':'Lowess Target Value','index':'Instance'},
                                            color_discrete_sequence=color_line)
        vars()[figure].data = [t for t in vars()[figure].data if t.mode == "lines"]

        vars()[figure].add_annotation(xref = 'x domain', yref = 'y domain',
         text=f"R²:  {vars()['r2_'+i]:.3f}",showarrow=False,font=dict(family="arial",
                size=15,
                color="black"
            ))

        vars()[figure].update_traces(hovertemplate=None)
        vars()[figure].update_layout(title_text=f"{i.capitalize()} Set Accuracy Graph ",hovermode="x")
        vars()[figure].update_traces(showlegend=True)


        # mae and r_2 graphs
        bias_var = make_subplots(rows=1, cols=2, shared_yaxes = False,subplot_titles=("MAE <br> ","R-Squared <br> "))
        bias_var.add_trace(
            go.Bar(y=mae,x = ['train','cv','test'],text=[f'{i:.2f}'for i in mae],marker_color=color_bar[0]),
            row=1, col=1
        )

        bias_var.add_trace(
            go.Bar(y=r_2, x = ['train','cv','test'], text=[f'{i:.2f}'for i in r_2],marker_color=color_bar[1]),
            row=1, col=2
        )

        bias_var.update_layout(title_text="Bias and Variance",showlegend=False) # height=600, width=800, 
        bias_var.update_traces(textfont_size=14, textangle=0, textposition="outside", cliponaxis=False)
        bias_var.update_yaxes(visible=True)
        bias_var.update_traces(hoverinfo='y')

        # Loss graph
        umae = np.abs(vars()['y_'+i]-vars()['y_pred_'+i])
        array = np.concatenate([vars()['y_'+i],vars()['y_pred_'+i],umae],axis=1)
        e = pd.DataFrame(array,columns=['y_true','y_pred','error'])
        vars()[ind] = px.scatter(e,y='y_true',x='y_pred',color='error',
                color_continuous_scale=color_error,size='error',
                labels={'y_pred':'Target Predicted','y_true':'True Target','error':'Absolute Error'}, 
                title=f"Individual Error in the {i.capitalize()} Set")

    return vars()['fig_1'], vars()['fig_2'],vars()['fig_3'],vars()['fig_4'], vars()['fig_5'],vars()['fig_6'],bias_var

def binary_classification_model_evaluation(y_train,y_pred_train_prob,y_cv,
            y_pred_cv_prob,y_test,y_pred_test_prob,zero='0',one='1',color_cf='BrBG',color_line=None):

    """
        Classification model evaluation function
        all dataframe must be in array (numpy) format
        y_train = True labes
        y_pred_train = Labels Predicted
        y_pred_train_prob = Probability of the predicted labels
    """

    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import f1_score        
    from sklearn.metrics import average_precision_score
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_recall_curve
    from sklearn.model_selection import cross_val_score

    set_ =  ['train','cv','test']
    
    # Recall/precision graphs
    threshold_confusion_matrix = []
    figures_pr = ['fig_1','fig_3','fig_5'] # precision/recall
    for i, figure in zip(set_,figures_pr):
        ap = average_precision_score(vars()['y_'+i],vars()['y_pred_'+i+'_prob'])
        precision, recall, threshold = precision_recall_curve(vars()['y_'+i],vars()['y_pred_'+i+'_prob'])
        threshold = np.append(threshold,1)
        with np.errstate(divide='warn'):
             f1_score = (precision*recall) / (precision+recall)
             f1_score [(precision+recall) == 0] = 0
        max_f1 = max(f1_score)
        index_max_f1 = np.argmax(f1_score)
        best_threshold= threshold[index_max_f1]
        threshold_confusion_matrix.append(best_threshold)
        y_pred_best_f1 = (vars()['y_pred_'+i+'_prob']>best_threshold).astype(int)
        accuracy = accuracy_score(vars()['y_'+i],y_pred_best_f1)
        vars()[figure] = px.line(x=precision,y=recall,labels={'x':'Precision','y':'Recall'},title=f'{i.capitalize()} Precision/Recall Metric',
            hover_data={'Precision':np.round(precision,3),'Recall':np.round(recall,3),'F1 score': np.round(f1_score,3),
            'Threshold':np.round(threshold,3)})
        vars()[figure].add_annotation(xref = 'x domain', yref = 'y domain', text=f'AP: {ap:.3f} <br> Best F1: {max_f1:.3f}'
        f'<br> Threshold: {best_threshold:.3f} <br> Accuracy: {accuracy:.3f}',showarrow=False, font=dict(size=15))
        vars()[figure].add_shape(type='line',x0=precision[index_max_f1],y0=0,x1=precision[index_max_f1],y1=1,
                            line=dict(color='MediumPurple',dash='dot'))
        vars()[figure].update_traces(line_color=color_line)

    # Create a confusion matrix, convert to a percentage and rounded to one decimal
    figures_cm = ['fig_2','fig_4','fig_6']
    vars()['y_pred_train'] = (y_pred_train_prob > threshold_confusion_matrix[0]).astype(int)
    vars()['y_pred_cv'] = (y_pred_cv_prob > threshold_confusion_matrix[1]).astype(int)
    vars()['y_pred_test'] = (y_pred_test_prob > (threshold_confusion_matrix[0]+threshold_confusion_matrix[1])/2).astype(int)
    for i,figure in zip(set_,figures_cm):
        vars()['cm_'+i] = np.around(confusion_matrix(vars()['y_'+i],vars()['y_pred_'+i],normalize='true')*100,decimals=1)
        vars()[figure] = px.imshow(vars()['cm_'+i],\
            labels={'x':'Predicted labels','y':'True labels'},\
                x = [zero,one], y = [zero,one],text_auto=True,color_continuous_scale=color_cf)

        vars()[figure].update_layout(title_text=f"{i.capitalize()} Confusion Percentage Matrix")

        if i == 'test':
            vars()[figure].update_layout(title_text=f"{i.capitalize()} Confusion Percentage Matrix <br>"
            f"<sup> Threshold = Mean (Training set, CV set) </sub>")

    return vars()['fig_1'], vars()['fig_2'],vars()['fig_3'],vars()['fig_4'], vars()['fig_5'],vars()['fig_6'],threshold_confusion_matrix[1]

def learning_curve( model, X, y, cv=None, n_jobs=-1, scoring=None, 
        train_sizes=np.linspace(0.1, 1.0, 5), transform_to_positive = False, y_range = None, random_state=42,color_line=None):

    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    title : str
        Title for the chart.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    scoring : str or callable, default=None
        A str (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """

    from sklearn.model_selection import learning_curve
    import numpy as np
    import pandas as pd
    import plotly.express as px

    train_sizes, train_scores, cv_scores, fit_times, _ = learning_curve(
        model,
        X,
        y,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
        random_state=random_state
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    cv_scores_mean = np.mean(cv_scores, axis=1)
    cv_scores_std = np.std(cv_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    if transform_to_positive:
        train_scores_mean *= -1 
        train_scores_std *=  -1
        cv_scores_mean *=  -1
        cv_scores_std *=  -1 

    # Plot learning curve
    model_name = str(model.__class__.__name__)
    learning_values = pd.DataFrame({'Train':train_scores_mean,'Cv':cv_scores_mean})
    learning_curve_ = px.line(learning_values,y=['Train','Cv'],x=train_sizes,
        title = model_name.capitalize()+' Learning Curve',markers=True, 
        labels={'variable':'','x':'Training examples','value':'Score: '+scoring.replace('_', ' ').capitalize()},
        color_discrete_sequence=color_line)
    if y_range != None:
        learning_curve_.update_yaxes(range=y_range)
    # Plot n_samples vs fit_times
    fit_times_vs_n_exam_values = pd.DataFrame({'Time':fit_times_mean})
    fit_times_vs_n_exam_curve_ = px.line(fit_times_vs_n_exam_values,y='Time',
        x=train_sizes,title = model_name.capitalize()+': Scalability of the model',markers=True, 
        labels={'variable':'','x':'Training examples','value':'Time (sec)'},
        color_discrete_sequence=color_line)
    
    # Plot fit_time vs score
    fit_time_argsort = fit_times_mean.argsort()
    fit_time_sorted = fit_times_mean[fit_time_argsort]
    test_scores_mean_sorted = cv_scores_mean[fit_time_argsort]
    test_scores_std_sorted = cv_scores_std[fit_time_argsort]

    fit_times_vs_score_values = pd.DataFrame({'Time':fit_time_sorted,'score':test_scores_mean_sorted})
    fit_times_vs_score_curve_ = px.line(fit_times_vs_score_values,y='score',x='Time',
        title = model_name.capitalize()+': Performance of the model',markers=True, 
        labels={'variable':'','x':'Time (sec)','value':'Score: '+scoring.replace('_',' ').capitalize()},
        color_discrete_sequence=color_line)

    return learning_curve_, fit_times_vs_n_exam_curve_, fit_times_vs_score_curve_

def calibration_curve(estimator_name,y_true,y_pred,color_line=None,color_his=None):

    """  Calibration curves (also known as reliability diagrams) compare how 
    well the probabilistic predictions of a binary classifier are calibrated"""

    from sklearn.calibration import calibration_curve
    import plotly.express as px
    estimator = estimator_name.capitalize()
    freq , med = calibration_curve(y_true,y_pred,n_bins=25)
    calibration_cur = px.line(x=med,y=freq,markers=True, title=estimator+' Calibration Curve',labels={'x':'Mean predicted probability (Positive class: 1)','y':'Fraction of positives (Positive class: 1)'})
    calibration_cur.add_shape(type='line',x0=0 ,y0=0,x1=1,y1=1,line=dict(color='MediumPurple',dash='dot'))
    calibration_cur.update_traces(line_color=color_line)
    hist = px.histogram(x=y_pred,nbins=20,title=estimator+': Number of samples in \n each bin',labels={'x':'Mean predicted probability (Positive class: 1)'},\
                        color_discrete_sequence=[color_his])
    hist.update_layout(yaxis_title="Count") 
    return calibration_cur, hist

# Other Tools

def neural_network_fig(layers_list):

    """
    It makes a representative graph of the neural network use
    layers_list = number of neurons at each layer e.g. [1,5,10]
    """

    import itertools
    import networkx as nx

    subset_sizes = layers_list
    def multilayered_graph(*subset_sizes):
        extents = nx.utils.pairwise(itertools.accumulate((0,) + subset_sizes))
        layers = [range(start, end) for start, end in extents]
        G = nx.Graph()
        for (i, layer) in enumerate(layers):
            G.add_nodes_from(layer, layer=i)
        for layer1, layer2 in nx.utils.pairwise(layers):
            G.add_edges_from(itertools.product(layer1, layer2))
        return G


    G = multilayered_graph(*subset_sizes)
    pos = nx.multipartite_layout(G, subset_key="layer")

    edge_x = []
    edge_y = []
    for i in list(G.edges):
        x0, y0 = pos[i[0]]
        x1, y1 = pos[i[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            # colorscale options
            #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale='Blues',
            reversescale=False,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))
    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append('# of connections: '+str(len(adjacencies[1])))

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title='<br>Network Graph <br>',
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    annotations=[ dict(
                        text="",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002 ) ],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    return fig

# <---------------------------- DASK FUNCTIONS ----------------------------> #

def dask_shape(df):

    """ Shape but for a dash dataframe """
    
    import dask.dataframe as dd
    from dask.diagnostics import ProgressBar
    with ProgressBar():
        a = df.shape
        shape = [a[0].compute(),a[1]]
    return shape

def dask_shape_figure (rows_number,columns_number,color=px.colors.diverging.BrBG):

    """ Shape figure but for a dash dataframe """

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    shape_ = make_subplots(rows=1, cols=2, shared_yaxes = False)
    shape_.add_trace(
        go.Bar(y=[rows_number],x = ["Rows"],text=str(rows_number),marker_color=color[-1]),
        row=1, col=1
    )

    shape_.add_trace(go.Bar(y=[columns_number], x = ["Columns"], text=str(columns_number),marker_color=color[1]),row=1, col=2)

    shape_.update_layout(title_text="Shape <br><sup>Without target</sup>",showlegend=False) # height=600, width=800, 
    shape_.update_traces(marker_line_width=0,textfont_size=17, textangle=0, textposition="inside", cliponaxis=False)
    shape_.update_yaxes(visible=False)
    shape_.update_traces(hoverinfo='y') 
    return shape_

def dask_missing (df):

    """ Missing values but for a dash dataframe """

    import dask.dataframe as dd
    from dask.diagnostics import ProgressBar
    missing_values = df.isnull().sum()
    missing_perc = ((missing_values/df.index.size)*100)
    with ProgressBar():
        missing_perc = missing_perc.compute()
    return missing_perc

def dask_missing_figure(missing,color=None):

    """ Missing figure but for a dash dataframe """

    import plotly.express as px
    hist = px.histogram(y=missing,title='Missing Values',labels={'y':'Percentage'},color_discrete_sequence=color)
    hist.update_layout(xaxis_title="Counts") 
    hist.update_traces(texttemplate='%{x}'+ ' features',textposition='outside')
    return hist
