import plotly.express as px
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from numpy import mean
from sklearn.compose import ColumnTransformer
import plotly
from sklearn.base import clone
import itertools
import re 
# Encoder
from sklearn.preprocessing import OneHotEncoder
# Imputers
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
# Outliers
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
# Feature selectors
from sklearn.feature_selection import SelectKBest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import r_regression
# Scalers
from sklearn.preprocessing import RobustScaler, MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler
# Estimators
from sklearn.linear_model import LinearRegression, Ridge, TweedieRegressor, QuantileRegressor
from sklearn.linear_model import Lasso, ElasticNet, BayesianRidge, ARDRegression
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.neural_network import MLPRegressor
# Distribution
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.compose import ColumnTransformer, make_column_selector

""" <-------------------------------- SET ---------------------------------> """

df = pd.read_csv('train.csv')
df.drop(columns=['Id'],inplace=True)
X_raw = df.drop(columns=['target']).copy()
y_raw = df['target'].copy()
numerical_data = list(X_raw.columns)
categorical_data = []
encoding_data = []

# Split, es necesario cambiar el nombre de la columna objetivo a 'target'

X_train, X_rem, y_train, y_rem = train_test_split(X_raw,y_raw, train_size=0.6,random_state=42)
X_cv, X_test, y_cv, y_test = train_test_split(X_rem,y_rem, train_size=0.5,random_state=42)

X = np.concatenate([X_train,X_cv])
X = pd.DataFrame(X,columns=X_raw.columns)
y = np.concatenate([y_train,y_cv])
y = pd.DataFrame(y,columns=['target'])

""" <----------------------------- FUNCTIONS ------------------------------> """

def delete_nan(X,y,data):
    df_ = pd.concat([X,y],axis=1)
    df_ = df_.dropna(subset=data).reset_index(drop=True).copy() 
    X, y = df_.drop(columns=['target']), pd.DataFrame(df_['target'])          
    return X, y

def outlier_models(X,y,estimator,numerical_data):
    df = pd.concat([X,y],axis=1)

    if estimator == 'lof':
        lof = LocalOutlierFactor()
        X_bool_outliers = lof.fit_predict(df[numerical_data])

    elif estimator == 'ifo':
        ifo = IsolationForest()
        X_bool_outliers = ifo.fit_predict(df[numerical_data])

    mask = X_bool_outliers != -1
    X, y = df.drop(columns=['target']).iloc[mask,:], pd.DataFrame(df['target'][mask])      
    return X,y

def parameter_iterator(transformer,params):
    combination = []
    for t,k in enumerate(transformer):
        keys, values = zip(*params[t].items())
        permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
        for i in permutations_dicts:
            try:
                combination.append(clone(k.set_params(**i)))
            except:
                combination.append(k)
                  
    return combination

def name_columns(transformer_names,transformer):
    columns_name = []
    for i in transformer.get_feature_names_out():
        for k in transformer_names:
            if k in i:
                name = i.replace(k,'')
                columns_name.append(name)
    return columns_name

def best_transformer(name,best_pipeline):
    for n,i in zip(best_pipeline[name].keys(),best_pipeline[name].values()):
        try:
            if i > j:
                best = n
        except:
            j=i
            best = n
    return best

def alpha_preprocessing(X,y,name_numerical_imputer,numerical_imputer,
                         name_categorical_imputer,categorical_imputer,
                         name_encoder, encoder,
                         name_scaler,scaler,scaler_params,
                         name_estimator,estimator,estimator_params,
                         numerical_data,categorical_data,encoding_data,n_iterations,df=False):

    """  MISSING DATA AND ENCODIG
        Este algoritmo realiza la combinación de 2 imputadores (uno númerico y otro categorico),
        un codificador y "n" número de estimadores para determinar que combinación de los imputadores y
        el codificador da un mejor score para un estimador u otro. 
        [ A / B of iteration C/D ] = A es la combinación en la cual va el algoritmo,
                                     B es el número de combinaciones totales,
                                     C es la iteración en la cual va
                                     D es el número de iteraciones totales
        Si el target posee valores perdidos agregarlo a la categoria correspondiente ya sea numerico o categoric,
        es decir categorical_data = ['target'] + categorical_data
    """

    iteration = 1
    number_of_iterations = len(name_numerical_imputer)*len(name_categorical_imputer)*len(name_encoder)*len(name_estimator)*len(name_scaler)
                    #    len(name_outlier)*len(name_num_cat_feature)*len(name_numerical_feature)*\
                    #    len(name_categorical_feature )*len(name_distribution)*len(name_scaler)*len(name_estimator)
        
    numerical_imputer_conter = []
    categorical_imputer_conter = []
    categorical_encoder_conter = []
    estimator_conter = []
    scaler_conter = []
    categorical_data_copy = categorical_data.copy()

    columns = ['Numerical_Imputer','Categorical_Imputer','Encoding','Scaler','Estimator','Score','Iteration']
    results = pd.DataFrame(columns=columns)
    while iteration <= n_iterations:
        
        conter=1
        score_ = []
        data = []
        for name_numerical_imputer_, numerical_imputer_ in \
                            zip(name_numerical_imputer,numerical_imputer):

            for name_categorical_imputer_, categorical_imputer_ in \
                                zip(name_categorical_imputer,categorical_imputer):

                for name_encoder_, encoder_ in zip(name_encoder, encoder): 

                    for name_scaler_, scaler_, scaler_params_ in zip(name_scaler,scaler,scaler_params):

                        for name_estimator_, estimator_,estimator_params_ in zip(name_estimator,estimator,estimator_params):  

                            if  name_numerical_imputer_ == 'passthrough' and name_categorical_imputer_ == 'passthrough':
                                X_,y_ = delete_nan(X,y,categorical_data+numerical_data)

                            elif name_numerical_imputer_  == 'passthrough':
                                X_,y_ = delete_nan(X,y,numerical_data)
                            
                            elif  name_categorical_imputer_ == 'passthrough':
                                X_,y_ = delete_nan(X,y,categorical_data)

                            else:
                                X_,y_ = X,y
    
                            # Preprocessor Imputer

                            numerical_preprocessing = Pipeline([
                                ('numerical_imputer', numerical_imputer_),
                            ])

                            categorical_preprocessing = Pipeline([
                                ('categorical_imputer', categorical_imputer_),
                            ])

                            preprocessor = ColumnTransformer([
                                            ('numerical',numerical_preprocessing,numerical_data),
                                            ('categorical', categorical_preprocessing,categorical_data),
                                        ],verbose_feature_names_out=True,remainder='passthrough')

                            X_ = preprocessor.fit_transform(X_)
                            transformer = ['numerical__','categorical__','remainder__']
                            columns_name = []
                            for i in preprocessor.get_feature_names_out():
                                for k in transformer:
                                    if k in i:
                                        name = i.replace(k,'')
                                        columns_name.append(name)

                            X_ = pd.DataFrame(X_,columns=columns_name)

                            # Encoder
                            encoding_preprocessing = Pipeline([
                                ('encoding', encoder_)
                            ])

                            preprocessor_encoder = ColumnTransformer([
                                    ('encoder',encoding_preprocessing,encoding_data)
                                ],verbose_feature_names_out=True,remainder='passthrough')
                                
                            X_ = preprocessor_encoder.fit_transform(X_)

                            transformer = ['encoder__','remainder__']
                            columns_name = []
                            for i in preprocessor_encoder.get_feature_names_out():
                                for k in transformer:
                                    if k in i:
                                        name = i.replace(k,'')
                                        columns_name.append(name)

                            X_ = pd.DataFrame(X_,columns=columns_name)

                            categorical_data = [x for x in X_.columns if x not in numerical_data]
                            params = {**estimator_params_,**scaler_params_}
                            regression_pipe = Pipeline([
                                        ('scaler',scaler_),
                                        ('estimator',estimator_)
                            ])

                            regression_grid = GridSearchCV(regression_pipe,param_grid=params,scoring='r2',cv=5,n_jobs=1)
                            regression_grid.fit(X_,y_.values.ravel())
                            best_params = regression_grid.best_params_
                            score_.append(regression_grid.best_score_)
                            information = "Numerical_Imputer: {0}; "\
                                        "Catergorical_Imputer: {1}; "\
                                        "Encoding: {2}; "\
                                        "Scaler: {3}; "\
                                        "Estimator: {4}; "\
                                        "Score: {5}".format(name_numerical_imputer_,
                                                            name_categorical_imputer_,
                                                            name_encoder_,
                                                            name_scaler_,
                                                            name_estimator_,
                                                            regression_grid.best_score_)
                            data.append(information)
                            #print(information + " [{0} / {1} of iteration {2}/{3}]".format(conter,number_of_iterations,iteration,n_iterations)+'\n')
                            if df == False:
                                results.loc[len(results)] = [name_numerical_imputer_,
                                                            name_categorical_imputer_,
                                                            name_encoder_,
                                                            name_scaler_,
                                                            name_estimator_,
                                                            regression_grid.best_score_,
                                                            " [{0} / {1} of iteration {2}/{3}]".format(conter,number_of_iterations,iteration,n_iterations)]
                                results.to_csv('results_preprocess_1.csv',index=False)

                            categorical_data = categorical_data_copy 
                            conter+=1
        idx = (-(np.array(score_))).argsort()[:5] # Select high score index
        for i in idx:
            numerical_imputer_conter.append(data[i].split('; ')[-6].split(': ')[1]) # Estimator conter
            categorical_imputer_conter.append(data[i].split('; ')[-5].split(': ')[1])
            categorical_encoder_conter.append(data[i].split('; ')[-4].split(': ')[1])
            scaler_conter.append(data[i].split('; ')[-3].split(': ')[1])
            estimator_conter.append(data[i].split('; ')[-2].split(': ')[1])
        iteration +=1

    best_pipeline = {}
    final_results = {}
    if df == False:
        with open('best_estimators_1.txt','w') as f:
            f.write('\nNUMERICAL_IMPUTER\n')
            for i in list(set(numerical_imputer_conter)):
                final_results[i]=numerical_imputer_conter.count(i)
                f.write(i+': '+str(final_results[i])+'\n')
                try:
                    best_pipeline['NUMERICAL_IMPUTER']={**{i:final_results[i]},**best_pipeline['NUMERICAL_IMPUTER']}
                except:
                    best_pipeline['NUMERICAL_IMPUTER']={i:final_results[i]}
            f.write('\nCATEGORICAL_IMPUTER\n')
            for i in list(set(categorical_imputer_conter)):
                final_results[i]=categorical_imputer_conter.count(i)
                f.write(i+': '+str(final_results[i])+'\n')
                try:
                    best_pipeline['CATEGORICAL_IMPUTER']={**{i:final_results[i]},**best_pipeline['CATEGORICAL_IMPUTER']}
                except:
                    best_pipeline['CATEGORICAL_IMPUTER']={i:final_results[i]}
            f.write('\nCATEGORICAL_ENCODER\n')
            for i in list(set(categorical_encoder_conter)):
                final_results[i]=categorical_encoder_conter.count(i)
                f.write(i+': '+str(final_results[i])+'\n')
                try:
                    best_pipeline['CATEGORICAL_ENCODER']={**{i:final_results[i]},**best_pipeline['CATEGORICAL_ENCODER']}
                except:
                    best_pipeline['CATEGORICAL_ENCODER']={i:final_results[i]}
            f.write('\nSCALER\n')
            for i in list(set(scaler_conter)):
                final_results[i]=scaler_conter.count(i)
                f.write(i+': '+str(final_results[i])+'\n')
                try:
                    best_pipeline['SCALER']={**{i:final_results[i]},**best_pipeline['SCALER']}
                except:
                    best_pipeline['SCALER']={i:final_results[i]}
            f.write('\nESTIMATOR\n')
            for i in list(set(estimator_conter)):
                final_results[i]=estimator_conter.count(i)
                f.write(i+': '+str(final_results[i])+'\n')
                try:
                    best_pipeline['ESTIMATOR']={**{i:final_results[i]},**best_pipeline['ESTIMATOR']}
                except:
                    best_pipeline['ESTIMATOR']={i:final_results[i]} 
    else:
        with open('parameter_for_preprocess_2.txt','w') as f:
            f.write('\nNUMERICAL_IMPUTER\n')
            for i in list(set(numerical_imputer_conter)):
                f.write((i)+'\n')
            f.write('\nCATEGORICAL_IMPUTER\n')
            for i in list(set(categorical_imputer_conter)):
                f.write((i)+'\n')
            f.write('\nCATEGORICAL_ENCODER\n')
            for i in list(set(categorical_encoder_conter)):
                f.write((i)+'\n')

    if df == True:
        return X_,y
    return best_pipeline

def beta_preprocessing(X,y,name_outlier,outlier,
                         name_num_cat_feature,num_cat_feature,
                         name_numerical_feature,numerical_feature,
                         name_categorical_feature,categorical_feature,
                         name_distribution,distribution,
                         name_scaler,scaler,
                         name_estimator,estimator,estimator_params,
                         numerical_data,categorical_data,n_iterations,return_best=False):

    """  MISSING DATA AND ENCODIG
        Este algoritmo realiza la combinación de 2 imputadores (uno númerico y otro categorico),
        un codificador y "n" número de estimadores para determinar que combinación de los imputadores y
        el codificador da un mejor score para un estimador u otro. 
        [ A / B of iteration C/D ] = A es la combinación en la cual va el algoritmo,
                                     B es el número de combinaciones totales,
                                     C es la iteración en la cual va
                                     D es el número de iteraciones totales
    """

    iteration = 1
    number_of_iterations = len(name_outlier)*len(name_num_cat_feature)*\
                           len(name_numerical_feature)*len(name_estimator)*len(name_scaler)*\
                           len(name_categorical_feature )*len(name_distribution)
                    #    len(name_outlier)*len(name_num_cat_feature)*len(name_numerical_feature)*\
                    #    len(name_categorical_feature )*len(name_distribution)*len(name_scaler)*len(name_estimator)
    
    # For the file text    
    distribution_conter = []
    outlier_conter = []
    num_cat_feature_conter = [] 
    numerical_feature_conter = []
    categorical_feature_conter = []
    scaler_conter = []
    estimator_conter = []
    # Fin
    categorical_data_copy = categorical_data.copy()
    numerical_data_copy = numerical_data.copy()
    columns = ['Distribution','Outlier','Num_Cat_Feature','Numerical_Feature','Categorical_Feature','Scaler','estimator','Score','Iteration']
    results = pd.DataFrame(columns=columns)
    while iteration <= n_iterations:
        
        conter=1
        score_ = []
        data = []
        for name_outlier_,outlier_ in zip(name_outlier,outlier):

            for name_num_cat_feature_,num_cat_feature_ in zip(name_num_cat_feature,num_cat_feature):

                for  name_numerical_feature_,numerical_feature_ in zip( name_numerical_feature,numerical_feature): 

                    for name_categorical_feature_,categorical_feature_ in zip(name_categorical_feature,categorical_feature):

                        for  name_distribution_,distribution_ in zip(name_distribution,distribution): 

                            for name_scaler_, scaler_ in zip(name_scaler,scaler):

                                for name_estimator_, estimator_,estimator_params_ in \
                                                zip(name_estimator,estimator,estimator_params):  

                                    X_,y_ = X,y
            
                                    # Predict outliers

                                    if name_outlier_ != 'None':
                                        df = pd.concat([X_,y_],axis=1)
                                        X_bool_outliers = outlier_.fit_predict(df[numerical_data])
                                        mask = X_bool_outliers != -1
                                        X_, y_ = df.drop(columns=['target']).iloc[mask,:], pd.DataFrame(df['target'][mask])
                                    
                                    #print('Outlier',X_.shape)

                                    # Feature selection
                                    try:

                                        num_cat = ColumnTransformer([('num_cat_feature',num_cat_feature_,numerical_data+categorical_data)],remainder='passthrough',verbose_feature_names_out=True)

                                        X_ = num_cat.fit_transform(X_,y_)

                                        columns_name = name_columns(['num_cat_feature__','remainder__'],num_cat)
                                        X_ = pd.DataFrame(X_,columns=columns_name)

                                        categorical_data = [x for x in X_.columns if x not in numerical_data]
                                        numerical_data = [x for x in X_.columns if x not in categorical_data]
                                        #print('MUTUAL: ',X_.shape)

                                        num = ColumnTransformer([('numerical_feature', numerical_feature_,numerical_data)],verbose_feature_names_out=True,remainder='passthrough')
                                        X_ = num.fit_transform(X_,y_.values.ravel())

                                        columns_name = name_columns(['numerical_feature__','remainder__'],num)
                                        X_ = pd.DataFrame(X_,columns=columns_name)
                                        
                                        categorical_data = [x for x in X_.columns if x not in numerical_data]
                                        numerical_data = [x for x in X_.columns if x not in categorical_data]

                                        #print('PEARSON: ',X_.shape)
                                        cat = ColumnTransformer([('categorical_feature', categorical_feature_,categorical_data)],verbose_feature_names_out=True,remainder='passthrough')
                                        X_ = cat.fit_transform(X_,y_)

                                        columns_name = name_columns(['categorical_feature__','remainder__'],cat)
                                        X_ = pd.DataFrame(X_,columns=columns_name)

                                        categorical_data = [x for x in X_.columns if x not in numerical_data]
                                        numerical_data = [x for x in X_.columns if x not in categorical_data]

                                        #print('CHI2: ',X_.shape)
                                    
                                    except:
                                        conter+=1
                                        categorical_data = categorical_data_copy 
                                        numerical_data = numerical_data_copy
                                        break

                                    dist = ColumnTransformer([('distribution', distribution_,numerical_data)],verbose_feature_names_out=True,remainder='passthrough')
                                    X_ = dist.fit_transform(X_,y_)

                                    columns_name = name_columns(['distribution__','remainder__'],dist)
                                    X_ = pd.DataFrame(X_,columns=columns_name)

                                    categorical_data = [x for x in X_.columns if x not in numerical_data]
                                    numerical_data = [x for x in X_.columns if x not in categorical_data]

                                    #print('DISTRIBUTION: ',X_.shape)

                                    sca = ColumnTransformer([('scaler',scaler_,numerical_data)],verbose_feature_names_out=True,remainder='passthrough')
                                    X_ = sca.fit_transform(X_)


                                    columns_name = name_columns(['scaler__','remainder__'],sca)
                                    X_ = pd.DataFrame(X_,columns=columns_name)

                                    categorical_data = [x for x in X_.columns if x not in numerical_data]
                                    numerical_data = [x for x in X_.columns if x not in categorical_data]
                                    #print('SCALER: ',X_.shape)

                                    regression_pipe = Pipeline([
                                        ('estimator',estimator_)
                                    ])
 
                                    regression_grid = GridSearchCV(regression_pipe,param_grid=estimator_params_,scoring='r2',cv=5,n_jobs=3)
                                    regression_grid.fit(X_,y_.values.ravel())
                                    best_params = regression_grid.best_params_
                                    score_.append(regression_grid.best_score_)
                                    information = "Distribution: {0}; "\
                                                  "Outlier: {1}; "\
                                                  "Num_cat_feature: {2}; "\
                                                  "Numerical_feature: {3}; "\
                                                  "Categorical_feature: {4}; "\
                                                  "Scaler: {5}; "\
                                                  "Estimator: {6}; "\
                                                  "Score: {7}".format(name_distribution_,
                                                                        name_outlier_,
                                                                        name_num_cat_feature_,
                                                                        name_numerical_feature_,
                                                                        name_categorical_feature_,
                                                                        name_scaler_,
                                                                        name_estimator_,
                                                                        regression_grid.best_score_)
                                    data.append(information)
                                    #print(information + " [{0} / {1} of iteration {2}/{3}]".format(conter,number_of_iterations,iteration,n_iterations)+'\n')
                                    results.loc[len(results)] = [name_distribution_,
                                                                 name_outlier_,
                                                                 name_num_cat_feature_,
                                                                 name_numerical_feature_,
                                                                 name_categorical_feature_,
                                                                 name_scaler_,
                                                                 name_estimator_,
                                                                 regression_grid.best_score_,
                                                                " [{0} / {1} of iteration {2}/{3}]".format(conter,number_of_iterations,iteration,n_iterations)]
                                    results.to_csv('results_preprocess_2.csv',index=False)
                                    categorical_data = categorical_data_copy 
                                    numerical_data = numerical_data_copy
                                    conter+=1
                                    #print(X_)
                                    
        idx = (-(np.array(score_))).argsort()[:5] # Select high score index
        for i in idx:
            distribution_conter.append(data[i].split('; ')[-8].split(': ')[1])
            outlier_conter.append(data[i].split('; ')[-7].split(': ')[1])
            num_cat_feature_conter.append(data[i].split('; ')[-6].split(': ')[1]) # Estimator conter
            numerical_feature_conter.append(data[i].split('; ')[-5].split(': ')[1])
            categorical_feature_conter.append(data[i].split('; ')[-4].split(': ')[1])
            scaler_conter.append(data[i].split('; ')[-3].split(': ')[1])
            estimator_conter.append(data[i].split('; ')[-2].split(': ')[1])
        iteration +=1

    final_results = {}
    with open('best_estimators_2.txt','w') as f:
        f.write('DISTRIBUTION\n')
        for i in list(set(distribution_conter)):
            final_results[i]=distribution_conter.count(i)
            f.write(i+': '+str(final_results[i])+'\n')
        f.write('\nOUTLIER\n')
        for i in list(set(outlier_conter)):
            final_results[i]=outlier_conter.count(i)
            f.write(i+': '+str(final_results[i])+'\n')
        f.write('\nNUM_CAT_FEATURE\n')
        for i in list(set(num_cat_feature_conter)):
            final_results[i]=num_cat_feature_conter.count(i)
            f.write(i+': '+str(final_results[i])+'\n')
        f.write('\nNUMERICAL_FEATURE\n')
        for i in list(set(numerical_feature_conter)):
            final_results[i]=numerical_feature_conter.count(i)
            f.write(i+': '+str(final_results[i])+'\n')
        f.write('\nCATEGORICAL_FEATURE\n')
        for i in list(set(categorical_feature_conter)):
            final_results[i]=categorical_feature_conter.count(i)
            f.write(i+': '+str(final_results[i])+'\n')
        f.write('\nSCALER\n')
        for i in list(set(scaler_conter)):
            final_results[i]=scaler_conter.count(i)
            f.write(i+': '+str(final_results[i])+'\n')
        f.write('\nESTIMATOR\n')
        for i in list(set(estimator_conter)):
            final_results[i]=estimator_conter.count(i)
            f.write(i+': '+str(final_results[i])+'\n')
        if return_best == True:
            f.write('BEST PARAMS\n')
            f.write(str(best_params))

""" ------------------------ALPHA PREPROCESSING PARAMETERS------------------------ """

# <--------------------------------------------------------------------------> #      
# <-------------------------- NUMERICAL IMPUTER -----------------------------> #

numerical_imputer = ['passthrough',SimpleImputer(),KNNImputer(),IterativeImputer(random_state=0,max_iter=3000)]

numerical_imputer_params = [
     {'None':[None]}, # Delete Imputer
     {'strategy':['median','mean']},
     {'None':[None]},
     {'None':[None]} 
    ]


numerical_imputer = parameter_iterator(numerical_imputer,numerical_imputer_params)

name_numerical_imputer = [str(i) for i in numerical_imputer]

# <-------------------------- CATEGORICAL IMPUTER ---------------------------> #

categorical_imputer = ['passthrough',SimpleImputer()]

categorical_imputer_params = [
     {'None':[None]}, # Delete-Imputer
     {'strategy':['most_frequent']}, # Simple-Imputer
    ]


categorical_imputer = parameter_iterator(categorical_imputer,categorical_imputer_params)
name_categorical_imputer = [str(i) for i in categorical_imputer]

# <---------------------------- ENCODER -------------------------------------> #

encoder = [OneHotEncoder()]

encoder_params = [
     {'None':[None]} # OneHotEncoder
    ]
    

encoder = parameter_iterator(encoder,encoder_params)

name_encoder = [str(i) for i in encoder]

# <--------------------------------- SCALER ---------------------------------> #

scaler = ['passthrough',RobustScaler(), MinMaxScaler(), MaxAbsScaler(),StandardScaler()]

scaler_params = [
     {}, # None
     {}, # Robust
     {}, # MinMax
     {}, # MaxAbs
     {}, # StandardScaler
    ]


name_scaler = [str(i) for i in scaler]

# <------------------------------- PREDICTOR --------------------------------> #

name_estimator = ['LinearRegression']

estimator = [LinearRegression()]

estimator_params = [
        {} # LogisticRegression
      ]

# <--------------------------------------------------------------------------> #

""" ------------------------ RUN PREPROCESSING ONE------------------------- """

best_pipeline = alpha_preprocessing(X,y,name_numerical_imputer,numerical_imputer,
                        name_categorical_imputer,categorical_imputer,
                        name_encoder, encoder,
                        name_scaler,scaler,scaler_params,
                        name_estimator,estimator,estimator_params,
                        numerical_data,categorical_data,encoding_data,20,df=False) 

numerical_imputer_index = name_numerical_imputer.index(best_transformer('NUMERICAL_IMPUTER',best_pipeline))

categorical_imputer_index = name_categorical_imputer.index(best_transformer('CATEGORICAL_IMPUTER',best_pipeline))

encoder_imputer_index = name_encoder.index(best_transformer('CATEGORICAL_ENCODER',best_pipeline))

# Para el segundo alpha_preprocessing el df = True para que devuelva los dataframes,
# el número de iteraciones igual a 1 porque no es necesario más

X_preprocessed_1, y_preprocessed_1 = alpha_preprocessing(X,y,[name_numerical_imputer[numerical_imputer_index]],[numerical_imputer[numerical_imputer_index]],
                        [name_categorical_imputer[categorical_imputer_index]],[categorical_imputer[categorical_imputer_index]],
                        [name_encoder[encoder_imputer_index]], [encoder[encoder_imputer_index]],
                        [name_scaler[0]],[scaler[0]],[scaler_params[0]],
                        name_estimator,estimator,estimator_params,
                        numerical_data,categorical_data,encoding_data,1,df=True) 

# Para el 1 paso beta return_best = False para elegir lso mejores transformadoes y n_iterations debe ser alto para realizar varias
# pruebas ya que los transformadores pueden ser estocasticos

categorical_data = list(X_preprocessed_1.columns[[i not in numerical_data for i in X_preprocessed_1.columns]])

# <------------------------------- OUTLIER ----------------------------------> #

outlier = [None,LocalOutlierFactor(),IsolationForest()]

outlier_params = [
     {'None':[None]}, # None
     {'None':[None]}, # LOF
     {'None':[None]} # IFO
    ]


outlier = parameter_iterator(outlier,outlier_params)

name_outlier = [str(i) for i in outlier]

# <----------------------- NUM CAT FEATURES / DIME RED ----------------------> #

num_cat_feature = ['passthrough',SelectKBest(score_func=mutual_info_regression),LinearDiscriminantAnalysis(),PCA()]

num_cat_feature_params = [
     {'None':[None]}, # None
     {'k':[i for i in range(1,len(X_preprocessed_1.columns)+1)]}, # KBest
     {'n_components':[i for i in range(1,len(X_preprocessed_1.columns)+1)]}, # LDA
     {'n_components':[i for i in range(1,len(X_preprocessed_1.columns)+1)]} # PCA
    ]  

num_cat_feature = parameter_iterator(num_cat_feature,num_cat_feature_params)

name_num_cat_feature = [str(i) for i in num_cat_feature]

# <--------------------------- NUMERICAL FEATURES ---------------------------> #

numerical_feature = ['passthrough',SelectKBest(score_func=r_regression)]

numerical_feature_params = [
     {'None':[None]}, # None
     {'k':[i for i in range(1,len(numerical_data)+1)]}, # KBest
    ]

numerical_feature = parameter_iterator(numerical_feature,numerical_feature_params)

name_numerical_feature = [str(i) for i in numerical_feature]

# <-------------------------- CATEGORICAL FEATURES --------------------------> #

categorical_feature = ['passthrough',SelectKBest(score_func=chi2)]

categorical_feature_params = [
     {'None':[None]}, # None
     {'k':[i for i in range(1,len(categorical_data)+1)]}, # KBest
    ]

categorical_feature = parameter_iterator(categorical_feature,categorical_feature_params)

name_categorical_feature = [str(i) for i in categorical_feature]

# <--------------------------------- SCALER ---------------------------------> #

scaler = ['passthrough',RobustScaler(), MinMaxScaler(), MaxAbsScaler(),StandardScaler()]

scaler_params = [
     {'None':[None]}, # None
     {'None':[None]}, # Robust
     {'None':[None]}, # MinMax
     {'None':[None]}, # MaxAbs
     {'None':[None]}, # StandardScaler
    ]

scaler = parameter_iterator(scaler,scaler_params)

name_scaler = [str(i) for i in scaler]

# <------------------------------ DISTRIBUTION ------------------------------> #

distribution = ['passthrough',QuantileTransformer(output_distribution='normal'),PowerTransformer()]

distribution_params = [
     {'None':[None]}, # passthroug
     {'None':[None]}, # Quantile
     {'None':[None]}  # Power
    ]

distribution = parameter_iterator(distribution,distribution_params)

name_distribution = [str(i) for i in distribution]

# <------------------------------- PREDICTOR --------------------------------> #

name_estimator = ['LinearRegression']

estimator = [LinearRegression()]

estimator_params = [
        {} # LogisticRegression
      ]
# <--------------------------------------------------------------------------> #

beta_preprocessing(X_preprocessed_1, y_preprocessed_1,name_outlier,outlier,
                         name_num_cat_feature,num_cat_feature,
                         name_numerical_feature,numerical_feature,
                         name_categorical_feature,categorical_feature,
                         name_distribution,distribution,
                         name_scaler,scaler,
                         name_estimator,estimator,estimator_params,
                         numerical_data,categorical_data,3,return_best=False)
                         

