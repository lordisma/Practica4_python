# -*- coding: utf-8 -*-
"""
@author: Ismael Marin Molina
"""
#Science kit learn files
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.linear_model import Lasso ,Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.externals import 	joblib

#Others files
import matplotlib.pyplot as plt
import numpy as np


#Catcher Warnings
import warnings
from sklearn.exceptions import DataConversionWarning

################################################################################

def plot_WeightLasso(XTrain,YTrain):
    n_alphas = 100
    alphas = np.logspace(-10, -2, n_alphas)

    coefs = []
    for a in alphas:
        lasso = Lasso(alpha=a, fit_intercept=False)
        lasso.fit(XTrain, YTrain)
        coefs.append(lasso.coef_)

    # #############################################################################
    # Display results

    ax = plt.gca()

    ax.plot(alphas, coefs)
    ax.set_xscale('log')
    ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
    plt.xlabel('alpha')
    plt.ylabel('weights')
    plt.title('Ridge coefficients as a function of the regularization')
    plt.axis('tight')
    pass

def plot_WeightRidge(XTrain,YTrain):
    n_alphas = 100
    alphas = np.logspace(-10, -2, n_alphas)

    coefs = []
    for a in alphas:
        ridge = Ridge(alpha=a, fit_intercept=False)
        ridge.fit(XTrain, YTrain)
        coefs.append(ridge.coef_)

    # #############################################################################
    # Display results

    ax = plt.gca()

    ax.plot(alphas, coefs)
    ax.set_xscale('log')
    ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
    plt.xlabel('alpha')
    plt.ylabel('weights')
    plt.title('Ridge coefficients as a function of the regularization')
    plt.axis('tight')
    pass

def plot_Polinomial_Score(XTrain,YTrain, model, alpha = [0.0001], bias = False, score = 'explained_variance'):
    parameters = {'Pol__degree':[1,2,3,4,5,6,7], 'Model__alpha':alpha}

    pipe = Pipeline([('Pol',preprocessing.PolynomialFeatures(include_bias = bias)),
                      ('Scale',preprocessing.StandardScaler()),
                      ('Model',model)])

    grid = GridSearchCV(pipe, param_grid=parameters, cv=5, scoring=score)

    with warnings.catch_warnings(): #Catch conversion warnings
        warnings.simplefilter("ignore")
        grid.fit(XTrain,YTrain)

    resultados = grid.cv_results_['mean_test_score']
    resultados[resultados < 0.0] = 0.0
    print(grid.cv_results_['mean_test_score'])

    plt.plot([1,2,3,4,5,6,7], resultados)
    plt.ylim([0.0,1.0])
    plt.xlabel('Polinomio')
    plt.ylabel('Varianza Explicada')
    plt.title('Grado Polinomico')
    plt.axis('tight')

    pass

def plot_Alpha_Weigth(XTrain,YTrain, model, degree = 4, bias = False, score = 'explained_variance'):
    parameters = {'Model__alpha':[0.01,0.001,0.0001,1e-5,1e-6,1e-7,1e-8,1e-9,1e-10]}

    pipe = Pipeline([('Pol',preprocessing.PolynomialFeatures(degree = degree, include_bias = bias)),
                      ('Scale',preprocessing.StandardScaler()),
                      ('Model',model)])

    grid = GridSearchCV(pipe, param_grid=parameters, cv=5, scoring=score)

    with warnings.catch_warnings(): #Catch conversion warnings
        warnings.simplefilter("ignore")
        grid.fit(XTrain,YTrain)

    resultados = grid.cv_results_['mean_test_score']
    print(grid.cv_results_['mean_test_score'])

    ax = plt.gca()

    ax.plot([0.01,0.001,0.0001,1e-5,1e-6,1e-7,1e-8,1e-9,1e-10], resultados)
    ax.set_xscale('log')
    ax.set_xlim(ax.get_xlim()[::-1])
    plt.xlabel('alpha')
    plt.ylabel('Varianza Explicada')
    plt.title('Modelo por el Alpha')
    plt.axis('tight')

    pass


def Save(cls,name):
    joblib.dump(cls,name)
    pass


#################################################################################
#if __name__ == 'main':
print(__doc__)
seed = 50627728
splits = 5
siz_Test = 0.3
saveLasso = 'Lasso_model.pkl'
saveRidge = 'Ridge_model.pkl'

#Para ignorar los errores provocados por la no convergencia de la función
#así como la conversión de los datos y los de desactualizado
#Valores de salida del print acotados a 4 decimales
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})

#######Cargado de los datos#######
Feature  =  np.load("datos/airfoil_self_noise_X.npy")
Label    =  np.load("datos/airfoil_self_noise_y.npy")

#Particionamiento de los datos
Validation_Feature, Test_Feature , Validation_Label, Test_Labelt = train_test_split(
    Feature, Label, test_size = siz_Test, random_state = seed)


######Seleccion de parametros a usar#####
parameters_Lasso = [{'Pol__degree':[1,2,3,4,5,6,8],'Pol__include_bias':[True,False],'Model__alpha': [0.0001,0.01,1e-8,1e-16],'Model__selection':['random','cyclic']}]
parameters_Ridge = [{'Pol__degree':[1,2,3,4,5,6,8],'Pol__include_bias':[True,False],'Model__alpha': [0.0001,0.01,1e-8,1e-16]}]

######Pipe donde incluimos Escalado y Modelo##########
pipeLasso = Pipeline([('Pol',preprocessing.PolynomialFeatures()),
                      ('Scale',preprocessing.StandardScaler()),
                      ('Model',Lasso(random_state=seed))])
pipeRidge = Pipeline([('Pol',preprocessing.PolynomialFeatures()),
                      ('Scale',preprocessing.StandardScaler()),
                      ('Model',Ridge(random_state=seed))])

#Metricas usadas para explicar los datos
scores = ['neg_mean_squared_error','r2','explained_variance']

for score in scores:
    print("=========={}===========".format(score))
    print()
    grid = GridSearchCV(pipeLasso, param_grid=parameters_Lasso, cv=splits, scoring=score)
    gridR = GridSearchCV(pipeRidge, param_grid=parameters_Ridge, cv=splits, scoring=score)

    #####Ajustado de los datos####
    with warnings.catch_warnings(): #Catch conversion warnings
        warnings.simplefilter("ignore")
        grid.fit(Validation_Feature, Validation_Label)
        gridR.fit(Validation_Feature, Validation_Label)

    Save(grid,saveLasso) #Guardado del modelo para un uso más rapido en futuros momentos
    Save(gridR,saveRidge)

    ####Impresion de los datos####
    print("Lasso: Mejor valor de la cross validation: {:.4f}".format(grid.best_score_))
    print("Lasso: Mejores parametros: {}".format(grid.best_params_))
    print()
    print("Ridge: Mejor valor de la cross validation: {:.4f}".format(gridR.best_score_))
    print("Ridge: Mejores parametros: {}".format(gridR.best_params_))
    print()
    print("Puntuaciones de los hiperparametros")
    print()
    print("Lasso Result")
    for mean, std, params in zip(grid.cv_results_['mean_test_score'],grid.cv_results_['std_test_score'],grid.cv_results_['params']):
        print("%0.3f (+/-%0.03f) con los para: %r" % (mean, std*2, params))

    print()
    print("Ridge Result")
    for mean, std, params in zip(gridR.cv_results_['mean_test_score'],gridR.cv_results_['std_test_score'],gridR.cv_results_['params']):
        print("%0.3f (+/-%0.03f) con los para: %r" % (mean, std*2, params))

    print()
    print()
#Graficas : Cambio de la funcion de perdida según el alpha y el grado del polinomio
# varianza explicada método Biplot
print("Graficas de los modelos Lasso y Ridge")
plot_Alpha_Weigth(Validation_Feature, Validation_Label, Lasso(random_state=seed), degree = 5)
plt.show()
plot_Alpha_Weigth(Validation_Feature, Validation_Label, Ridge(random_state=seed))
plt.show()
plot_Polinomial_Score(Validation_Feature, Validation_Label, Lasso(random_state=seed), alpha = [1e-16])
plt.show()
plot_Polinomial_Score(Validation_Feature, Validation_Label, Ridge(random_state=seed))
plt.show()
plot_WeightRidge(Validation_Feature, Validation_Label)
plt.show()
plot_WeightLasso(Validation_Feature, Validation_Label)
plt.show()
####################Resultados sobre el TEST###########################
# Parte final incorporada solo antes de entregar los resultados
# no se lo que puede salir
print()
print("Puntuacion final sobre el test de Lasso: ")
print(grid.score(Test_Feature, Test_Labelt))
print()
print("Puntuacion final sobre el test de Ridge: ")
print(gridR.score(Test_Feature, Test_Labelt))
