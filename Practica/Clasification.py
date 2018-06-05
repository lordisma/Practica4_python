# -*- coding: utf-8 -*-

"""
    File name: Classification.py
    Author: Ismael Marín Molina, and Manuel Herrera Ojea
    Python Version: 3.5
"""


# Archivos requeridos de scikit-learn

from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.externals import 	joblib
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

# Otras dependencias

import matplotlib.pyplot as plt
import numpy as np
import itertools # Necesaria para el dibujo de la matriz y la curva ROC
from scipy import interp # Necesario para la curva ROC
import os # Necesario para buscar archivos
import pandas as pd
import seaborn as sns

# Advertencias mostradas por pantalla

import warnings
from sklearn.exceptions import DataConversionWarning



def plot_security(cm, Xtest, Ytest):

    value = cm.predict_proba(Xtest)
    value_8 = value[:,0]

    points = value_8[Ytest == 0]
    plt.scatter(range(len(points)), points)
    plt.title("Numeros y seguridad")
    plt.xlabel("Indice")
    plt.ylabel("Seguridad")


# Gráficas de la matriz de confusión

def plot_confusion_matrix(cm, classes, normalize=False, title='Matriz de Confusión', cmap=plt.cm.Blues):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Valor')
    plt.xlabel('Prediccion')
    pass


# Gráficas de la curva ROC multiclase

def plot_ROC(XTest, YTest, clf, Labels ):

    row = np.where( YTest == i )

    fpr, tpr, thresholds = roc_curve( YTest, clf.predict_proba(XTest)[:,Labels], pos_label=Labels )

    plt.figure()
    plt.plot(fpr,tpr,
             label='ROC modelo',
             color='deeppink', linestyle=':', linewidth=4)


    plt.plot([0,1],[0,1],'k--',lw=2)
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel('Falsos Positivos')
    plt.ylabel('Verdaderos Positivos')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    pass


# Tratamiento de valores perdidos

def Imputation( X ):

    missings = np.where(X == 'unknown')
    row = missings[0]
    Ypos = missings[1]
    columns = np.unique(Ypos)
    categoricalAttributes = [1,2,3,4,5,6,7,8,9,14]
    Rest = np.setdiff1d(categoricalAttributes, columns)

    numberCha = []
    classesNumber = np.zeros(X.shape[0])
    for column in columns:
        numberCha.append(len(row[Ypos == column])/X.shape[0])
        le = preprocessing.LabelEncoder()
        X[:,column] = le.fit_transform(X[:,column])
        index, = np.where(le.classes_ == 'unknown')
        classesNumber[column] = index[0]

    for rest in Rest:
        le = preprocessing.LabelEncoder()
        X[:,rest] = le.fit_transform(X[:,rest])

    for i in range(len(columns)):
        print("La columna {} tiene un {}% de valores perdidos".format(columns[i],numberCha[i]*100))

    KnnImputer = np.array([columns[i] for i in range(len(numberCha)) if numberCha[i] > 0.01])
    SimpleImputer = np.setdiff1d(columns, KnnImputer)
    print()
    print("Las columnas seleccionadas para la imputacion con KNN: {}".format(KnnImputer))
    print("Las columnas seleccionadas para la imputacion simple : {}".format(SimpleImputer))

    for i in SimpleImputer:
        Simple = preprocessing.Imputer(missing_values=classesNumber[i], strategy='most_frequent')
        X[:,SimpleImputer] = Simple.fit_transform(X[:,SimpleImputer])

    for i in KnnImputer:
        Knn = KNeighborsClassifier(n_neighbors = 3, p=2, algorithm='kd_tree', leaf_size=12, n_jobs=-1)
        Label = X[:,i]
        row_miss = row[Ypos == i]
        normal_row = np.setdiff1d(range(len(Label)), row_miss)
        Caracteristicas = np.delete(X, i, axis=1)
        Test_Label  = Label[row_miss]
        Train_Label = Label[normal_row].astype(np.int64)
        Knn.fit(Caracteristicas[normal_row,:],Train_Label)
        X[row_miss,i] = Knn.predict(Caracteristicas[row_miss,:])
        print("{}:Tamaño conjunto de Test: {}".format(i,Test_Label.shape))
        print("X: {}: :Caracteristicas: {}".format(X.shape,Caracteristicas.shape))

    return X


#

def plot_Importance( rfc ):
    importance = rfc.feature_importances_
    indices = np.argsort(importance)

    indices = indices[np.where(importance > 0.05)]

    plt.figure()
    plt.title("Feature importances (> 0.05)")
    plt.bar(range(len(indices)), importance[indices],
           color="r", align="center")
    plt.xticks(range(len(indices)), indices, rotation='vertical')
    plt.xlim([-1, len(indices)])
    pass


# Funciones adicionales de utilidad

def Find( name, path ):
    for root, dirs, files in os.walk( path ):
        if name in files:
            return True
    return False

def Save( cls,name ):
    joblib.dump( cls, name )
    pass

def Create( names ):

    data = pd.read_csv( 'data/bank-additional-full.csv', sep = ';' )
    Label = np.array( data[ 'y' ] )
    Features = data.values
    Features = Features[ :, 0:len( list( data ) ) - 1 ]

    np.save( 'data/Label.npy', Label )
    np.save( 'data/Feature.npy', Features )

    pass



### Programa principal ###



#if __name__ == 'main':

print(__doc__)


# Constantes iniciales

seed = 50627728
maxiter = 10000
splits = 5
siz_Test = 0.3
saveName = 'Bank.pkl'
featuresName = "Feature.npy"
labelsName = "Label.npy"
path = "data/"


# Dentro de cada dato de entrada, localización de
# las características categóricas y de las numéricas

categoricalAttributes = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 14 ]
numericAttributes = np.setdiff1d( range(20), categoricalAttributes )


# Generar los archivos .npy si no existían ya

if not Find( featuresName, path = path ):
    print( "############CREANDO ARCHIVOS NPY##################" )
    Create( names = [ featuresName, labelsName ] )


# Ignorar los errores provocados por la no convergencia de la función,
# así como porla conversión de los datos y los de desactualizado

warnings.filterwarnings( action = 'ignore', category = DataConversionWarning )


# Valores mostrados por pantalla acotados a 4 decimales

np.set_printoptions( formatter = { 'float': lambda x : "{0:0.4f}".format( x ) } )


# Cargado de los datos

Features = np.load( "data/Feature.npy" )
Labels   = np.load( "data/Label.npy" )

# Tratamiento de valores perdidos

missings = np.where( Features == 'unknown' )
print( "Cantidad de valores perdidos: {}".format( len( missings[1] ) ) )
print( "Distribucion:" )
Features = Imputation( Features )
missings = np.where( Features == 'unknown' )
print( "Valores perdidos tras el procesamiento: {}".format( len( missings[1] ) ) )

# Separación del conjunto en train y test

X_train, X_test , y_train, y_test = train_test_split(
    Features, Labels, stratify=Labels, test_size = siz_Test, random_state = seed)



# Eliminar las instancias donde la característica Duración vale 0

durationIndex = 10
removeIndices = np.where( X_train[:,durationIndex] == 0 )[0]
X_train = np.delete( X_train, removeIndices, axis = 0 )
y_train = np.delete( y_train, removeIndices, axis = 0 )

# Codificar las clases con índices numéricos.
# En el conjunto original aparecen como 'yes' y 'no'

Le = preprocessing.LabelEncoder().fit( y_train )

index_of_positive = np.where(Le.classes_ == 'yes')[0]

print("INDEX::::::::::::::{}".format(index_of_positive[0]))

y_train = Le.transform(y_train)
y_test = Le.transform(y_test)

for i in np.unique( y_train ):
    print( "Número de instancias en la clase {}: {}  {}"
        .format( i, len( np.where( y_train == i )[0] ), len( np.where( y_test == i )[0] ) )
    )

print()


#############################Valores Perdidos###############################


# Selección de hiperparametros que se evaluarán

parameters = [{
    'Model__C' : [ 1.0, 1e-6 ],
    'Model__kernel' : [ 'rbf', 'poly', 'sigmoid' ],
    'Model__decision_function_shape' : [ 'ovo', 'ovr' ]
}]


### Preprocesado de los datos ###

# Normalización de las características numéricas
SS = preprocessing.StandardScaler().fit( X_train[:,numericAttributes] )
X_train[:,numericAttributes] = SS.transform( X_train[:,numericAttributes] )
X_test[:,numericAttributes] = SS.transform( X_test[:,numericAttributes] )


# Binarización de características categóricas.
# Usamos M.todense() para ver los datos en tamaño normal. Si no, se guardan en formato COOmatrix

OHE = preprocessing.OneHotEncoder(
    categorical_features = categoricalAttributes,
    handle_unknown = 'ignore'
    ).fit( X_train )
X_train = OHE.transform(X_train).todense()
X_test = OHE.transform(X_test).todense()

# Equilibrado de representación de cada clase

sm = SMOTE( ratio = 'minority', random_state = seed, k_neighbors = 3 )
Xres, Yres = sm.fit_sample( X_train, y_train )


# Creación y ajuste del modelo de aprendizaje Random Forest

rfc = RandomForestClassifier(
    n_estimators = 50,
    n_jobs = -1,
    max_depth = 30,
    min_samples_leaf = 10,
    max_features = "sqrt" ).fit( Xres, Yres )


# Valoración del aprendizaje realizado

print( "Scorer al evaluar sobre el conjunto de aprendizaje: {}".format( rfc.score( X_train, y_train ) ) )

plot_Importance(rfc=rfc)
plt.show()

print()
print( "Valor en el test:" )
print( classification_report( y_test, rfc.predict( X_test ) ) )
plot_ROC(X_test, y_test, rfc, index_of_positive[0])
plt.show()

#Aquellos donde RandomForestClassifier no se equivoca

importance = rfc.feature_importances_
indices = np.argsort( importance )

indices = indices[ np.where( importance > 0.05 ) ]

prediction = rfc.predict( Xres )
prediction_index = np.where( Yres == prediction )[0]
prediction_index = prediction_index.reshape( -1 )
indices = indices.reshape( -1 )

Xres = Xres[prediction_index,:]
Xres = Xres[:,indices]
Yres = Yres[prediction_index]


##########################################################################

# Pipe donde incluimos Escalado y Modelo
######Pipe donde incluimos Escalado y Modelo##########

pipe = Pipeline( [ ( 'Model', SVC( max_iter = maxiter ) ) ] )
grid = GridSearchCV( pipe, param_grid = parameters, cv = splits, verbose=2, n_jobs=-1 )

# Ajuste de los datos

with warnings.catch_warnings(): #Catch conversion warnings
    warnings.simplefilter("ignore")
    grid.fit(Xres, Yres)


# Guardado del modelo para un uso más rápido en futuros momentos

Save( grid, saveName )



### Mostrado de los resultados finales ###


print( "Mejor valor de la cross validation: {:.4f}".format( grid.best_score_ ) )
print( "Mejores parámetros: {}".format( grid.best_params_ ) )


cls_nam = np.unique( y_train )
plot_confusion_matrix(
    confusion_matrix( y_test, grid.predict( X_test[:,indices] ) ),
    cls_nam,
    normalize = False
)
plt.show()

print()
print( "Valor en el test:" )
print( classification_report( y_test, grid.predict( X_test ) ) )
plot_ROC(X_test, y_test, grid, index_of_positive[0])
plt.show()

"""
####Impresion de los datos####
print("Mejor valor de la cross validation: {:.4f}".format(grid.best_score_))
print("Mejores parametros: {}".format(grid.best_params_))


plot_ROC_multiclass( X_test, y_test, grid )
plt.show()

plot_security( grid, X_train, y_train )
plt.show()

"""
