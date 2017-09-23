import pandas as pd
import sys
import tensorflow as tf
import tempfile
import numpy as np


dataStore= []
resultStore = []


sex = tf.feature_column.categorical_column_with_vocabulary_list("sex",["M","F","I"])
length = tf.feature_column.numeric_column("length")
diameter = tf.feature_column.numeric_column("diameter")
height = tf.feature_column.numeric_column("height")
whole_weight = tf.feature_column.numeric_column("whole_weight")
shucked_weight = tf.feature_column.numeric_column("shucked_weight")
viscera_weight = tf.feature_column.numeric_column("viscera_weight")
shell_weight = tf.feature_column.numeric_column("shell_weight")
#rings = tf.feature_column.numeric_column("rings")

base_columns = [sex,length,diameter,height,whole_weight,shucked_weight,viscera_weight,shell_weight]




def input_fn(data_file,num_epochs,shuffle,index,train):
    CSV_COLUMNS = ["sex","length","diameter","height","whole_weight","shucked_weight","viscera_weight","shell_weight","rings"]
    df_split = pd.read_csv(sys.argv[1],names=CSV_COLUMNS,delimiter=' ')
    df_split = df_split.dropna(how="any", axis=0)

    df = np.array_split(df_split,5)
    
    df_test = df[index]

    if index == 0:
        df_train = df[1]
        for i in range(2,5):
            df_train = pd.concat((df_train,df[i]),axis=0)
    else:
        df_train = df[0]
        for i in range(1,5):
            if i != index:
                df_train = pd.concat((df_train,df[i]),axis=0 )

    if( train == 1):
        labels = df_train["rings"].astype(float) 
        print(labels)
        return tf.estimator.inputs.pandas_input_fn(x=df_train,
                y=labels,
                batch_size=100,
                num_epochs=num_epochs,
                shuffle=shuffle,
                num_threads=5)

    else:
        labels = df_test["rings"].astype(float) 
        print(labels)
        return tf.estimator.inputs.pandas_input_fn(x=df_test,
                y=labels,
                batch_size=100,
                num_epochs=num_epochs,
                shuffle=shuffle,
                num_threads=5)

    #print(df_train)
    #print(df_train)



def build_estimator(model_dir):
    #m = tf.contrib.learn.LinearClassifier(model_dir,feature_columns=base_columns)
    m = tf.estimator.LinearRegressor(feature_columns=base_columns)
    #m = tf.estimator.LinearRegressor(model_dir,feature_columns=base_columns)
    return m


def main():

    if len(sys.argv) != 2:
        print("Usage: ",sys.argv[0],"dataFile")
        #exit()
    print("The name of the script is",sys.argv[0])
    model_dir = tempfile.mkdtemp()
    m = build_estimator(model_dir)
    print("Training started")
    m.train(input_fn=input_fn(sys.argv[1],num_epochs=None,shuffle=True,index=1,train = 1),
            steps=None)
    print("Training done")
    print("model_director = %s" %model_dir)
    #print("Training started")
    #m.train(input_fn=input_fn(sys.argv[1],num_epochs=None,shuffle=True,index=1,train = 0),
    #        steps=None)
    
    #for key in results:
    #    print("%s: %s" %(key,results[key]))


if __name__ == "__main__":main()
