import pandas as pd
import sys


def input_fn(data_file,num_epoch,shuffle)
    dataStore= []
    resultStore = []
    CSV_COLUMNS = ["sex","length","diameter","height","whole_weight","shucked_weight","viscera_weight","shell_weight","rings"]
    df_train = pd.read_csv(sys.argv[1],names=CSV_COLUMNS,delimiter=' ')


    print(df_train)



def main():

    if len(sys.argv) != 2:
        print("Usage: ",sys.argv[0],"dataFile")
        exit()

    print("The name of the script is",sys.argv[0])


if __name__ == "__main__":main()
