import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import numpy as np
import csv
from sklearn.model_selection import GridSearchCV

#csvファイルをロード
def file_import():
    train=pd.read_csv(filepath_or_buffer="/Users/unokenta/kaggle/titanic/all/train.csv",encoding="ms932",sep=",")
    test=pd.read_csv(filepath_or_buffer="/Users/unokenta/kaggle/titanic/all/test.csv",encoding="ms932",sep=",")
    return train,test

#結果書き出し
def write_csv(result_list):
    with open("result.csv","w") as f:
        writer = csv.writer(f,lineterminator="\n")
        writer.writerow(["PassengerId","Survived"])
        writer.writerows(result_list)

#文字を数字に変換
def preprosess(file_list):
    
    file_list.loc[file_list["Sex"]=="male","Sex"] = 0
    file_list.loc[file_list["Sex"]=="female","Sex"] = 1

    #欠損値含む行を削除
    #file_list=file_list.dropna(how="any")

    #欠損値を平均値(mean)・中央値(median)・最頻値(mode().iloc[0])で置換
    file_list=file_list.fillna(file_list.median())
    
    return file_list


#不要な説明変数を削除
def rem_item(file_list):
    new_list=file_list.drop(columns=["PassengerId","Name","Ticket","Cabin","Embarked","Fare","Parch"])
    return new_list

#説明変数Xと目的変数Yに分離
def separate_XY(file_list):
    X=file_list.iloc[:,1:]
    Y=file_list.iloc[:,:1].values.flatten()

    return X,Y

#クロスバリデーション
def cross_validation(X,Y):
    clf=SVC()
    scores=cross_val_score(clf, X, Y, cv=10)
    print("cross-validation:{}".format(scores))
    print("average score:{}".format(np.mean(scores)))

#パラメータチューニング
def para_tune(X,Y):
    svc_param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1, 10, 100]
    }

    svc_grid_search = GridSearchCV(SVC(), svc_param_grid, cv=10)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8,shuffle=True)
    svc_grid_search.fit(X_train, Y_train)

    print('Train score: {:.3f}'.format(svc_grid_search.score(X_train, Y_train)))
    print('Test score: {:.3f}'.format(svc_grid_search.score(X_test, Y_test)))

    print('Best parameters: {}'.format(svc_grid_search.best_params_))
    print('Best estimator: {}'.format(svc_grid_search.best_estimator_))

#分類
def classifier(Test,clf):
    test_predict=clf.predict(Test)
    return test_predict

#予測csv出力
def predict(test_file,clf):
    test=rem_item(test_file)
    test=preprosess(test)
    test_file = test_file["PassengerId"].values
    result=classifier(test,clf)
    result_list=[]
    
    for res,tes in zip(result,test_s):
        meta=[]
        meta.append(tes)
        meta.append(res)
        result_list.append(meta)

    write_csv(result_list)


#メイン
if __name__ == "__main__":

    train,test_s=file_import()

    train=rem_item(train)
    
    train=preprosess(train)
    
    X,Y=separate_XY(train)

    #パラメータチューニング
    para_tune(X,Y)
    
    #クロスバリデーション
    cross_validation(X,Y)

    #モデル学習
    #clf=SVC(C=10,gamma=0.01)
    #clf.fit(X,Y)

    #予測
    #predict(test_s,clf)

    
    """
    svc=SVC()
    svc.fit(X_train,y_train)
    print('Train score: {:.3f}'.format(svc.score(X_train, y_train)))
    print('Test score: {:.3f}'.format(svc.score(X_test, y_test)))
    """
