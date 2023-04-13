import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

dataf = pd.read_excel('Thick_tile_GRADUATION.xlsx')#读取特征数据
dataf.iloc[:,dataf.columns!='Class'] = StandardScaler().fit_transform(dataf.iloc[:,dataf.columns!='Class'].values)#对特征进行标准化
X = dataf.iloc[:,dataf.columns!='Class']#提取特征
'''以下为PCA降维'''
cov_mat = np.cov(X.T)
#求特征值与特征向量
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
eig_pairs = [(np.abs(eig_vals[i]),eig_vecs[:,i]) for i in range(len(eig_vals))]
tot = sum(eig_vals)
var_exp = [(i/tot)*100 for i in sorted(eig_vals,reverse=True)]
cum_var_exp = np.cumsum(var_exp)
plt.rcParams['font.sans-serif']=['Arial Unicode MS']
plt.figure(figsize=(12,8))
plt.bar(range(8),var_exp[:8],alpha=0.5,align='center',label='individual explained variance')
plt.step(range(8),cum_var_exp[:8],where='mid',label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

matrix_w = np.concatenate([eig_pairs[0][1].reshape(-1,1),eig_pairs[1][1].reshape(-1,1),eig_pairs[2][1].reshape(-1,1)],axis=1)
X_pca = X.dot(matrix_w)
data_calss = dataf.iloc[:,dataf.columns=='Class']
dataf_pca= np.concatenate((X_pca,data_calss),axis =1)
X = X_pca#降维后的特征矩阵
y = data_calss#原标签

# normal_indices = data[data.Class == 0].index
# fraud_indics = data[data.Class == 1].index
# # data_fraud = data.iloc[fraud_indics,:]


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)#分训练集和测试集

sample_solver = SMOTE(random_state=0)#调用过采样函数
X_train_outsample,y_train_outsample = sample_solver.fit_resample(X_train,y_train)#对训练集进行过采样处理，使数据平衡
''' 以下为逻辑回归分类'''
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
# def printing_Kfold_scores(X_train_data,y_train_data):
#     fold = KFold(len(y_train_data),5,shuffle=False)   
fold = KFold(5,shuffle=False)
c_param_range=[0.001,0.1,10]
results_table=pd.DataFrame(columns=['C_parameter','Mean recall score'])
results_table['C_parameter']=c_param_range
j=0
for c_param in c_param_range:
    print('------------------------------------')
    print('正则化惩罚力度：',c_param)
    print('------------------------------------')
    print('')
    recall_accs=[]
    for train_index,test_index in fold.split(X_train_outsample):
        lr = LogisticRegression(C = c_param, penalty = 'l2')
        # Use the training data to fit the model. In this case, we use the portion of the fold to train the model
        lr.fit(X_train_outsample.iloc[train_index,:], y_train_outsample.iloc[train_index,:].values.ravel())
    
        # Predict values using the test indices in the training data
        y_pred_undersample = lr.predict(X_train_outsample.iloc[test_index,:].values)
    
        # Calculate the recall score and append it to a list for recall scores representing the current c_parameter
        recall_acc = recall_score(y_train_outsample.iloc[test_index,:].values,y_pred_undersample)
        recall_accs.append(recall_acc)
        print('召回率 = ', recall_acc)
y_pre=lr.predict(X_test.values)
recall_test=recall_score(y_test,y_pre)

def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    绘制混淆矩阵
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
import itertools
# lr = LogisticRegression(C = best_c, penalty = 'l1')


# 计算所需值
cnf_matrix = confusion_matrix(y_test,y_pre)
np.set_printoptions(precision=2)

print("召回率: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))

# 绘制
class_names = [0,1]
plt.rcParams['font.size']=30
plt.figure(figsize=(20,20))
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')
plt.show()