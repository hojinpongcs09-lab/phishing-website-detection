"""明確執行才評估測試集，使用既有模型，不重新調參。"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,roc_auc_score,average_precision_score
r=Path(__file__).resolve().parents[1]; base=r/'results/holdout80'
bundle=joblib.load(base/'model.joblib'); previous=json.loads((base/'metrics.json').read_text())
df=pd.read_csv(r/'data/phishing_dataset.csv'); X=df[bundle['features']]; y=1-df.Result
split=pd.read_csv(base/'splits.csv'); train=split.loc[split.split=='train','row_index'].to_numpy(); test=split.loc[split.split=='test','row_index'].to_numpy()
assert not previous['test_evaluated']
groups=pd.factorize(pd.MultiIndex.from_frame(X),sort=True)[0]
assert not set(groups[train])&set(groups[test])
assert bundle['threshold']==previous['threshold']
dummy=DummyClassifier(strategy='most_frequent').fit(X.iloc[train],y.iloc[train])
report={'note':'Fixed-model re-evaluation; test set previously inspected in exploratory experiments.', 'training_rows':len(train),'test_rows':len(test),'threshold':bundle['threshold'],'best_C':previous['best_C'],'scaling':previous['scaling']}
for name,model in [('baseline',dummy),('logistic_regression',bundle['pipeline'])]:
 p=model.predict_proba(X.iloc[test])[:,list(model.classes_).index(1)]
 pred=model.predict(X.iloc[test]) if name=='baseline' else p>=bundle['threshold']
 tn,fp,fn,tp=confusion_matrix(y.iloc[test],pred,labels=[0,1]).ravel()
 report[name]={'accuracy':accuracy_score(y.iloc[test],pred),'precision':precision_score(y.iloc[test],pred,zero_division=0),'recall':recall_score(y.iloc[test],pred,zero_division=0),'f1':f1_score(y.iloc[test],pred,zero_division=0),'fpr':float(fp/(fp+tn)),'roc_auc':roc_auc_score(y.iloc[test],p),'average_precision':average_precision_score(y.iloc[test],p),'confusion_matrix':[[int(tn),int(fp)],[int(fn),int(tp)]]}
out=r/'results/final_comparison'; out.mkdir(exist_ok=True)
(out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
