import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score,classification_report
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import joblib
# loading data
df=pd.read_csv('/storage/emulated/0/Download/InsuranceClaimCleaned.csv')
# numerical_columns
num_cols=['Age','PolicyDurationMonths',   'PreviousClaims','DaysToProcess',
   'Dependants','ClaimPremiumRatio',
   'ClaimToPremiumRatio','CoverageUtilisation',
   'PolicyMaturityYears','CoverageLimit_log','ClaimAmount_log',
   'MonthlyPremium_log']
# Categorical_columns
cat_cols=['State','Insurer',
   'PlanType','ClaimType','HospitalTier',
   'Occupation','RiskFlag']
# binary_columns
bina_cols=['Gender','PreExistingCondition',
   'DocumentationComplete',
   'HospitalAccredited','DeductiblePaid']
# Encoding Target
le = LabelEncoder() 
df['Gender']=le.fit_transform(df['Gender'])
df['ClaimStatusEncoded'] = le.fit_transform(df['ClaimStatus'])
X=df[num_cols + cat_cols + bina_cols]
Y=df['ClaimStatusEncoded']
# Transformers/pipeline
preprocessor=ColumnTransformer(transformers=[('Scaler', StandardScaler(), num_cols),('ohe', OneHotEncoder(drop='first',sparse_output=False,handle_unknown='ignore'), cat_cols)],remainder='passthrough')
# splitting
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42,stratify=Y)
#preprocessing
X_train=preprocessor.fit_transform(X_train)
X_test=preprocessor.transform(X_test)
# Handling Imbalance
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_res, Y_train_res = smote.fit_resample(X_train, Y_train)
#training
models = {
    "LogisticRegression": LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42,
        C=1.0),
    "DecisionTree": DecisionTreeClassifier(
        max_depth=10,
        class_weight='balanced',
        random_state=42),
    "RandomForest": RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1)}
results = {}
for name, model in models.items():
    model.fit(X_train_res, Y_train_res)
    y_pred = model.predict(X_test)
    acc = accuracy_score(Y_test, y_pred)
    results[name] = acc
    print(f"\n{name}: {acc*100:.2f}%")
    print(classification_report(
        Y_test, y_pred,zero_division=0,
        target_names=le.classes_
        ))
#feature importance
feature_names =preprocessor.get_feature_names_out()
importances = models['DecisionTree'].feature_importances_
feat_imp = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})
feat_imp = feat_imp.sort_values(by='Importance', ascending=False)
feat_imp['Original_Feature'] = feat_imp['Feature'].apply(
    lambda x: x.split('__')[-1].split('_')[0]
)
grouped = feat_imp.groupby('Original_Feature')['Importance'].sum().sort_values(ascending=False)

"""
# gridsearch
params={
  'class_weight': [None, 'balanced'],
  'max_depth': [5,7,12],
  #'criterion': ['gini', 'entropy'],
  'min_samples_split': [2, 5, 10],
    
}
grid=GridSearchCV(
    LogisticRegression(random_state=42),    
    params,
    scoring="f1_weighted", 
    cv=5
)
grid.fit(X_train_res, Y_train_res)
print("Best Params:", grid.best_params_)
print("Best Score:", grid.best_score_)
y_pred=grid.best_estimator(X_test_res)
acc = accuracy_score(Y_test_res, y_pred)
print(f"FineTuneLogisticRegre: {acc*100:.2f}%")
print(classification_report(Y_test_res, y_pred,zero_division=0,target_names=le.classes_))
"""
#saving models
joblib.dump(models['LogisticRegression'], '/storage/emulated/0/download/HealthInsuranceProject/LogisticRegre_model.joblib')
joblib.dump(models['DecisionTree'], '/storage/emulated/0/download/HealthInsuranceProject/DecisionTree_model.joblib')
joblib.dump(models['RandomForest'], '/storage/emulated/0/download/HealthInsuranceProject/RandomForest_model.joblib')
joblib.dump(le, '/storage/emulated/0/download/HealthInsuranceProject/label_encoder.joblib')
joblib.dump(preprocessor, '/storage/emulated/0/download/HealthInsuranceProject/PreProcessor.joblib')
print(grouped)
