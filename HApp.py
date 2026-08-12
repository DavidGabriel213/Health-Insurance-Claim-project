from flask import Flask, render_template, request
import numpy as np 
import pandas as pd
import joblib
import os
#loading model and preprocessor
model=joblib.load('/storage/emulated/0/Download/HealthInsuranceProject/DecisionTree_model.joblib')
preprocessor=joblib.load('/storage/emulated/0/Download/HealthInsuranceProject/PreProcessor.joblib')
app=Flask(__name__)
@app.route('/',methods=['GET','POST'])
def myfunc():
    category=None
    Category_Class=None
    if request.method=='POST':
        # numerical features
        Age = float(request.form['age'])
        Duration = float(request.form['duration'])
        PreviousClaims =float(request.form['previous_claim'])
        DaysToProcess = float(request.form['days'])
        CoverageLimit = float(request.form['coverage_limit'])
        ClaimAmount = float(request.form['claim_amount'])
        MonthlyPremium = float(request.form['monthly_premium'])
        ClaimPremiumRatio = float(request.form['claim_ratio'])
        Dependants = float(request.form['dependants'])
        # Categorical columns
        State = request.form['state']
        Insurer = request.form['insurer']
        PlanType = request.form['plan_type']
        ClaimType = request.form['claim_type']
        HospitalTier = request.form['hospital_tier']
        Occupation = request.form['occupation']
        # Binary columns
        Gender = float(request.form['gender'])
        PreExistingCondition = float(request.form['condition'])
        DocumentationComplete = float(request.form['documentation_complete'])
        HospitalAccredited = float(request.form['hospital_credited'])
        DeductiblePaid = float(request.form['deductible_paid'])
        #Engineering
        CoverageLimit_log = np.log1p(CoverageLimit)
        ClaimAmount_log = np.log1p(ClaimAmount)
        MonthlyPremium_log = np.log1p(MonthlyPremium)
        ClaimToPremiumRatio = (ClaimAmount /(MonthlyPremium * 12 + 1))
        CoverageUtilisation = (ClaimAmount /(CoverageLimit + 1))
        RiskFlag = int((PreviousClaims > 5) + (ClaimToPremiumRatio > 10) +(DocumentationComplete == 0) + (DeductiblePaid == 0))
        PolicyMaturityYears = (Duration / 12)
        #features
        features = pd.DataFrame({
    'Age':[Age],
    'PolicyDurationMonths':[Duration],   
    'PreviousClaims':[PreviousClaims],
    'DaysToProcess':[DaysToProcess],
    'Dependants':[Dependants],'ClaimPremiumRatio':[ClaimPremiumRatio],'ClaimToPremiumRatio':[ClaimToPremiumRatio],'CoverageUtilisation':[CoverageUtilisation],'PolicyMaturityYears':[PolicyMaturityYears],'CoverageLimit_log':[CoverageLimit_log],'ClaimAmount_log':[ClaimAmount_log],'MonthlyPremium_log':[MonthlyPremium_log],'State':[State],'Insurer':[Insurer],'PlanType':[PlanType],'ClaimType':[ClaimType],'HospitalTier':[HospitalTier],'Occupation':[Occupation],'RiskFlag':[RiskFlag],'Gender':[Gender],'PreExistingCondition':[PreExistingCondition],'DocumentationComplete':[DocumentationComplete],'HospitalAccredited':[HospitalAccredited],'DeductiblePaid':[DeductiblePaid]        })
    #preprocessing and prediction
        FEATURES=preprocessor.transform(features)
        prediction=model.predict(FEATURES)[0]
        if prediction==0:
            category='Approved'
        elif prediction==1:
            category='Fraudulent'
        elif prediction==2:
            category='Pending'
        else:
            category='Rejected'
    class_map = {
       "Approved":"result-approved",
       "Fraudulent":"result-fraudulent",
       "Pending":"result-pending",
       "Rejected":"result-rejected"}
    Category_class = class_map.get(category, "")
    return render_template("insurance.html", category=category, Category_class=Category_class)
if __name__==('__main__'):
    port =int(os.environ.get('PORT',5000))
    app.run(host='0.0.0.0',port=port,debug=True)