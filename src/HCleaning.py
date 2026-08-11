import numpy as np
import pandas as pd
#Loading data
df = pd.read_csv("/storage/emulated/0/Download/nigerian_health_insurance_messy.csv")
# duplicates
df = df.drop_duplicates()

df['State'] = df['State'].astype(str).str.strip().str.capitalize()
# PlanType
df['PlanType']=df['PlanType'].astype(str).str.strip()
def plan_corrector(c):
    corrector={
    'HMO Premium' : ["Premium HMO",'HMO Prem','HMO-Premium','HMO Premium'],
    'HMO Basic' : ['HMO_Basic','hmo basic','HMO basic','Basic HMO','HMO-Basic','HMO Basic'],
    'PPO Standard' : ['Standard PPO','PPO-Standard','ppo standard'],
    'HMO Standard' : ['HMO-Standard','Standard HMO','HMO std','hmo standard','HMO Standard'],
    'Individual Plan' : ['individual','Indv Plan','INDIVIDUAL','Individual'],
    'Group Plan' : ['GROUP','Group Policy','group plan','Group','Grp Plan','Group Plan'],
    'PPO Basic' : ['PPO-Basic','Basic PPO','PPO basic','ppo basic','PPO Basic']
    }
    for actuals,variants in corrector.items():
        if c in variants:
            return actuals      
    return np.nan
df['PlanType'] = df['PlanType'].apply(plan_corrector)
df['PlanType'] = df['PlanType'].fillna(df.groupby(['State','Insurer'])['PlanType'].transform(lambda x: x.mode()[0]))
# ClaimType
df['ClaimType'] = df['ClaimType'].astype(str).str.strip()
def claim_corrector(c):
    corrector={
    'Optical' : ['Optical'],
    'Laboratory' : ['Laboratory'],
    'Specialist' : ['Specialist'],
    'Dental' : ['Dental'],
    'Pharmacy' : ['Pharmacy'],
    'Emergency' : ['Emergency','Emerg','ER','E/R','EMERGENCY'],
    'Inpatient' : ['Inpatient','In Patient','In-patient','IP','IPD'],
    'Outpatient' : ['Outpatient','Out Patient','Out-patient','OP','OPD'],
    'Maternity' : ['Maternity','maternal','MCH','MATERNITY'],
    'Surgery' : ['Surgery','SURGERY','Surgical','Operation']
    }
    for actuals,variants in corrector.items():
        if c in variants:
            return actuals
    return np.nan       
df['ClaimType'] = df['ClaimType'].apply(claim_corrector)
df['ClaimType'] = df['ClaimType'].fillna(df.groupby(['Insurer','PlanType'])['ClaimType'].transform(lambda x: x.mode()[0]))
# HospitalTier
df['HospitalTier'] = df['HospitalTier'].astype(str).str.strip().str.capitalize()
def tier_corrector(c):
    T1={'Tier1','Tier-1','Level1',
    'Level 1','T1'}
    T2={'Tier2','Tier-2','Level2',
    'Level 2','T2'}
    T3={'Tier3','Tier-3','Level3',
    'Level 3','T3'}
    if c in T1:
        return 'Tier 1'
    elif c in T2:
        return 'Tier 2'
    elif c in T3:
        return 'Tier 3'
    else:
        return c
df['HospitalTier'] = df['HospitalTier'].apply(tier_corrector)
# Occupation
df['Occupation']=df['Occupation'].astype(str).str.strip().str.capitalize()
df['Occupation']=df['Occupation'].apply(lambda x: np.nan if x=='Nan' else x)
df['Occupation'] = df['Occupation'].fillna(df.groupby(['HospitalTier','PlanType','Insurer'])['Occupation'].transform(lambda x: x.mode()[0]))
# Gender
df['Gender'] = df['Gender'].astype(str).str.strip().str.capitalize()
def gender_corrector(c):
    corrector={
    'Male':['Male','Masculine','1',
    'M','Man','M.','Mr','Males'],
    'Female':['Female','0','F','Mrs',
    'Females','Ms','F.','Woman','Feminine']}
    for actuals, variants in corrector.items():
        if c in variants:
            return actuals
    return np.nan
df['Gender'] = df['Gender'].apply(gender_corrector)
# Handling outlier function
def handle_outliers(s):
    Q1 = s.quantile(0.25)
    Q3 = s.quantile(0.75)
    IQR = Q3 - Q1
    Min = Q1 - 1.5 * IQR
    Max = Q3 + 1.5 * IQR

    s = s.clip(Min, Max)    
    return s
df['Age'] = df['Age'].astype(str).str.strip().str.capitalize()
def age_corrector(c):
    corrector=['yrs','years']
    for x in corrector:
        if x in str(c):
            return c.replace(x,'')
    return c
df['Age'] = df['Age'].apply(age_corrector)
df['Age'] = pd.to_numeric(df['Age'],errors='coerce')
df['Age']=handle_outliers(df['Age'])
df['Age'] = df['Age'].fillna(df.groupby(['Gender','Occupation'])['Age'].transform('mean'))
df['Age']=df['Age'].astype(int)
# PolicyDurationMonth
df['PolicyDurationMonths'] = df['PolicyDurationMonths'].astype(str).str.strip()
df['PolicyDurationMonths']=df['PolicyDurationMonths'].apply(
lambda x: str(x).replace('s','')
                .replace('month','')
                .replace('approx','')
                .replace('mnth','')
                .replace('mth','')                    .strip())
def duration_corrector(c):
    if 'year' in c:
        k=c.index('y')
        m=c.index('r')
        return float(c[:k])*12+float(c[m+1:])
    return c
df['PolicyDurationMonths'] = df['PolicyDurationMonths'].apply(duration_corrector)
df['PolicyDurationMonths'] = pd.to_numeric(df['PolicyDurationMonths'],errors='coerce')
df['PolicyDurationMonths'] = handle_outliers(df['PolicyDurationMonths'])
df['PolicyDurationMonths'] = df['PolicyDurationMonths'].fillna(df.groupby(['PlanType','Occupation'])['PolicyDurationMonths'].transform(lambda x: x.mode()[0]))
df['PolicyDurationMonths']=df['PolicyDurationMonths'].astype(int)

def premium_claim_corrector(c):
    variants=['NGN','naira','₦','#',',','/month','"','N','-','Limit:']
    c=str(c)
    for x in variants:
        if x in c:
            c=c.replace(x,'')
    c=c.strip()
    for k in ['k','K']:
        if k in c:
            try:
                return float(c.replace(k,'')) * 1000
            except:
                return np.nan
    try:
        return float(c)
    except:
        return np.nan  
# ClaimAmount #MonthlyPremium #CovergageLimit
for c in ['MonthlyPremium','ClaimAmount','CoverageLimit']:
    df[c] = df[c].apply(premium_claim_corrector)
    df[c] = handle_outliers(df[c]) 
df['MonthlyPremium'] = df['MonthlyPremium'].fillna(df.groupby(['HospitalTier','PlanType'])['MonthlyPremium'].transform('mean'))
df['MonthlyPremium'] = df['MonthlyPremium'].round(2)
df['ClaimAmount'] = df['ClaimAmount'].fillna(df.groupby(['PlanType','ClaimType'])['ClaimAmount'].transform('mean'))
df['ClaimAmount'] = df['ClaimAmount'].round(2)
df['CoverageLimit'] = (df['CoverageLimit'].fillna(df.groupby(['ClaimType','HospitalTier'])['CoverageLimit'].transform('mean'))).round(2)
#PreviousClaim
df['PreviousClaims'] = df['PreviousClaims'].fillna(df['PreviousClaims'].median())
df['PreviousClaims'] = df['PreviousClaims'].astype(int)
# DaysToProccess
df['DaysToProcess'] = df['DaysToProcess'].astype(str).str.strip()
def Days_toP_corrector(c):
    variants=['day(s)','days','pending','-','"']
    c=str(c)
    for x in variants:
        if x in c:
            c=c.replace(x,'')
    c=c.strip()
    return c
df['DaysToProcess'] = df['DaysToProcess'].apply(Days_toP_corrector)
df['DaysToProcess'] = pd.to_numeric(df['DaysToProcess'],errors='coerce')
df['DaysToProcess'] = handle_outliers(df['DaysToProcess'])
df['DaysToProcess'] = df['DaysToProcess'].fillna(df.groupby("Insurer")['DaysToProcess'].transform('median'))
df['DaysToProcess'] = (df['DaysToProcess']).astype(int)
#Dependants
df['Dependants'] = pd.to_numeric(df['Dependants'],errors='coerce')
df['Dependants'] = df['Dependants'].fillna(df.groupby('PlanType')['Dependants'].transform(lambda x: x.mode()[0]))
df['Dependants'] = (df['Dependants']).astype(int)
#PreExistingCondition,DocumentationComplete,HospitalAccredited
for c in ['PreExistingCondition','DocumentationComplete','HospitalAccredited','DeductiblePaid']:
    df[c] = df[c].fillna(df.groupby(['PlanType','ClaimType','HospitalTier'])[c].transform(lambda x: x.mode()[0]))
    df[c] = df[c].astype(int)
df['ClaimPremiumRatio'] = df['ClaimPremiumRatio'].astype(str).str.strip()
def claim_ratio_corrector(c):
    c=str(c)
    variants = ['x','ratio',':1',':']
    for x in variants:
        if x in c:
            c=c.replace(x,'')
    c=c.strip()
    return c
df['ClaimPremiumRatio'] = df['ClaimPremiumRatio'].apply(claim_ratio_corrector)
df['ClaimPremiumRatio'] = pd.to_numeric(df['ClaimPremiumRatio'], errors='coerce')
df['ClaimPremiumRatio'] = handle_outliers(df['ClaimPremiumRatio'])
df['ClaimPremiumRatio'] = df['ClaimPremiumRatio'].fillna(df.groupby(['Insurer','HospitalTier'])['ClaimPremiumRatio'].transform('mean'))
df['ClaimPremiumRatio'] = (df['ClaimPremiumRatio']).round(2)
# ClamStatus
df['ClaimStatus'] = df['ClaimStatus'].astype(str).str.strip()
approved = [
    'Approved','approved','APPROVED',
    'App','app','1','Accept','accept'
    'Accepted','accepted','ACCEPTED',
    'A','a','Approve','approve','Yes'
    'yes','YES','Settled','settled',
    'SETTLED','Paid','paid','PAID']
pending = [
    'Pending','pending','PENDING',
    'Pend','pend','P','p','2',
    'In Review','in review','IN REVIEW',
    'Under Review','under review',
    'UNDER REVIEW','Processing','processing','PROCESSING','On Hold',
    'on hold','ON HOLD','Awaiting',
    'awaiting','Review','review','P1'
    ]
rejected = [
    'Rejected','rejected','REJECTED',
    'Rej','rej','R','r','3','Denied',
    'denied','DENIED','Decline'
    'decline','Declined','declined',
    'No','no','NO','Not Approved','not approved','NOT APPROVED',
    'Not approved','Failed','failed','FAILED'
    ]
fraudulent = [
    'Fraudulent','fraudulent',
    'FRAUDULENT','Fraud','fraud',
    'FRAUD','F','f','4','Suspicious',
    'suspicious','SUSPICIOUS',
    'Flagged','flagged','FLAGGED',
    'Invalid','invalid','INVALID',
    'Blacklisted','blacklisted','F1',
    'f1'
    ]
def clean_status(val):
    val = str(val).strip()
    if val in approved:
        return 'Approved'
    if val in pending:
        return 'Pending'
    if val in rejected:
        return 'Rejected'
    if val in fraudulent:
        return 'Fraudulent'
    return np.nan 
df['ClaimStatus'] = df['ClaimStatus'].apply(clean_status)
df['ClaimStatus'] = df['ClaimStatus'].fillna(df.groupby(['PreExistingCondition','DocumentationComplete','HospitalAccredited'])['ClaimStatus'].transform(lambda x: x.mode()[0]))
# Feature Engineering
df['ClaimToPremiumRatio'] = (df['ClaimAmount'] /(df['MonthlyPremium'] * 12 + 1)).round(3)
# Coverage utilisation
df['CoverageUtilisation'] = (df['ClaimAmount'] /(df['CoverageLimit'] + 1)).round(3)
# Risk flag 
df['RiskFlag'] = ((df['PreviousClaims'] > 5).astype(int) + (df['ClaimToPremiumRatio'] > 10).astype(int) +(df['DocumentationComplete'] == 0).astype(int) + (df['DeductiblePaid'] == 0).astype(int))
# Policy maturity
df['PolicyMaturityYears'] = (df['PolicyDurationMonths'] / 12).round(1)
for c in ['CoverageLimit','ClaimAmount','MonthlyPremium']:
    df[c+'_log']=np.log1p(df[c])
    df[c+'_log'] = (df[c+'_log'].round(4))
    df=df.drop(columns=[c])
df = df.drop(columns=['ClaimID'])
df.to_csv('/storage/emulated/0/Download/InsuranceClaimCleaned.csv',index='false')
