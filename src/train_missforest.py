"""MissForest-style imputation (IterativeImputer+RandomForest, fit on train subsample, applied to all).
Same patient-level split, same base features + missing indicators, + pulse_delta cat variant.
Saves prep_mf.npz for retraining CatBoost/XGB to compare against median version.
"""
from pathlib import Path
import numpy as np, pandas as pd, time
from sklearn.impute import MissingIndicator
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

DATA=Path("/sessions/funny-happy-fermi/mnt/Inhospital adverse event/data_321381.csv")
OUT="/sessions/funny-happy-fermi/mnt/outputs/prep_mf.npz"; RS=42
DROP=["Unnamed: 0","LOC","DTYM","IDCODE","OPDNO","CSN","IDOPD","IDCSN","TMP_d","PULSE_d","SBP_d","DBP_d","RR_d","shock_index_d","GCSE_d","GCSV_d","GCSM_d","internal.medicine","cardiac_arrest","vent_2day","death_2day","icu_2day","cpr_2day","adverse_event","split"]

df=pd.read_csv(DATA,low_memory=False); df=df.loc[df["AGE"]>=65].copy()
df["SEX"]=np.where(df["SEX"]=="M",1,0)
df["adverse_event"]=((df["death_2day"].fillna(0)==1)|(df["icu_2day"].fillna(0)==1)|(df["vent_2day"].fillna(0)==1)).astype(int)
pool=df[df["LOC"]!=6].copy(); ext=df[df["LOC"]==6].copy()
pids=pool["IDCODE"].dropna().unique()
tr,tmp=train_test_split(pids,test_size=0.2,random_state=RS); va,ic=train_test_split(tmp,test_size=0.5,random_state=RS)
tr,va,ic=set(tr),set(va),set(ic)
pool["split"]=pool["IDCODE"].map(lambda x:"train" if x in tr else ("val" if x in va else "internal_test"))
ext["split"]="external_test"; full=pd.concat([pool,ext],ignore_index=True)
y=full["adverse_event"].astype(int).values
X=full.drop(columns=[c for c in DROP if c in full.columns],errors="ignore")
for c in X.columns: X[c]=pd.to_numeric(X[c],errors="coerce")
sp=full["split"].values
def sub(k): return X.loc[sp==k].reset_index(drop=True)
Xtr,Xva,Xin,Xex=sub("train"),sub("val"),sub("internal_test"),sub("external_test")
ytr,yva,yin,yex=y[sp=="train"],y[sp=="val"],y[sp=="internal_test"],y[sp=="external_test"]

def add_pd(x):
    x=x.copy(); x["PULSE_delta_v1"]=pd.to_numeric(x.get("PULSE_in"),errors="coerce")-pd.to_numeric(x.get("PULSE_a"),errors="coerce"); return x

def preprocess_mf(xtr,xva,xin,xex,tag):
    cols=list(xtr.columns)
    ind=MissingIndicator(features="missing-only",sparse=False); itr=ind.fit_transform(xtr)
    icols=[f"missing__{c}" for c in xtr.columns[ind.features_]]
    def merge(x,i): return pd.concat([x.reset_index(drop=True),pd.DataFrame(i,columns=icols)],axis=1)
    xtr2=merge(xtr,itr); xva2=merge(xva,ind.transform(xva)); xin2=merge(xin,ind.transform(xin)); xex2=merge(xex,ind.transform(xex))
    # MissForest: IterativeImputer + RF, fit on TRAIN subsample, applied unchanged
    t=time.time()
    imp=IterativeImputer(estimator=RandomForestRegressor(n_estimators=10,max_depth=12,n_jobs=-1,random_state=RS),
                         max_iter=2,random_state=RS)
    rng=np.random.default_rng(RS); sidx=rng.choice(len(xtr2),min(6000,len(xtr2)),replace=False)
    imp.fit(xtr2[cols].iloc[sidx])
    for xx in (xtr2,xva2,xin2,xex2): xx[cols]=imp.transform(xx[cols])
    feats=list(xtr2.columns)
    print(f"  [{tag}] MissForest fit+transform: {time.time()-t:.1f}s, nan_left={int(np.isnan(xtr2[cols].to_numpy(float)).sum())}")
    import joblib
    joblib.dump({"indicator":ind,"imputer":imp,"impute_cols":cols,"feature_order":feats,"icols":icols},
                f"/sessions/funny-happy-fermi/mnt/outputs/missforest_pipeline_{tag}.joblib")
    return (xtr2[feats].to_numpy(),xva2[feats].to_numpy(),xin2[feats].to_numpy(),xex2[feats].to_numpy(),feats)

cat=preprocess_mf(add_pd(Xtr),add_pd(Xva),add_pd(Xin),add_pd(Xex),"cat")
np.savez_compressed(OUT, Xtr_c=cat[0],Xva_c=cat[1],Xin_c=cat[2],Xex_c=cat[3],feats_c=np.array(cat[4],dtype=object),
                    ytr=ytr,yva=yva,yin=yin,yex=yex)
print("SAVED",OUT,"| n_features(cat)=",len(cat[4]))
