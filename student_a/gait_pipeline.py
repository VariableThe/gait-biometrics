"""Prepare WISDM and UCI-HAR inertial data for gait/activity models.

The implementation follows the supplied papers' reusable preprocessing ideas:
timestamp-aware resampling, optional FIR-like low-pass filtering, sliding
windows, training-only standardization, and subject-safe splits.
"""
from __future__ import annotations
import argparse, json, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from scipy.interpolate import interp1d
    from scipy.signal import firwin, filtfilt
except Exception:
    interp1d = firwin = filtfilt = None

ROOT = Path(__file__).resolve().parent

def resample(values, timestamps, hz):
    t = np.asarray(timestamps, float); x = np.asarray(values, float)
    keep = np.isfinite(t) & np.all(np.isfinite(x), axis=1)
    t, x = t[keep], x[keep]
    order = np.argsort(t); t, x = t[order], x
    t, ix = np.unique(t, return_index=True); x = x[ix]
    if len(t) < 3: return x
    grid = np.arange(t[0], t[-1] + 1e-9, 1.0 / hz)
    return np.column_stack([np.interp(grid, t, x[:, j]) for j in range(x.shape[1])])

def lowpass(x, hz, cutoff=8.0):
    if filtfilt is None or firwin is None or len(x) < 32: return x
    taps = firwin(31, cutoff, fs=hz)
    return filtfilt(taps, [1.0], x, axis=0).astype(np.float32)

def windows(x, y, subject, size, step):
    out, labels, subs = [], [], []
    for start in range(0, max(0, len(x)-size+1), step):
        z = x[start:start+size]
        if len(z) != size: continue
        # majority label, with a purity score to flag boundary windows
        vals, counts = np.unique(y[start:start+size], return_counts=True)
        k = int(np.argmax(counts)); out.append(z); labels.append(vals[k]); subs.append(subject)
    return np.asarray(out, np.float32), np.asarray(labels), np.asarray(subs)

def save_npz(out, name, X, y, subjects, meta, stats=None):
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f'{name}.npz', X=X, y=y, subject=subjects)
    (out / f'{name}.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    if stats: np.savez(out / f'{name}_normalization.npz', **stats)

def process_wisdm(args, out):
    zpath = ROOT/'Dataset'/'WISDM_Dataset.zip'
    allsets = []
    with zipfile.ZipFile(zpath) as z:
        for split in ('train','test'):
            with z.open(f'{split}.csv') as f: df = pd.read_csv(f)
            cols = ['acc_x','acc_y','acc_z']; X = resample(df[cols].values, df['timestamp'].values, args.hz)
            X = lowpass(X, args.hz, args.cutoff)
            raw_y = df.Activity.to_numpy(); tt = df.timestamp.to_numpy(float)
            grid = np.linspace(tt.min(), tt.max(), len(X))
            y = raw_y[np.clip(np.searchsorted(tt, grid), 0, len(raw_y)-1)]
            # WISDM archive has no subject column; preserve source split as group.
            w, lab, sub = windows(X, y, split, args.window, args.step)
            allsets.append((split,w,lab,sub))
    train = next(v for v in allsets if v[0]=='train')[1]
    mu, sd = train.mean(axis=(0,1)), train.std(axis=(0,1)); sd[sd < 1e-6] = 1
    for split,w,y,s in allsets:
        save_npz(out, f'wisdm_{split}', ((w-mu)/sd).astype(np.float32), y,s,
                 {**vars(args), 'normalization':'train-only standard score'}, {'mean':mu,'std':sd})

def read_uci(z):
    with zipfile.ZipFile(z) as outer:
        nested = outer.read('UCI HAR Dataset.zip')
    import io
    with zipfile.ZipFile(io.BytesIO(nested)) as d:
        def txt(name): return np.loadtxt(d.open(name))
        Xtr, Xte = txt('UCI HAR Dataset/train/X_train.txt'), txt('UCI HAR Dataset/test/X_test.txt')
        ytr, yte = txt('UCI HAR Dataset/train/y_train.txt').astype(int).ravel(), txt('UCI HAR Dataset/test/y_test.txt').astype(int).ravel()
        str_, ste = txt('UCI HAR Dataset/train/subject_train.txt').astype(int).ravel(), txt('UCI HAR Dataset/test/subject_test.txt').astype(int).ravel()
    # Dataset already provides 128-sample windows x 561 engineered features.
    return (Xtr,ytr,str_), (Xte,yte,ste)

def process_uci(args, out):
    tr,te = read_uci(ROOT/'Dataset'/'human+activity+recognition+using+smartphones.zip')
    for name,(X,y,s) in [('uci_har_train',tr),('uci_har_test',te)]: save_npz(out,name,X.astype(np.float32),y,s,vars(args))

def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',default='processed')
    p.add_argument('--hz',type=float,default=50); p.add_argument('--window',type=int,default=90)
    p.add_argument('--step',type=int,default=45); p.add_argument('--cutoff',type=float,default=8)
    a=p.parse_args(); out=ROOT/a.out; process_wisdm(a,out); process_uci(a,out)
    print(json.dumps({'output':str(out), 'window_samples':a.window, 'step_samples':a.step}, indent=2))
if __name__=='__main__': main()
