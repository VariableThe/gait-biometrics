"""Lossless storage, fit-free replay and exact artifact/report bindings."""
from pathlib import Path
import os
import tempfile
import zipfile
import encodings.cp437  # NPZ member names load this codec; identical closure in every stage.
import numpy as np
from numpy.lib import format as numpy_format
import common as c
import guards
import kernel
import panel

TRUST = "Content/procedural integrity under cooperative local execution; no OS sandbox or absolute process authenticity claim. A writable tree can be forged; hashes are not signatures."


def specs(n):
    return dict(raw=((n,6,446),"<f8"),magnitudes=((n,2,446),"<f8"),filtered=((n,2,446),"<f8"),
        correlations=((n,3,121),"<f8"),correlation_stats=((n,2,121,2,2),"<f8"),
        correlation_valid=((n,2,121,2),"|b1"),correlation_finite=((n,2,121,2),"|b1"),
        coordinates=((n,200),"<f8"),lower=((n,200),"<i8"),upper=((n,200),"<i8"),fraction=((n,200),"<f8"),
        samples=((n,2,200),"<f8"),map_stats=((n,2,2),"<f8"),map_means=((n,2),"<f8"),
        map_valid=((n,2),"|b1"),map_finite=((n,2),"|b1"),mapped=((n,),"|b1"),
        features=((n,400),"<f8"),valid=((n,),"|b1"),selected_lag=((n,),"<i8"),selected_pair=((n,2),"<i8"))


def schema(n):
    return {k:dict(shape=list(shape),dtype=dtype) for k,(shape,dtype) in specs(n).items()}


def allocate(n):
    return {k:np.full(shape,np.nan if np.dtype(dtype).kind=="f" else False if dtype=="|b1" else -1,dtype=dtype)
            for k,(shape,dtype) in specs(n).items()}


def store(arrays,index,record,data,mapped):
    for name,values in data.items():
        arrays[name][index]=values
    arrays["valid"][index]=record["valid"]
    arrays["selected_lag"][index]=-1 if record["selected_lag"] is None else record["selected_lag"]
    arrays["selected_pair"][index]=[-1,-1] if record["selected_pair"] is None else record["selected_pair"]
    if mapped is None:
        return
    arrays["mapped"][index]=True
    for name in ("coordinates","lower","upper","fraction","samples"):
        arrays[name][index]=mapped[name]
    for name in ("stats","means","valid","finite"):
        arrays["map_"+name][index]=mapped[name]
    if mapped["feature"] is not None:
        arrays["features"][index]=mapped["feature"]


def bits_equal(left,right,label):
    c.require(left.shape==right.shape and left.dtype==right.dtype and left.tobytes()==right.tobytes(),"Lossless mismatch: "+label)


def validate_shapes(arrays,n):
    expected=specs(n)
    c.require(set(arrays)==set(expected),"Exact array names")
    for name,(shape,dtype) in expected.items():
        c.require(arrays[name].shape==shape and arrays[name].dtype==np.dtype(dtype),"Array schema: "+name)


def gates(records,arrays,inventory,fixed_checks_pass):
    c.require(len(records)==len(inventory),"Denominator preserved")
    nondeg=[i for i,r in enumerate(inventory) if r["kind"]=="legacy" and r["family"]!=panel.legacy.FAMILIES[3]]
    balanced=[i for i in nondeg if inventory[i]["family"]==panel.legacy.FAMILIES[2] and inventory[i]["period_samples"] in (160,180,200)]
    invalid=[i for i,r in enumerate(inventory) if (r["kind"]=="legacy" and r["family"]==panel.legacy.FAMILIES[3]) or
             (r["kind"]=="stress" and r["condition"] in panel.CONDITIONS[5:7])]
    c.require(len(inventory)==1440 and len(nondeg)==960 and len(balanced)==96 and len(invalid)==344,"Gate denominators")
    legacy_fail=[i for i in nondeg if not records[i]["valid"] or records[i]["lag_error"] is None or abs(records[i]["lag_error"])>1 or records[i]["duration_error"] is None or abs(records[i]["duration_error"])>1]
    invalid_fail=[i for i in invalid if records[i]["valid"]]
    mapping_fail=[i for i,r in enumerate(records) if r["valid"] and not r["mapping_verification"]["pass_verification_only"]]
    base={(r["family"],r["period_samples"],r["phase_index"]):i for i,r in enumerate(inventory) if r["kind"]=="legacy"}
    transformations=[]
    for i,item in enumerate(inventory):
        if item["kind"]!="invariance": continue
        j=base[(item["family"],160,item["phase_index"])]
        a,b=records[i],records[j]
        unchanged=all(a[k]==b[k] for k in ("valid","selected_lag","selected_pair"))
        delta=float(np.max(np.abs(arrays["features"][i]-arrays["features"][j]))) if a["valid"] and b["valid"] else None
        transformations.append(dict(input_index=i,base_index=j,decisions_unchanged=unchanged,
            both_valid=a["valid"] and b["valid"],feature_max_abs_delta=delta,
            pass_gate=bool(unchanged and (delta is None or delta<=1e-10))))
    transform_fail=[r["input_index"] for r in transformations if not r["pass_gate"]]
    results=dict(fixed_numerical_mapping_checks=bool(fixed_checks_pass),nondegenerate_legacy_pass=not legacy_fail,
        all_weak_and_constant_stress_invalid=not invalid_fail,transforms_pass=not transform_fail,mapping_verification_pass=not mapping_fail)
    passed=all(results.values())
    # Noise diagnostics are unconditional even if an earlier gate failed.
    stress=[dict(input_index=i,condition=r["condition"],valid=r["valid"],failure=r["failure"],
             lag=r["selected_lag"],duration=r["duration"],lag_error=r["lag_error"],duration_error=r["duration_error"],
             lag_alias=r["lag_alias"],duration_alias=r["duration_alias"],noise_only_false_valid=r["noise_only_false_valid"],
             walking_label=r["walking_label"],bothhalf_available_repeat=r["bothhalf_available_repeat"])
            for i,r in enumerate(records) if r["kind"]=="stress"]
    return dict(gates=results,all_pass=passed,denominators=dict(all=1440,nondegenerate=960,balanced=96,invalid_required=344,stress=96,transformed=64),
        failures=dict(legacy=legacy_fail,invalid=invalid_fail,mapping=mapping_fail,transforms=transform_fail),
        transformations=transformations,stress_diagnostics=stress,quality_cutoff=None,
        next_step="DEV_REGISTRATION_PREPARATION_ELIGIBLE_ONLY" if passed else "STOP_NO_RETRY_NO_PARAMETER_CHANGE",
        native_feasibility_or_DEV_authorized=False)


def replay(records,arrays,inventory,fixed_checks_pass,*,full_panel=True):
    """Recompute every correlation/stat/mask and all decisions. No generation or filter."""
    n=len(inventory)
    validate_shapes(arrays,n)
    c.require(len(records)==n,"Record inventory")
    expected=allocate(n)
    replayed=[]
    for i,item in enumerate(inventory):
        c.CHECK()
        raw=arrays["raw"][i]
        mags=guards.magnitudes(raw)
        bits_equal(mags,arrays["magnitudes"][i],"raw magnitudes")
        filtered=arrays["filtered"][i]  # frozen filter result: intentionally no refiltering
        corr,stats,valid,finite=guards.joint_correlations(filtered)
        for name,value in (("correlations",corr),("correlation_stats",stats),("correlation_valid",valid),("correlation_finite",finite)):
            bits_equal(value,arrays[name][i],name)
        record,mapped=kernel.decide(item,mags,filtered,corr)
        c.require(c.typed(record)==c.typed(records[i]),"Record replay mismatch: "+str(i))
        store(expected,i,record,dict(raw=raw,magnitudes=mags,filtered=filtered,correlations=corr,
              correlation_stats=stats,correlation_valid=valid,correlation_finite=finite),mapped)
        replayed.append(record)
    for name in arrays:
        bits_equal(expected[name],arrays[name],name)
    result=gates(replayed,expected,inventory,fixed_checks_pass) if full_panel else None
    return dict(status="pass",rows=n,correlations_stats_masks_replayed=True,
                minima_pairs_mapping_features_decisions_replayed=True,filter_or_generator_called=False,
                full_gates_replayed=full_panel),result


def inventory():
    result={}
    for path in sorted(c.ROOT.rglob("*")):
        c.require(not path.is_symlink(),"Artifact symlink forbidden")
        if path.is_file():
            name=path.relative_to(c.ROOT).as_posix()
            if name!="artifact_index.json": result[name]=c.sha(path)
    return result


def seal_index():
    temp=c.ROOT/".artifact_index.pending"
    c.require(not temp.exists(),"Prior failed index write retained; stop for root review")
    value=dict(schema=1,files_sha256=inventory(),excludes_only="artifact_index.json",trust=TRUST)
    path=c.ROOT/"artifact_index.json"
    # Same-filesystem atomic index publication; no log/replay files are placed here.
    # After replacement the temporary filename no longer exists, so exact inventory
    # excludes ONLY the final root index. A failed pending write is retained, not retried.
    with temp.open("xb") as handle:
        handle.write(c.canonical(value)+b"\n"); handle.flush(); os.fsync(handle.fileno())
    os.replace(temp,path)


def validate_index():
    value=c.load_json("artifact_index.json")
    expected=dict(schema=1,files_sha256=inventory(),excludes_only="artifact_index.json",trust=TRUST)
    c.require(c.typed(value)==c.typed(expected),"Exact artifact set/index mismatch")
    c.require((c.ROOT/"artifact_index.json").read_bytes()==c.canonical(value)+b"\n","Noncanonical index")
    return value


def save_arrays(arrays):
    (c.ROOT/"arrays").mkdir()
    for name,array in arrays.items():
        with (c.ROOT/"arrays"/(name+".npy")).open("xb") as handle:
            np.save(handle,array,allow_pickle=False)
        bits_equal(array,np.load(c.ROOT/"arrays"/(name+".npy"),allow_pickle=False),name)
    with (c.ROOT/"evidence.npz").open("xb") as handle:
        np.savez(handle,**arrays)
    with np.load(c.ROOT/"evidence.npz",allow_pickle=False) as saved:
        c.require(set(saved.files)==set(arrays),"NPZ exact keys")
        for name in arrays: bits_equal(arrays[name],saved[name],name)


def load_arrays():
    with np.load(c.ROOT/"evidence.npz",allow_pickle=False) as handle:
        arrays={name:handle[name] for name in handle.files}
    validate_shapes(arrays,1440)
    for name,array in arrays.items():
        bits_equal(array,np.load(c.ROOT/"arrays"/(name+".npy"),allow_pickle=False),name)
    return arrays


def attempt_binding():
    a=c.load_json("attempt.started.json"); w=c.load_json("worker.started.json")
    c.require(set(a)=={"automatic_retry","cpu_budget_seconds","launcher_pid","launcher_starttime_ticks","route","schema","status","wall_budget_seconds"},"Attempt schema")
    c.require(set(w)=={"attempt","schema","status","worker_pid","worker_starttime_ticks"},"Worker schema")
    c.require(a["automatic_retry"] is False and a["cpu_budget_seconds"]==600 and a["wall_budget_seconds"]==900 and
              a["schema"]==2 and a["status"]=="exclusive_attempt_consumed" and a["route"] in ("direct","shell"),"Attempt values")
    c.require(c.typed(w["attempt"])==c.typed(a) and w["schema"]==1 and w["status"]=="exclusive_worker_started","Worker attempt binding")
    for value,pid,start in ((a,"launcher_pid","launcher_starttime_ticks"),(w,"worker_pid","worker_starttime_ticks")):
        ticks=value[start]
        c.require(type(value[pid]) is int and value[pid]>0 and type(ticks) is str and ticks.isascii() and ticks.isdecimal() and ticks[0]!="0","Persisted identity")
    c.require((a["route"]=="direct" and a["launcher_pid"]==w["worker_pid"] and a["launcher_starttime_ticks"]==w["worker_starttime_ticks"]) or
              (a["route"]=="shell" and a["launcher_pid"]!=w["worker_pid"]),"Worker route identity")
    for name,value in (("attempt.started.json",a),("worker.started.json",w)):
        c.require((c.ROOT/name).read_bytes()==c.canonical(value)+b"\n","Canonical receipt")
    return dict(attempt=a,worker=w)


def bindings():
    names=("manifest.proposed.json","keys.json","schema.json","sources.json","parent_index.json","runtime.preparation.json",
           "preparation.json","selftests.json","freeze.json","attempt.started.json","worker.started.json",
           "prefix_gate.json","records.jsonl","evidence.npz","gates.json","saved_replay.json")
    return dict(files={name:c.sha(c.ROOT/name) for name in names},
                arrays={p.name:c.sha(p) for p in sorted((c.ROOT/"arrays").glob("*.npy"))},attempt=attempt_binding())
