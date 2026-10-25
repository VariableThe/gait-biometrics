"""Fixed hand cases and ONE nonprimary P100 gravity phase0 fixture only."""
import io
import ast
from contextlib import redirect_stdout
import math
import time
import traceback
from types import SimpleNamespace
import numpy as np
from scipy.interpolate import CubicSpline
import common as c
import evidence
import guards
import kernel
import legacy
import panel
import plateau_helpers


def scalar_reference(magnitudes,pair):
    """Independent scalar knot search, weighting, mean and population variance."""
    a,b=pair
    result=[]
    for channel in magnitudes:
        sampled=[]
        for k in range(200):
            coordinate=a+(b-a)*k/200
            lo=a
            while lo+1<=b and lo+1<=coordinate:
                lo+=1
            hi=min(lo+1,b)
            sampled.append(float(channel[lo])*(hi-coordinate)+float(channel[hi])*(coordinate-lo)
                           if hi!=lo else float(channel[lo]))
        mean=math.fsum(sampled)/200
        std=math.sqrt(math.fsum((value-mean)**2 for value in sampled)/200)
        result.append((sampled,mean,std,[(value-mean)/std for value in sampled]))
    return result


def run():
    cpu,wall=time.process_time(),time.perf_counter()
    checks=[]
    def ok(name,condition):
        c.require(bool(condition),"Fixed handcheck failed: "+name)
        checks.append(name)
    # In-memory argv and process-identity hand cases. Never enter execute/reserve a worker.
    source=(c.ROOT/"study.py").read_text()
    nodes=ast.parse(source).body
    argv_guard=next(node for node in nodes if isinstance(node,ast.If))
    grammar=compile(ast.Module(body=[argv_guard],type_ignores=[]),"<fixed-argv-handcheck>","exec")
    cases=[(["prepare"],None),(["check-preparation"],None),(["freeze"],None),(["execute"],None),
           (["verify"],None),(["verify","--replay-correlations"],None),
           (["--replay-correlations","execute"],2),(["--help","execute"],2),
           (["execute","--help"],2),(["prepare","execute"],2),(["execute","execute"],2),([],2),
           (["--help"],0),(["-h"],0)]
    for args,expected in cases:
        fake=SimpleNamespace(argv=["study.py",*args],stderr=io.StringIO())
        exit_code=None
        with redirect_stdout(io.StringIO()):
            try: exec(grammar,{"__name__":"__main__","sys":fake})
            except SystemExit as exc: exit_code=exc.code
        ok("fixed_argv_"+repr(args),exit_code==expected)
    helpers=[node for node in nodes if isinstance(node,ast.FunctionDef) and node.name=="_live_ancestor"]
    namespace={}
    exec(compile(ast.Module(body=helpers,type_ignores=[]),"<fixed-live-ancestor-handcheck>","exec"),namespace)
    worker=(200,100,"20","R")
    for label,start,state,ancestor,expected in (("matching", "10","S",100,True),
                ("reused_pid","11","S",100,False),("dead","10","Z",100,False),
                ("not_ancestor","10","S",99,False)):
        table={100:(100,1,start,state),1:(1,0,"1","S")}
        namespace["_process_identity"]=lambda pid:table[pid]
        accepted=True
        try: namespace["_live_ancestor"](ancestor,"10",worker)
        except RuntimeError: accepted=False
        ok("fixed_handoff_"+label,accepted is expected)
    ok("exclusive_worker_receipt_precedes_driver_import",source.index('/worker.started.json", "x"')<source.index('from driver import main'))
    # No extraction call for these manually specified vectors/pairs.
    t=np.arange(446,dtype=float)
    mags=np.stack((5+.01*t+.0001*t*t,11+.3*np.sin(t/9)+.002*t))
    pair=[13,140]
    mapped=kernel.linear_mapping(mags,pair)
    ref=scalar_reference(mags,pair)
    scalar_samples=np.array([r[0] for r in ref])
    scalar_features=np.array([r[3] for r in ref]).reshape(400)
    ok("independent_scalar_samples",np.max(np.abs(mapped["samples"]-scalar_samples))<=1e-12)
    ok("independent_scalar_standardization_acc_first",np.max(np.abs(mapped["feature"]-scalar_features))<=1e-12)
    ok("shared_coordinates",np.array_equal(mapped["coordinates"],13+127*np.arange(200,dtype=float)/200))
    ok("left_included_right_excluded",mapped["coordinates"][0]==13 and mapped["coordinates"][-1]<140)
    ok("right_endpoint_is_interpolation_knot",mapped["upper"][-1]==140 and 0<mapped["fraction"][-1]<1)
    changed=mags.copy(); changed[:,140]+=7
    changed_mapping=kernel.linear_mapping(changed,pair)
    ok("b_knot_actually_influences_last_sample",np.all(changed_mapping["samples"][:,-1]!=mapped["samples"][:,-1]))
    inclusive=np.stack([np.interp(np.linspace(*pair,200,endpoint=True),np.arange(446),row) for row in mags])
    ok("wrong_old_inclusive_linear_mapping_rejected",np.max(np.abs(inclusive-mapped["samples"]))>1e-3)
    old=CubicSpline(np.arange(pair[0],pair[1]+1),mags[:,pair[0]:pair[1]+1],axis=1,bc_type="natural")(np.linspace(*pair,200,endpoint=True))
    ok("wrong_old_natural_cubic_inclusive_rejected",np.max(np.abs(old-mapped["samples"]))>1e-3)
    ok("norm20_no_extra_global_normalization",kernel.mapping_checks(mapped["feature"])["pass_verification_only"])
    for label,vector,expected in (
        ("zero",np.zeros(200),False),("constant",np.ones(200),False),
        ("DC_5e-13",1+5e-13*np.tile([-1.,1.],100),False),
        ("DC_2e-12",1+2e-12*np.tile([-1.,1.],100),True),
        ("finite",np.arange(200,dtype=float),True),
        ("nan",np.array([1.,np.nan]),False),("posinf",np.array([1.,np.inf]),False),
        ("neginf",np.array([1.,-np.inf]),False)):
        good,_,_=guards.vector_guard(vector)
        ok("vector_guard_"+label,good is expected)
    # Map guard uses the exact same tests; coordinates at D200 are integer samples.
    for label,amplitude,expected in (("DC_5e-13",5e-13,False),("DC_2e-12",2e-12,True)):
        v=1+amplitude*np.where(np.arange(446)%2,-1.,1.)
        m=kernel.linear_mapping(np.stack((v,v)),[10,210])
        ok("mapping_guard_"+label,bool(m["valid"].all()) is expected)
    for label,value in (("constant",1.),("nan",np.nan),("posinf",np.inf),("zero",0.)):
        m=kernel.linear_mapping(np.full((2,446),value),[10,210])
        ok("mapping_reject_"+label,m["feature"] is None)
    x=np.array([1.,2.,4.,8.,16.])
    for label,y,target in (("same",x,1.),("negated",-x,-1.),("positive_affine",3*x+7,1.)):
        r,_,valid,_=guards.pearson(x,y)
        ok("pearson_"+label,valid.all() and abs(r-target)<=1e-14)
    # Fixed saved-correlation vector: missing gyro cannot reweight/fallback.
    corr,_,valid,_=guards.joint_correlations(np.stack((np.sin(t/7),np.ones(446))))
    ok("undefined_modality_no_joint_fallback",not np.isfinite(corr[2]).any() and not valid[1].any())
    pchecks=plateau_helpers.synthetic_helper_checks()
    ok("reused_16_strict_vs_plateau_helpers",pchecks["fixed_case_count"]==16)
    # Integer admissibility, all ordered pairs across intervening minima and exact ranking.
    eligible,chosen=kernel.pairs([10,50,90,110,170,210,250,330,435],100,446)
    independent=[[a,b] for a in [10,50,90,110,170,210,250,330,435]
                 for b in [10,50,90,110,170,210,250,330,435]
                 if b>a and b-a>=80 and b-a<=200 and 10*abs(b-a-100)<=100]
    ok("all_pairs_including_intervening_minima",eligible==independent and [10,110] in eligible)
    ok("integer_rank",chosen==min(independent,key=lambda ab:(abs(ab[1]-ab[0]-100),abs(sum(ab)-445),ab[0],ab[1])))
    nearchecks,_=legacy.fixed_checks()
    ok("reused_18_nearmax_checks",len(nearchecks["rows"])==18)
    for value in (True,1,1.,-0.,0.,None,"x",{"negative_zero":-0.,"list":[1,1.,True]}):
        ok("json_type_and_signedzero_roundtrip_"+str(len(checks)),c.typed(c.decode_json(c.canonical(value)))==c.typed(value))
    for invalid in ({1:2},{"bad":float("nan")}):
        rejected=False
        try: c.canonical(invalid)
        except RuntimeError: rejected=True
        ok("json_reject_nonstring_or_nonfinite_"+str(len(checks)),rejected)
    float_probe=np.array([-0.,0.,np.nan,np.inf,-np.inf,1.],dtype="<f8")
    stream=io.BytesIO(); np.savez(stream,probe=float_probe); stream.seek(0)
    with np.load(stream,allow_pickle=False) as saved: evidence.bits_equal(float_probe,saved["probe"],"fixed NPZ bits")
    stream=io.BytesIO(); np.save(stream,float_probe,allow_pickle=False); stream.seek(0)
    evidence.bits_equal(float_probe,np.load(stream,allow_pickle=False),"fixed NPY bits")
    ok("NPZ_NPY_signedzero_nonfinite_payload_bits",True)
    # The sole predeclared nonprimary input fixture. No primary P160/180/200 panel calls.
    item=dict(input_index=0,kind="hand_fixture",family=legacy.FAMILIES[0],period_samples=100,
              phase_index=0,length=446,condition="original",transformation="identity",key="nonprimary/gravity/P100/phase0/N446")
    raw=legacy.fixture(legacy.FAMILIES[0],100,0,446)
    prefix=legacy.fixture(legacy.FAMILIES[0],100,0,256)
    ok("single_fixture_N256_prefix_bits",raw[:,:256].tobytes()==prefix.tobytes())
    record,data,mapped=kernel.extract(item,raw)
    arrays=evidence.allocate(1); evidence.store(arrays,0,record,data,mapped)
    audit,_=evidence.replay([record],arrays,[item],True,full_panel=False)
    ok("single_fixture_complete_saved_replay",audit["status"]=="pass")
    invariance=[]
    for transform in panel.TRANSFORMS:
        if transform.startswith("common_R"):
            r=panel.rotation(transform); transformed=np.vstack((r@raw[:3],r@raw[3:]))
        else:
            ga,gg=(.9,1.1) if transform==panel.TRANSFORMS[2] else (1.1,.9)
            transformed=np.vstack((ga*raw[:3],gg*raw[3:]))
        target=dict(item,transformation=transform)
        other,_,mapping=kernel.extract(target,transformed)
        unchanged=all(record[k]==other[k] for k in ("valid","selected_lag","selected_pair"))
        delta=None if mapped is None or mapping is None or mapped["feature"] is None or mapping["feature"] is None else float(np.max(np.abs(mapped["feature"]-mapping["feature"])))
        ok("one_fixture_invariance_"+transform,unchanged and (delta is None or delta<=1e-10))
        invariance.append(dict(transform=transform,decisions_unchanged=unchanged,feature_max_abs_delta=delta))
    ok("fixed_metadata_counts_only",len(panel.keys())==1440)
    return dict(status="pass",checks=checks,count=len(checks),plateau_checks=pchecks,
                nearmax_checks=nearchecks,single_fixture=dict(key=item["key"],valid=record["valid"],
                lag=record["selected_lag"],pair=record["selected_pair"],raw_sha256=c.hashlib.sha256(raw.tobytes()).hexdigest(),
                prefix_sha256=c.hashlib.sha256(prefix.tobytes()).hexdigest(),invariance=invariance,saved_replay=audit),
                no_primary_or_full_panel=True,full_panel_extractions=0,primary96_extractions=0,
                candidate_hand_fixture_extractions=5,cpu_seconds=time.process_time()-cpu,wall_seconds=time.perf_counter()-wall)


def recorded_run():
    """Preserve every failed preparation attempt; no automatic retry or tolerance tuning."""
    directory=c.ROOT/"preparation_history"
    directory.mkdir(exist_ok=True)
    attempt="preparation_history/check-"+str(time.time_ns())
    (c.ROOT/attempt/"source").mkdir(parents=True)
    hashes={}
    for path in sorted(c.ROOT.iterdir()):
        if path.is_file() and (path.suffix in (".py",".sh",".source",".md") or path.name=="copy_provenance.json"):
            data=path.read_bytes()
            with (c.ROOT/attempt/"source"/path.name).open("xb") as handle: handle.write(data)
            hashes[path.name]=c.hashlib.sha256(data).hexdigest()
    c.save_json(attempt+"/source_hashes.json",hashes)
    name=attempt+"/result.json"
    try:
        result=run()
    except Exception as exc:
        c.save_json(name,dict(status="failed",error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc()))
        raise
    c.save_json(name,result)
    return result,name
