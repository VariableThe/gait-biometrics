"""Preparation and ROOT-only future stages; imported only after entry guard."""
import ast
import json
import sys
import time
import traceback
from pathlib import Path
import common as c
import numpy as np
import evidence
import guards
import kernel
import legacy
import panel
import selftest

GENERATED=("manifest.proposed.json","keys.json","schema.json","sources.json","parent_index.json",
           "runtime.preparation.json","selftests.json","preparation.json")


def sources():
    return {p.name:c.sha(p) for p in sorted(c.ROOT.iterdir()) if p.is_file() and
            (p.suffix in (".py",".sh",".source",".md") or p.name=="copy_provenance.json")}


def provenance():
    saved=c.load_json("copy_provenance.json")
    for name,digest in saved["parents_sha256"].items():
        c.require(c.sha(Path(name))==digest,"Allowed parent SOURCE/brief/completed metadata changed: "+name)
    for item in saved["copies"]:
        parent=Path(item["parent"]).read_text()
        local=(c.ROOT/item["destination"]).read_text()
        if item["mode"]=="verbatim_AST_function":
            def segment(text):
                node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==item["function"])
                return ast.get_source_segment(text,node)
            pseg,lseg=segment(parent),segment(local)
            c.require(pseg==lseg and c.hashlib.sha256(lseg.encode()).hexdigest()==item["function_utf8_sha256"],"Copied function changed")
        elif item["mode"]=="verbatim_file":
            c.require(parent==local,"Verbatim source copy changed")
        elif item["mode"]=="verbatim_prefix":
            c.require(parent.encode()[:item["bytes"]]==local.encode(),"Entry guard prefix changed")
    prefix=(c.ROOT/"entry_guard.source").read_bytes()
    c.require((c.ROOT/"study.py").read_bytes().startswith(prefix),"Actual entry guard diverged")
    # The copied function closure includes these manually emitted legacy constants.
    def constants(text):
        return {n.targets[0].id:ast.dump(n.value) for n in ast.parse(text).body
                if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)}
    old=constants((c.ROOT/"legacy_parent.source").read_text())
    new=constants((c.ROOT/"legacy.py").read_text())
    for name in ("PERIODS","FAMILIES","RULES","LAGS","GUARD","ATOL"):
        c.require(old[name]==new[name],"Copied constant closure changed: "+name)
    return saved["parents_sha256"]


def proposal():
    value=panel.proposal()
    value.update(brief_sha256=c.sha(c.ROOT/"BRIEF.md"),source_files_sha256=sources(),
                 parent_files_sha256=provenance(),runtime_versions=c.EXPECTED_VERSIONS,
                 keys_sha256=c.fingerprint(panel.keys()),schema_sha256=c.fingerprint(evidence.schema(1440)),
                 source_closure="All local .py/.sh/inert .source/.md plus copy_provenance.json; no parent entrypoint imported; existing caches bound by runtime closure",
                 runtime_limits="Launcher permanent reservation; exclusive worker receipt before dependency imports/validation; PID+starttime and live ancestry shell handoff; root only run.sh execute; no OS sandbox/authenticity claim",
                 expected_file_scope="Exact recursive file inventory excludes ONLY root artifact_index.json; logs/replay output outside owned tree")
    return value


def prepare():
    c.require(not any((c.ROOT/name).exists() for name in GENERATED+("freeze.json","attempt.started.json","worker.started.json")),"Preparation already exists: preserve it; do not overwrite")
    timings={}
    with c.stage(timings,"handchecks"):
        tests,history=selftest.recorded_run()
    with c.stage(timings,"source_parent_runtime_validation"):
        parents=provenance()
        runtime=c.runtime_identity()
        source=sources()
        manifest=proposal()
        inventory=panel.keys()  # METADATA ONLY
    for name,value in (("manifest.proposed.json",manifest),("keys.json",inventory),
                       ("schema.json",evidence.schema(1440)),("sources.json",dict(files=source)),
                       ("parent_index.json",parents),("runtime.preparation.json",runtime),
                       ("selftests.json",dict(result=tests,history=history,source_files_sha256=source))):
        c.save_json(name,value)
    c.save_json("preparation.json",dict(status="PREPARED_FOR_ROOT_REVIEW_NOT_FROZEN_NOT_MEASURED",utc=c.utc(),
        bindings={name:c.sha(c.ROOT/name) for name in GENERATED if name!="preparation.json"},
        runtime_fingerprint=c.fingerprint(runtime),manifest_fingerprint=c.fingerprint(manifest),
        timings=timings,process=c.elapsed(),full_panel_extractions=0,primary96_extractions=0,
        root_review_required=True,adaptive_exploratory=True,latest_joint_measurement_outcomes_read=False,
        unresolved_brief_ambiguities=[]))
    evidence.seal_index()
    return dict(status="prepared",full_panel_extractions=0,frozen=False)


def validate_inputs(*,index=True):
    if index: evidence.validate_index()
    source=c.load_json("sources.json")["files"]
    c.require(source==sources(),"Source closure changed")
    parents=provenance()
    c.require(c.typed(parents)==c.typed(c.load_json("parent_index.json")),"Allowed parent index changed")
    prep=c.load_json("preparation.json")
    for name,digest in prep["bindings"].items():
        c.require(c.sha(c.ROOT/name)==digest,"Preparation binding changed: "+name)
    tests=c.load_json("selftests.json")
    c.require(tests["result"]["status"]=="pass" and tests["source_files_sha256"]==source,"Fixed tests stale/failed")
    c.require(c.typed(c.load_json(tests["history"]))==c.typed(tests["result"]),"Handcheck history binding")
    c.require(c.typed(c.load_json("keys.json"))==c.typed(panel.keys()),"Keys changed")
    c.require(c.typed(c.load_json("schema.json"))==c.typed(evidence.schema(1440)),"Schema changed")
    manifest=c.load_json("manifest.proposed.json")
    c.require(c.typed(manifest)==c.typed(proposal()),"Proposed manifest changed")
    runtime=c.validate_runtime(c.load_json("runtime.preparation.json"))
    c.require(c.fingerprint(runtime)==prep["runtime_fingerprint"] and c.fingerprint(manifest)==prep["manifest_fingerprint"],"Runtime/manifest mismatch")
    return prep


def freeze():
    c.require(not any((c.ROOT/name).exists() for name in ("attempt.started.json","worker.started.json","report.json")),"Attempt already consumed or inconsistent run state")
    validate_inputs()
    c.save_json("freeze.json",dict(status="ROOT_FROZEN_ONE_ATTEMPT",utc=c.utc(),
        files={name:c.sha(c.ROOT/name) for name in GENERATED},source_files_sha256=sources(),
        runtime_sha256=c.sha(c.ROOT/"runtime.preparation.json"),keys_sha256=c.sha(c.ROOT/"keys.json"),
        parent_index_sha256=c.sha(c.ROOT/"parent_index.json"),root_review_required=True,
        rule="ANY gate failure STOP; no retry, length search or outcome changes"))
    evidence.seal_index()
    return dict(status="frozen",execution_authorized_by_implementation=False)


def validate_freeze():
    frozen=c.load_json("freeze.json")
    c.require(frozen["status"]=="ROOT_FROZEN_ONE_ATTEMPT","Freeze required")
    for name,digest in frozen["files"].items():
        c.require(c.sha(c.ROOT/name)==digest,"Frozen file changed: "+name)
    c.require(frozen["source_files_sha256"]==sources(),"Frozen source changed")
    c.require(frozen["runtime_sha256"]==c.sha(c.ROOT/"runtime.preparation.json") and
              frozen["keys_sha256"]==c.sha(c.ROOT/"keys.json") and frozen["parent_index_sha256"]==c.sha(c.ROOT/"parent_index.json"),"Freeze closure mismatch")
    return frozen


def prefix_gate(inventory):
    """Root measurement prerequisite ONLY; no correlations, filter or candidate."""
    checked=[]
    for item in inventory:
        if item["kind"]!="legacy": continue
        c.CHECK()
        p,k,f=item["period_samples"],item["phase_index"],item["family"]
        native=legacy.fixture(f,p,k,446)
        old=legacy.fixture(f,p,k,256)
        c.require(native[:,:256].tobytes()==old.tobytes(),"Full N256 prefix bit gate failed: "+item["key"])
        checked.append(dict(key=item["key"],N256_sha256=c.hashlib.sha256(old.tobytes()).hexdigest()))
    c.require(len(checked)==1280,"Full legacy prefix denominator")
    return dict(status="pass",rows=1280,candidate_extractions=0,parent_measurement_cache_read=False,
                source_verbatim_checked=True,checked=checked)


def execute():
    """ROOT only. The entry guard has ALREADY permanently consumed the attempt."""
    c.arm_budget()
    timings={}
    try:
        receipt=evidence.attempt_binding()
        c.require(receipt["attempt"]["route"]=="shell","Root execute must use run.sh: wall deadline must enclose uv startup")
        # The pre-attempt index necessarily lacks the two newly written receipts. Check its
        # entire exact set after removing ONLY those two expected additions, then bind both.
        before=c.load_json("artifact_index.json")
        actual=evidence.inventory()
        for name in ("attempt.started.json","worker.started.json"):
            c.require(name not in before["files_sha256"],"Attempt existed in prior index")
            actual.pop(name)
        c.require(actual==before["files_sha256"],"Pre-execute exact frozen inventory mismatch")
        with c.stage(timings,"premeasurement_validation"):
            validate_inputs(index=False)
            validate_freeze()
            inventory=c.load_json("keys.json")
        # Source, manifest, runtime, keys and parent hashes validated BEFORE inputs generated.
        with c.stage(timings,"full_prefix_prerequisite"):
            control=prefix_gate(inventory)
            c.save_json("prefix_gate.json",control)
        with c.stage(timings,"measurement"):
            arrays=evidence.allocate(1440)
            records=[]; stress_cache={}
            for i,item in enumerate(inventory):
                c.CHECK()
                raw=panel.generate(item,stress_cache)
                record,data,mapped=kernel.extract(item,raw)
                evidence.store(arrays,i,record,data,mapped)
                records.append(record)
        with c.stage(timings,"lossless_save"):
            with (c.ROOT/"records.jsonl").open("xb") as handle:
                for record in records: handle.write(c.canonical(record)+b"\n")
            evidence.save_arrays(arrays)
        with c.stage(timings,"saved_evidence_verification"):
            # Saved objects are read back: no in-memory-only verification claim.
            saved_records=[c.decode_json(line) for line in (c.ROOT/"records.jsonl").read_bytes().splitlines()]
            saved_arrays=evidence.load_arrays()
            audit,result=evidence.replay(saved_records,saved_arrays,inventory,True)
            c.save_json("saved_replay.json",audit)
            c.save_json("gates.json",result)
        c.save_json("report.json",dict(status="COMPLETE_PASS" if result["all_pass"] else "COMPLETE_GATE_FAIL_STOP",
            bindings=evidence.bindings(),gate_fingerprint=c.fingerprint(result),timings=timings,process=c.elapsed(),
            full_panel_inputs=1440,adaptive_exploratory=True,quality_cutoff=None,
            eligibility=result["next_step"],DEV_execution_authorized=False,trust=evidence.TRUST))
        evidence.seal_index()
        return dict(status="complete",gates_pass=result["all_pass"],next_step=result["next_step"])
    except BaseException as exc:
        # Exclusive histories are retained even when startup, validation or a gate fails.
        name="execution_failure-"+str(time.time_ns())+".json"
        try:
            c.save_json(name,dict(error_type=type(exc).__name__,error=str(exc),timings=timings,
                process=c.elapsed(),automatic_retry=False,traceback=traceback.format_exc()))
            evidence.seal_index()
        except BaseException:
            pass
        raise


def verify():
    """Read-only replay; caller redirects logs outside the owned tree."""
    validate_inputs()
    validate_freeze()
    report=c.load_json("report.json")
    c.require(c.typed(report["bindings"])==c.typed(evidence.bindings()),"Report binding mismatch")
    inventory=c.load_json("keys.json")
    records=[c.decode_json(line) for line in (c.ROOT/"records.jsonl").read_bytes().splitlines()]
    arrays=evidence.load_arrays()
    audit,result=evidence.replay(records,arrays,inventory,True)
    c.require(c.typed(audit)==c.typed(c.load_json("saved_replay.json")),"Saved replay audit changed")
    c.require(c.typed(result)==c.typed(c.load_json("gates.json")) and report["gate_fingerprint"]==c.fingerprint(result),"Full gate replay mismatch")
    c.require(report["status"]==("COMPLETE_PASS" if result["all_pass"] else "COMPLETE_GATE_FAIL_STOP") and
              report["eligibility"]==result["next_step"] and report["DEV_execution_authorized"] is False and
              report["full_panel_inputs"]==1440 and report["adaptive_exploratory"] is True and
              report["quality_cutoff"] is None and report["trust"]==evidence.TRUST,"Report interpretation/schema mismatch")
    return dict(status="verified",all_gates_pass=result["all_pass"],native_feasibility_claim=False)


def main(stage):
    if stage=="prepare":
        try:
            result=prepare()
        except BaseException as exc:
            directory="preparation_history/infrastructure-failure-"+str(time.time_ns())
            (c.ROOT/directory/"source").mkdir(parents=True)
            for name in sources():
                with (c.ROOT/directory/"source"/name).open("xb") as handle:
                    handle.write((c.ROOT/name).read_bytes())
            c.save_json(directory+"/result.json",dict(status="failed",stage="prepare",
                error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),
                source_files_sha256=sources(),full_panel_extractions=0,primary96_extractions=0))
            raise
    elif stage=="check-preparation": validate_inputs(); result=dict(status="preparation_valid",frozen=(c.ROOT/"freeze.json").exists())
    elif stage=="selftest":
        c.require(not any((c.ROOT/n).exists() for n in ("preparation.json","freeze.json","attempt.started.json")),"Selftest after binding prohibited")
        result,history=selftest.recorded_run(); result=dict(status=result["status"],history=history)
    elif stage=="freeze": result=freeze()
    elif stage=="execute": result=execute()
    elif stage=="verify": result=verify()
    else: raise RuntimeError("Unknown stage")
    print(c.canonical(result).decode(),flush=True)

