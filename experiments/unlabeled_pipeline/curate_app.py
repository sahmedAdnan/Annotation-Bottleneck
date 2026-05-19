"""
Interactive curation web app for the unlabeled annotation pipeline.

Launches a local web server that shows every cluster with guided decision
support, lets a human override the AI at any granularity, and on Finalize
writes a clean imagefolder dataset that exactly reflects the human decisions.

Guidance for the ecologist:
  - Each cluster shows the AI top-1 and the AI runner-up as one-click choices.
  - Each choice is backed by reference exemplars: the highest-confidence
    images the AI assigned to that species elsewhere in the dataset.
  - Free typing is the fallback only, for the rare third-species case.

Single source of truth per image:
    final_label  -- the species the image belongs to
    status       -- one of: species | noise | junk
cluster_id and ai_annotation are read-only provenance. The exported dataset
is built ONLY from final_label and status, so there is no second merge step.

Usage:
    python experiments/unlabeled_pipeline/curate_app.py \
        --annotations results/Annotation_eumoth/annotations.csv --port 8000
Then open http://localhost:8000 in a browser.
"""
import argparse
import io
import json
import os
import shutil
import threading
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import pandas as pd
from PIL import Image

N_REF = 6  # reference exemplars per species (highest-confidence members)

STATE = {
    "df": None, "final": {}, "status": {}, "candidates": [],
    "valid_paths": set(), "exemplars": {}, "out_dir": None,
}
THUMB_CACHE = {}
LOCK = threading.Lock()


def _clean(x):
    s = str(x).strip()
    return "" if s.lower() in ("", "nan", "none") else s


def load_state(annotations_csv, out_dir):
    df = pd.read_csv(annotations_csv)
    df["cluster_id"] = df["cluster_id"].astype(int)
    STATE["df"] = df
    STATE["out_dir"] = out_dir

    for _, r in df.iterrows():
        p = r["file_path"]
        STATE["final"][p] = str(r["ai_annotation"])
        STATE["status"][p] = "noise" if int(r["cluster_id"]) == -1 else "species"
        STATE["valid_paths"].add(p)

    STATE["candidates"] = sorted({_clean(x) for x in df["ai_annotation"]} - {""})

    # Exemplars: per species, the N_REF highest-confidence images the AI
    # assigned to that species (option (a), the cleanest visual prototype).
    conf = "confidence" if "confidence" in df.columns else None
    for sp, g in df.groupby("ai_annotation"):
        sp = _clean(sp)
        if not sp:
            continue
        gg = g.sort_values(conf, ascending=False) if conf else g
        STATE["exemplars"][sp] = list(gg["file_path"].head(N_REF))


def _majority(series):
    vals = [_clean(v) for v in series if _clean(v)]
    return Counter(vals).most_common(1)[0][0] if vals else ""


def build_state_json():
    df = STATE["df"]
    clusters = {}
    for _, r in df.iterrows():
        clusters.setdefault(int(r["cluster_id"]), []).append(r)

    out = []
    for cid in sorted(clusters, key=lambda c: (c != -1, c)):
        rows = clusters[cid]
        top1 = _majority([x["ai_annotation"] for x in rows])
        top2 = _majority([x.get("ai_second_choice", "") for x in rows])
        margins = [float(x["margin"]) for x in rows
                   if "margin" in x and pd.notna(x["margin"])]
        margin = round(sum(margins) / len(margins), 3) if margins else None
        images = [{
            "path": x["file_path"],
            "name": os.path.basename(x["file_path"]),
            "ai": str(x["ai_annotation"]),
            "ai2": _clean(x.get("ai_second_choice", "")),
            "final": STATE["final"][x["file_path"]],
            "status": STATE["status"][x["file_path"]],
        } for x in rows]
        out.append({
            "cluster_id": cid, "is_noise": cid == -1, "n": len(images),
            "ai_top1": top1, "ai_top2": top2, "margin": margin,
            "ref_top1": STATE["exemplars"].get(top1, []),
            "ref_top2": STATE["exemplars"].get(top2, []),
            "images": images,
        })
    return {"clusters": out, "candidates": STATE["candidates"]}


def make_thumb(path, box=200):
    if path in THUMB_CACHE:
        return THUMB_CACHE[path]
    im = Image.open(path).convert("RGB")
    im.thumbnail((box, box))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=80)
    THUMB_CACHE[path] = buf.getvalue()
    return THUMB_CACHE[path]


def finalize():
    df = STATE["df"]
    out_dir = STATE["out_dir"]
    clean = os.path.join(out_dir, "dataset")
    noise_dir = os.path.join(out_dir, "_NOISE")
    junk_dir = os.path.join(out_dir, "_FLAGGED")
    for d in (clean, noise_dir, junk_dir):
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    prov, counts = [], {"species": 0, "noise": 0, "junk": 0, "corrected": 0}
    for _, r in df.iterrows():
        src = r["file_path"]
        ai = str(r["ai_annotation"])
        cid = int(r["cluster_id"])
        final = STATE["final"][src]
        st = STATE["status"][src]
        fname = os.path.basename(src)

        dest_dir = (junk_dir if st == "junk" else
                    noise_dir if st == "noise" else
                    os.path.join(clean, final.replace(" ", "_")))
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, fname)
        b, e = os.path.splitext(fname)
        n = 1
        while os.path.exists(dest):
            dest = os.path.join(dest_dir, f"{b}_{n}{e}")
            n += 1
        try:
            shutil.copy2(src, dest)
        except Exception as ex:
            print(f"[warn] copy failed {src}: {ex}")
            continue

        if st == "species" and final != ai:
            counts["corrected"] += 1
        counts[st] = counts.get(st, 0) + 1
        prov.append({
            "file": fname, "original_cluster_id": cid, "ai_annotation": ai,
            "final_label": final, "status": st,
            "human_changed": bool((st == "species" and final != ai)
                                  or st in ("noise", "junk")),
        })

    pd.DataFrame(prov).to_csv(os.path.join(out_dir, "provenance.csv"),
                              index=False)
    n_species = len([d for d in os.listdir(clean)
                     if os.path.isdir(os.path.join(clean, d))])
    return {"clean_dir": clean, "species_folders": n_species,
            "kept": counts["species"], "noise": counts["noise"],
            "junk": counts["junk"], "human_corrected": counts["corrected"]}


PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<title>Dataset Curation</title><style>
body{font-family:system-ui,Arial,sans-serif;margin:0;background:#eef0f3;color:#111}
header{position:sticky;top:0;background:#1f2937;color:#fff;padding:10px 16px;
 display:flex;gap:14px;align-items:center;z-index:20}
header b{font-size:15px}.stat{font-size:13px;opacity:.85}
button{cursor:pointer;border:1px solid #cbd5e1;background:#fff;border-radius:6px;
 padding:5px 10px;font-size:13px}
button.primary{background:#2563eb;color:#fff;border:0}
button.ok{background:#16a34a;color:#fff;border:0}
button.alt{background:#7c3aed;color:#fff;border:0}
button.warn{background:#b91c1c;color:#fff;border:0}
.cluster{background:#fff;margin:12px;border-radius:10px;box-shadow:0 1px 3px #0001}
.chead{display:flex;gap:10px;align-items:center;padding:10px 14px;
 border-bottom:1px solid #eee;flex-wrap:wrap}
.chead.noise{background:#fff7ed}
.refs{display:flex;gap:18px;padding:10px 14px;border-bottom:1px solid #f0f0f0;
 flex-wrap:wrap;background:#fafafa}
.refbox{font-size:12px}.refbox .t{font-weight:600;margin-bottom:4px}
.refbox.t1 .t{color:#1d4ed8}.refbox.t2 .t{color:#7c3aed}
.refrow{display:flex;gap:4px}
.refrow img{width:74px;height:74px;object-fit:cover;border-radius:4px;
 border:1px solid #ddd;background:#eee}
.cbody{display:flex;flex-wrap:wrap;gap:10px;padding:12px}
.card{width:172px;border:1px solid #e5e7eb;border-radius:8px;overflow:hidden}
.card img{width:172px;height:160px;object-fit:cover;display:block;background:#eee}
.card .meta{padding:6px;font-size:11px}
.card input{width:100%;font-size:12px;padding:3px;border:1px solid #cbd5e1;
 border-radius:4px;margin-top:3px}
.badge{font-size:10px;padding:2px 6px;border-radius:10px}
.b-ai{background:#e0e7ff;color:#3730a3}.b-human{background:#dcfce7;color:#166534}
.b-noise{background:#ffedd5;color:#9a3412}.b-junk{background:#fee2e2;color:#991b1b}
.qrow{display:flex;gap:3px;margin-top:4px}.qrow button{flex:1;padding:3px;font-size:10px}
.collapsed .refs,.collapsed .cbody{display:none}
</style></head><body>
<header><b>Dataset Curation</b><span class="stat" id="stat">loading…</span>
 <span style="flex:1"></span>
 <button onclick="saveAll()">Save progress</button>
 <button class="primary" onclick="finalize()">Finalize &amp; Export</button>
</header>
<div id="app"></div><datalist id="cands"></datalist>
<script>
let DATA=null;
function esc(s){return (s||'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;',
 '>':'&gt;','"':'&quot;'}[c]))}
function tn(p){return '/thumb?path='+encodeURIComponent(p)}
async function load(){
 DATA=await (await fetch('/api/state')).json();
 document.getElementById('cands').innerHTML=
  DATA.candidates.map(c=>`<option value="${esc(c)}">`).join('');
 render();
}
function counts(){let h=0,n=0,j=0;DATA.clusters.forEach(c=>c.images.forEach(im=>{
 if(im.status==='noise')n++;else if(im.status==='junk')j++;
 else if(im.final!==im.ai)h++;}));return{h,n,j};}
function refbox(cls,label,name,paths,extra){
 if(!name)return '';
 return `<div class="refbox ${cls}"><div class="t">${label}: ${esc(name)}${extra||''}</div>
  <div class="refrow">${paths.map(p=>`<img loading="lazy" src="${tn(p)}">`).join('')}</div></div>`;
}
function render(){
 const a=document.getElementById('app');a.innerHTML='';
 DATA.clusters.forEach(cl=>{
  const d=document.createElement('div');d.className='cluster';
  const title=cl.is_noise?'NOISE (unassigned)':'Cluster '+cl.cluster_id;
  const dm=cl.margin!=null?` Δ${cl.margin}`:'';
  let head=`<div class="chead ${cl.is_noise?'noise':''}">
   <button onclick="this.closest('.cluster').classList.toggle('collapsed')">▾</button>
   <b>${title}</b><span class="stat">${cl.n} images</span>`;
  if(!cl.is_noise){
   head+=`<button class="ok" onclick="applyCluster(${cl.cluster_id},'${esc(cl.ai_top1)}')">
     Confirm: ${esc(cl.ai_top1)}</button>`;
   if(cl.ai_top2)head+=`<button class="alt"
     onclick="applyCluster(${cl.cluster_id},'${esc(cl.ai_top2)}')">
     Runner-up: ${esc(cl.ai_top2)}${dm}</button>`;
   head+=`<input list="cands" placeholder="other species…" style="width:190px"
     id="cl${cl.cluster_id}">
    <button onclick="applyTyped(${cl.cluster_id})">Apply</button>`;
  }
  head+=`<button class="warn" onclick="markCluster(${cl.cluster_id},'noise')">All noise</button>
   <button class="warn" onclick="markCluster(${cl.cluster_id},'junk')">All junk</button></div>`;
  let refs='';
  if(!cl.is_noise){
   refs=`<div class="refs">
    ${refbox('t1','Reference (AI top-1)',cl.ai_top1,cl.ref_top1)}
    ${refbox('t2','Reference (runner-up)',cl.ai_top2,cl.ref_top2,dm)}
   </div>`;
  }
  d.innerHTML=head+refs+`<div class="cbody"></div>`;
  const body=d.querySelector('.cbody');
  cl.images.forEach(im=>body.appendChild(card(im)));
  a.appendChild(d);
 });
 updateStat();
}
function card(im){
 const c=document.createElement('div');c.className='card';
 const badge=im.status==='noise'?'<span class="badge b-noise">noise</span>':
   im.status==='junk'?'<span class="badge b-junk">junk</span>':
   (im.final!==im.ai?'<span class="badge b-human">human</span>':
    '<span class="badge b-ai">ai</span>');
 let q=`<button onclick="setImg('${encodeURIComponent(im.path)}','${esc(im.ai)}')"
   title="${esc(im.ai)}">→ top-1</button>`;
 if(im.ai2)q+=`<button onclick="setImg('${encodeURIComponent(im.path)}','${esc(im.ai2)}')"
   title="${esc(im.ai2)}">→ runner-up</button>`;
 c.innerHTML=`<img loading="lazy" src="${tn(im.path)}">
  <div class="meta"><div title="${esc(im.name)}">${esc(im.name).slice(0,26)}</div>
  <div>${badge} <b>${esc(im.final)}</b></div>
  <div class="qrow">${q}</div>
  <input list="cands" value="${esc(im.final)}"
   onchange="setImg('${encodeURIComponent(im.path)}',this.value)">
  <div class="qrow">
   <button onclick="setStatus('${encodeURIComponent(im.path)}','noise')">noise</button>
   <button onclick="setStatus('${encodeURIComponent(im.path)}','junk')">junk</button>
   <button onclick="setImg('${encodeURIComponent(im.path)}','${esc(im.ai)}')">reset</button>
  </div></div>`;
 return c;
}
function findImg(p){for(const cl of DATA.clusters)for(const im of cl.images)
 if(im.path===p)return im;}
function setImg(ep,v){const im=findImg(decodeURIComponent(ep));
 if(!v)return;im.final=v;im.status='species';render();}
function setStatus(ep,s){findImg(decodeURIComponent(ep)).status=s;render();}
function applyCluster(cid,v){const cl=DATA.clusters.find(c=>c.cluster_id===cid);
 cl.images.forEach(im=>{im.final=v;im.status='species';});render();}
function applyTyped(cid){const v=document.getElementById('cl'+cid).value.trim();
 if(v)applyCluster(cid,v);}
function markCluster(cid,s){DATA.clusters.find(c=>c.cluster_id===cid)
 .images.forEach(im=>im.status=s);render();}
function updateStat(){const{h,n,j}=counts();
 document.getElementById('stat').textContent=
  `${h} human-corrected · ${n} noise · ${j} junk`;}
function payload(){const ch={};DATA.clusters.forEach(c=>c.images.forEach(im=>{
 ch[im.path]={final:im.final,status:im.status};}));return ch;}
async function saveAll(){await fetch('/api/save',{method:'POST',
 headers:{'Content-Type':'application/json'},
 body:JSON.stringify({changes:payload()})});alert('Progress saved in memory.');}
async function finalize(){
 if(!confirm('Write the clean dataset now? Existing export is overwritten.'))return;
 await fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({changes:payload()})});
 const r=await (await fetch('/api/finalize',{method:'POST'})).json();
 alert(`Done.\nClean dataset: ${r.clean_dir}\nSpecies folders: ${r.species_folders}`
  +`\nImages kept: ${r.kept}\nHuman-corrected: ${r.human_corrected}`
  +`\nNoise: ${r.noise}  Junk: ${r.junk}`);
}
load();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            self._send(200, "text/html; charset=utf-8", PAGE.encode())
        elif u.path == "/api/state":
            with LOCK:
                body = json.dumps(build_state_json()).encode()
            self._send(200, "application/json", body)
        elif u.path == "/thumb":
            path = parse_qs(u.query).get("path", [""])[0]
            if path not in STATE["valid_paths"]:
                self._send(403, "text/plain", b"forbidden")
                return
            try:
                self._send(200, "image/jpeg", make_thumb(path))
            except Exception as e:
                self._send(500, "text/plain", str(e).encode())
        else:
            self._send(404, "text/plain", b"not found")

    def do_POST(self):
        u = urlparse(self.path)
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n) if n else b"{}"
        if u.path == "/api/save":
            data = json.loads(raw or b"{}")
            with LOCK:
                for p, v in data.get("changes", {}).items():
                    if p in STATE["valid_paths"]:
                        STATE["final"][p] = v.get("final", STATE["final"][p])
                        STATE["status"][p] = v.get("status", STATE["status"][p])
            self._send(200, "application/json", b'{"ok":true}')
        elif u.path == "/api/finalize":
            with LOCK:
                result = finalize()
            self._send(200, "application/json", json.dumps(result).encode())
        else:
            self._send(404, "text/plain", b"not found")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations",
                    default="results/Annotation_eumoth/annotations.csv")
    ap.add_argument("--out", default=None)
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    out_dir = args.out or os.path.join(
        os.path.dirname(args.annotations), "curated_dataset")
    os.makedirs(out_dir, exist_ok=True)
    load_state(args.annotations, out_dir)

    df = STATE["df"]
    print(f"Loaded {len(df)} images, {df['cluster_id'].nunique()} clusters")
    print(f"Output dir : {out_dir}")
    print(f"Open       : http://localhost:{args.port}")
    print("Ctrl+C to stop.")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
