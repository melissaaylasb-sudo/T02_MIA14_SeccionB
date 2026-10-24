"""Exporta tablas LaTeX desde resultados existentes; no ajusta ningún modelo."""
from pathlib import Path
import csv
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / 'tablas'
OUT.mkdir(exist_ok=True)
SOURCE = ROOT / 'reports/feature_engineering/tables'

def read(name):
    with (SOURCE / name).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def num(v, digits=3):
    return f'{float(v):.{digits}f}'.replace('.', '{,}')

summary = read('comparacion_experimentos.csv')
primary = {r['experimento']: r for r in summary if r['protocolo']=='configuracion'}
secondary = {r['experimento']: r for r in summary if r['protocolo']=='familia'}
decisions = {'A':'Conservar', 'B':'Evaluar', 'C':'Descartar', 'D':'Descartar'}
lines=[]
for k in 'ABCD':
    r=primary[k]
    vals=[k, r['n_features_min'], num(r['MAE_N']), num(r['RMSE_N']), num(r['R2_OOF']),
          num(r['delta_MAE_vs_A_N']), num(r['MAE_fold_sd_N']),num(r['tiempo_s']),decisions[k]]
    lines.append(' & '.join(vals)+r' \\')
(OUT/'principal.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
lines=[]
for k in 'ABCD':
    r=secondary[k]
    lines.append(' & '.join([k,num(r['MAE_N']),num(r['RMSE_N']),num(r['MedAE_N']),num(r['R2_OOF'])])+r' \\')
(OUT/'familias.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
folds=[r for r in read('metricas_folds.csv') if r['protocolo']=='configuracion' and r['experimento'] in 'ABCD']
groups=sorted({int(r['fold']) for r in folds})
lines=[]
for f in groups:
    rows={r['experimento']:r for r in folds if int(r['fold'])==f}
    a,b=rows['A'],rows['B']
    group=a['grupo'].replace('|',r' / ').replace('.',',')
    delta=float(b['MAE_N'])-float(a['MAE_N'])
    vals=[str(f),group,a['n_test']]+[num(rows[k]['MAE_N']) for k in 'ABCD']+[num(delta)]
    lines.append(' & '.join(vals)+r' \\')
(OUT/'folds.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
models=[r for r in read('comparacion_modelos.csv') if r['protocolo']=='configuracion']
lines=[]
for r in models:
    vals=[r['nombre_modelo']]+[num(r[c]) for c in ['MAE_N','RMSE_N','R2_OOF','tiempo_s']]
    lines.append(' & '.join(vals)+r' \\')
(OUT/'modelos.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
manifest={
    'purpose':'Tablas derivadas de CSV de la ejecución histórica; sin reentrenamiento',
    'reference_commit':'8fdb7961d7eaaaebd9651d0b9a044168893d448a',
    'source_sha256':{str((SOURCE/n).relative_to(ROOT)).replace('\\','/'):hashlib.sha256((SOURCE/n).read_bytes()).hexdigest()
      for n in ['comparacion_experimentos.csv','metricas_folds.csv','comparacion_modelos.csv']},
    'output_sha256':{str(p.relative_to(HERE)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.glob('*.tex'))}
}
(HERE/'trazabilidad_tablas.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Tablas exportadas y fuentes identificadas por SHA-256.')
