"""Verifica el PDF compilado y contrasta las fuentes existentes sin entrenar."""
from pathlib import Path
import hashlib
import json
import re
import fitz
import numpy as np
import pandas as pd
from PIL import Image

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
JOB='Informe_Sprint_Semana_6'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    checks=[]
    def check(condition,description):
        if not condition:
            raise AssertionError(description)
        checks.append(description)

    pdf_path=HERE/(JOB+'.pdf')
    document=fitz.open(pdf_path)
    aux=(HERE/(JOB+'.aux')).read_text(encoding='utf-8')
    log=(HERE/(JOB+'.log')).read_text(encoding='utf-8',errors='replace')
    def page_label(name):
        return int(re.search(r'\\newlabel\{'+re.escape(name)+r'\}\{\{[^}]*\}\{(\d+)\}',aux)[1])
    first,last=page_label('body:first'),page_label('body:last')
    substantive=last-first+1
    check(12<=substantive<=16,'Extensión sustantiva: 12–16 páginas')
    for pattern in ['Overfull \\hbox','Overfull \\vbox','undefined references','undefined on input','Rerun to get']:
        check(pattern not in log,'Sin diagnóstico pendiente: '+pattern)
    for i,page in enumerate(document,1):
        check(abs(page.rect.width-595.276)<1 and abs(page.rect.height-841.89)<1,f'Página {i}: A4')
        text=page.get_text()
        check(len(text.strip())>30,f'Página {i}: sin página vacía')
        check('??' not in text and '\ufffd' not in text,f'Página {i}: sin marcadores indefinidos o caracteres de reemplazo')
        for b in page.get_text('blocks'):
            check(b[0]>=15 and b[1]>=15 and b[2]<=page.rect.width-15 and b[3]<=page.rect.height-15,f'Página {i}: bloque dentro del soporte')
    bbl=(HERE/(JOB+'.bbl')).read_text(encoding='utf-8')
    references=len(re.findall(r'\\bibitem\{',bbl))
    check(references>=7,'Referencias relevantes resueltas')
    captions={'figures':len(re.findall(r'\\newlabel\{fig:',aux)),
              'tables':len(re.findall(r'\\@writefile\{lot\}',aux))}
    check(captions['figures']==6 and captions['tables']>=10,'EDA y experimentos con figuras y tablas verificadas')
    for path in sorted((HERE/'figures').glob('*.png')):
        with Image.open(path) as im:
            check((im.size==(398,502)) if path.name=='logo_uni.png' else im.width>=1600,f'Resolución de figura: {path.name}')
    traces=[json.loads((HERE/'trazabilidad_tablas.json').read_text(encoding='utf-8')),
            json.loads((HERE/'figures/manifest_figuras.json').read_text(encoding='utf-8'))]
    for manifest in traces:
        for name,sha in manifest['source_sha256'].items():
            check(digest(ROOT/name)==sha,'Hash de fuente: '+name)
    for name,sha in traces[0]['output_sha256'].items():
        check(digest(HERE/name)==sha,'Hash tabla: '+name)
    for row in traces[1]['figures']:
        check(digest(HERE/row['file'])==row['sha256'],'Hash figura: '+row['file'])
    check(digest(ROOT/traces[1]['generator'])==traces[1]['generator_sha256'],'Hash generador de figuras')
    eda=json.loads((HERE/'manifest_eda.json').read_text(encoding='utf-8'))
    for category in ['inputs','source_code']:
        for name,sha in eda[category].items():
            check(digest(ROOT/name)==sha,'Hash EDA fuente: '+name)
    for category in ['figures','tables']:
        for name,sha in eda[category].items():
            check(digest(ROOT/name)==sha,'Hash EDA salida: '+name)
    tables=ROOT/'reports/feature_engineering/tables'
    summary=pd.read_csv(tables/'comparacion_experimentos.csv')
    predictions=pd.read_csv(tables/'predicciones_oof.csv')
    results=[]
    for _,row in summary.iterrows():
        p=predictions[(predictions.protocolo==row.protocolo)&(predictions.experimento==row.experimento)]
        check(p.Case_n.is_unique and len(p)==26,f'Cobertura OOF: {row.protocolo}/{row.experimento}')
        y=p.observado_N.to_numpy(); e=y-p.predicho_N.to_numpy()
        actual={'MAE_N':np.abs(e).mean(),'RMSE_N':np.sqrt(np.mean(e**2)),
                'MedAE_N':np.median(np.abs(e)), 'R2_OOF':1-np.sum(e**2)/np.sum((y-y.mean())**2)}
        for metric,v in actual.items():
            check(np.isclose(v,row[metric],rtol=1e-11,atol=1e-11),f'Métrica {row.protocolo}/{row.experimento}/{metric}')
        results.append({'protocolo':row.protocolo,'experimento':row.experimento,**actual})
    models=pd.read_csv(tables/'comparacion_modelos.csv')
    model_pred=pd.read_csv(tables/'predicciones_modelos_oof.csv')
    model_results=[]
    for _,row in models.iterrows():
        p=model_pred[(model_pred.protocolo==row.protocolo)&(model_pred.modelo==row.modelo)]
        check(p.Case_n.is_unique and len(p)==26,f'Cobertura modelos: {row.protocolo}/{row.modelo}')
        y=p.observado_N.to_numpy(); e=y-p.predicho_N.to_numpy()
        actual={'MAE_N':np.abs(e).mean(),'RMSE_N':np.sqrt(np.mean(e**2)),
                'MedAE_N':np.median(np.abs(e)), 'R2_OOF':1-np.sum(e**2)/np.sum((y-y.mean())**2)}
        for metric,v in actual.items():
            check(np.isclose(v,row[metric],rtol=1e-11,atol=1e-11),f'Métrica modelo {row.protocolo}/{row.modelo}/{metric}')
        model_results.append({'protocolo':row.protocolo,'modelo':row.modelo,**actual})
    all_text='\n'.join(page.get_text() for page in document)
    check('Procedencia y trazabilidad' not in all_text,'Sección de procedencia y trazabilidad retirada')
    check('Arquitectura de los modelos' not in all_text,'Sin desarrollo teorico de algoritmos')
    check('Modelos utilizados y alcance de sus experimentos' in all_text,'Modelos presentados en tabla')
    check('Variante 3' not in all_text,'Experimento centrado en dos variantes')
    check('Diccionario' in all_text or 'diccionario' in all_text,'Diccionario de entradas y salidas presente')
    check(not re.search(r'\b(?:26|27)\s+(?:casos|registros|observaciones|muestras)',all_text,re.I),'Sin conteos de observaciones en el informe')
    audit=json.loads((HERE/'auditoria_resultados.json').read_text(encoding='utf-8'))
    check(audit['status']=='passed','Auditoría científica detallada satisfactoria')
    report={
        'status':'passed','scope':'Comprobaciones automáticas del documento y sus fuentes; no reemplazan revisión visual.',
        'reference_commit':'8fdb7961d7eaaaebd9651d0b9a044168893d448a',
        'pdf':pdf_path.name,'pdf_sha256':digest(pdf_path),'pages_total':len(document),
        'pages_substantive':substantive,'body_folios':[first,last],
        'references':references,**captions,'checks_count':len(checks),'checks':checks,'metrics_recomputed':results,'model_metrics_recomputed':model_results,
        'compiler':'pdfTeX 1.40.25 / MiKTeX 24.1; BibTeX 0.99d; IEEEtran.bst 1.14',
        'visual_review':{'status':'not_recorded_by_automatic_verifier',
                         'note':'La revisión visual se documenta por separado para el hash exacto del PDF.'}
    }
    target=HERE/'verificacion_documento.json'
    if target.exists():
        previous=json.loads(target.read_text(encoding='utf-8'))
        if previous.get('pdf_sha256')==report['pdf_sha256']:
            for key in ['visual_review','editor_preview','standalone_package_check']:
                if key in previous:
                    report[key]=previous[key]
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['status','pages_total','pages_substantive','references','figures','tables','checks_count']},ensure_ascii=True))

if __name__=='__main__':
    main()
