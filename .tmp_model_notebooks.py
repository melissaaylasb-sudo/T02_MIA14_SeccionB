from pathlib import Path
import textwrap
import nbformat as nbf
from nbclient import NotebookClient
from nbconvert import HTMLExporter

ROOT = Path(__file__).resolve().parent
md = lambda source: nbf.v4.new_markdown_cell(textwrap.dedent(source).strip())
code = lambda source: nbf.v4.new_code_cell(textwrap.dedent(source).strip())
setup = '''
from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents]
             if (p / 'src' / 'experimento.py').is_file()), None)
if ROOT is None:
    raise FileNotFoundError('Abra el notebook desde este repositorio.')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.experimento import read_config, inspect_plan, load_model_data, find_completed_run, run_experiment
from src.particiones import nested_plan
from src.modelo_baseline import build_baseline, evaluate_regression
from dataclasses import asdict

config, config_hash = read_config(ROOT / 'config' / 'model_config.yml')
X, y, trace, groups, quality, data_path, report_path = load_model_data(config)
pd.set_option('display.float_format', lambda value: f'{value:.3f}')
plt.rcParams.update({'figure.dpi': 110, 'axes.spines.top': False, 'axes.spines.right': False})
print('Datos:', data_path.relative_to(ROOT))
print('Objetivo:', y.name, '[N]')
'''

def save_execute(name, cells):
    path = ROOT / 'notebooks' / (name + '.ipynb')
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata.update(kernelspec={'display_name': 'Python 3', 'language': 'python', 'name': 'python3'})
    nbf.write(nb, path)
    NotebookClient(nb, timeout=1800, kernel_name='python3', allow_errors=False).execute(cwd=str(ROOT))
    nbf.validate(nb)
    nbf.write(nb, path)
    exporter = HTMLExporter()
    exporter.exclude_input = True
    html, _ = exporter.from_notebook_node(nb)
    # En HTML las transiciones abren el siguiente reporte; el notebook conserva su enlace original.
    for notebook in (ROOT / 'notebooks').glob('*.ipynb'):
        html = html.replace('href="' + notebook.name + '"', 'href="' + notebook.stem + '.html"')
    (ROOT / 'reports' / (name + '.html')).write_text(html, encoding='utf-8')
    counts = [c.execution_count for c in nb.cells if c.cell_type == 'code']
    print(name, 'ejecutado:', counts, flush=True)

save_execute('03_Diseno_de_Particiones_y_Preprocesamiento', [
md('''
# 3. Diseño de particiones y preprocesamiento

**Objetivo:** materializar el protocolo preliminar antes de entrenar. Se usa validación
cruzada anidada: 5 particiones externas para evaluar y 3 internas para ajustar y seleccionar.
Cada caso aparece exactamente una vez en evaluación externa.

La unidad de evaluación es el caso de la misma campaña, bajo independencia provisional.
La fuente no declara lotes ni réplicas. Los identificadores únicos permiten rastrear los casos,
pero no demuestran independencia experimental. Si se confirman grupos dependientes, corresponde
repetir la evaluación con particiones agrupadas.
'''), code(setup),
md('## 3.1 Configuración y justificación registradas'),
code('''
status = inspect_plan(config)
display(pd.DataFrame([{
    'estado': status['status'], 'estrategia': config['protocol']['strategy'],
    'folds externos': config['protocol']['outer_splits'],
    'folds internos': config['protocol']['inner_splits'], 'semilla': config['protocol']['seed'],
    'familias candidatas': len(status['candidates']),
}]))
display(Markdown(config['protocol']['rationale']))
print('La configuración permite entrenamiento; esta inspección no ajusta modelos.')
'''),
md('## 3.2 Particiones y verificación del aislamiento'),
code('''
plan = nested_plan(X, config['protocol'], groups)
display(pd.DataFrame([{
    'fold': p['fold'], 'casos entrenamiento': len(p['train']),
    'casos evaluación': len(p['test']), 'particiones internas': len(p['inner']),
    'casos reservados': ', '.join(trace.iloc[p['test']]['Case_n'].astype(str)),
} for p in plan]))
all_test = np.concatenate([p['test'] for p in plan])
assert sorted(all_test.tolist()) == list(range(len(X)))
for p in plan:
    assert not set(p['train']) & set(p['test'])
    for a, b in p['inner']:
        assert not set(p['train'][a]) & set(p['train'][b])
        assert not set(p['train'][a]) & set(p['test'])
        assert not set(p['train'][b]) & set(p['test'])
print('Verificado: cada caso se evalúa una vez; no hay solapamientos entre train y evaluación.')
'''),
md('''
## 3.3 Preprocesamiento dentro del entrenamiento

Cada pipeline ajusta imputación de predictores, eliminación de constantes y escalamiento
dentro de su propio conjunto de entrenamiento. La carga de primera falla nunca se imputa.
Los identificadores y las respuestas posensayo no entran en X.
'''),
code('''
from src.modelos import candidate_specs
specs = candidate_specs(config)
display(pd.DataFrame([{
    'familia': family,
    'imputación': 'mediana de train',
    'constantes': 'eliminación dentro de train',
    'escalamiento': type(pipe.named_steps['scale']).__name__ if pipe.named_steps['scale'] != 'passthrough' else 'no requerido',
    'estimador': type(pipe.named_steps['model']).__name__,
} for family, (pipe, _) in specs.items()]))
assert y.name not in X.columns and 'Case_n' not in X.columns
print('X contiene exclusivamente los predictores declarados en el esquema.')
'''),
md('''
## 3.4 Alcance de esta evaluación

Este protocolo aproxima la predicción de otros casos comparables de la misma campaña.
No mide transferencia a nuevas campañas, materiales o configuraciones completamente inéditas.
El EDA ya se realizó sobre este conjunto, por lo que esta evaluación conserva carácter preliminar.

Siguiente: [04 · Baseline de referencia](04_Baseline_de_Referencia.ipynb).
''')])

save_execute('04_Baseline_de_Referencia', [
md('''
# 4. Baseline de referencia

**Objetivo:** medir el error de una referencia simple en las mismas particiones externas
que usarán los modelos. En cada fold se aprende la mediana exclusivamente con su entrenamiento
y se usa ese valor para los casos reservados. El MAE de esta referencia permite comprobar
si los descriptores geométricos aportan mejora predictiva.
'''), code(setup),
md('## 4.1 Ajuste y predicción fuera de muestra'),
code('''
plan = nested_plan(X, config['protocol'], groups)
predictions, metrics = [], []
for p in plan:
    train, test = p['train'], p['test']
    estimator = build_baseline().fit(np.zeros((len(train), 1)), y.iloc[train])
    pred = estimator.predict(np.zeros((len(test), 1)))
    frame = trace.iloc[test].copy()
    frame['fold'] = p['fold']
    frame['observed'] = y.iloc[test].to_numpy()
    frame['predicted'] = pred
    predictions.append(frame)
    metrics.append({'fold': p['fold'], 'mediana_train_N': float(np.median(y.iloc[train])),
                    **asdict(evaluate_regression(y.iloc[test], pred))})
baseline_oof = pd.concat(predictions, ignore_index=True)
display(pd.DataFrame(metrics))
'''),
md('## 4.2 Métricas agregadas y predicciones'),
code('''
baseline_metrics = asdict(evaluate_regression(baseline_oof['observed'], baseline_oof['predicted']))
display(pd.DataFrame([baseline_metrics], index=['baseline OOF']))
display(baseline_oof.sort_values('Case_n', key=lambda s: pd.to_numeric(s)).head(10))
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(baseline_oof['observed'], baseline_oof['predicted'], color='#0f766e')
bounds = [min(baseline_oof[['observed', 'predicted']].min()), max(baseline_oof[['observed', 'predicted']].max())]
ax.plot(bounds, bounds, '--', color='gray')
ax.set(xlabel='Carga observada [N]', ylabel='Carga predicha [N]', title='Baseline: predicción fuera de muestra')
plt.tight_layout(); plt.show()
'''),
md('''
## 4.3 Interpretación

MAE, RMSE y MedAE se expresan en newtons. R² puede ser negativo: significa que, en esta
evaluación, el error cuadrático supera al de predecir la media observada del conjunto evaluado.
Los folds tienen medianas de entrenamiento distintas; no se emplea una mediana global.

Siguiente: [05 · Entrenamiento, ajuste y validación](05_Entrenamiento_Ajuste_y_Validacion.ipynb).
''')])

save_execute('05_Entrenamiento_Ajuste_y_Validacion', [
md('''
# 5. Entrenamiento, ajuste y validación

**Corrida preliminar con datos reales.** Se comparan Ridge, Elastic Net, SVR-RBF, árbol,
bosque aleatorio y proceso gaussiano, además del baseline. La red neuronal sigue como extensión
exploratoria fuera de esta comparación.

En cada fold externo, la CV interna selecciona hiperparámetros y familia por MAE. Después se
reajusta usando su entrenamiento externo y se predicen los casos reservados. `selected` representa
ese procedimiento completo: puede elegir una familia distinta en cada fold.

**Para repetir:** ejecutar todas las celdas. Si existe una corrida completa de los mismos datos y
configuración se verifica y reutiliza. Cambiar `FORZAR_NUEVA_EJECUCION` a `True` inicia una nueva
corrida fechada. También puede ejecutarse `python -m src.experimento --run --fit-final` desde la raíz.
'''), code(setup),
md('## 5.1 Comprobar preparación'),
code('''
status = inspect_plan(config)
assert status['status'] == 'ready_for_reviewed_execution'
display(pd.DataFrame({'familia': list(status['candidates']), 'estado': 'lista para CV anidada'}))
print('Protocolo:', config['protocol']['outer_splits'], 'folds externos ×', config['protocol']['inner_splits'], 'internos')
'''),
md('## 5.2 Ejecutar o recuperar una corrida íntegra'),
code('''
FORZAR_NUEVA_EJECUCION = False
run_dir = None if FORZAR_NUEVA_EJECUCION else find_completed_run(config, config_hash, require_final=True)
if run_dir is None:
    print('Entrenando seis familias, baseline y reajuste final...')
    run_dir = run_experiment(config, config_hash, fit_final=True)
    print('Nueva corrida completada.')
else:
    print('Corrida real recuperada; hashes de datos, configuración y artefactos verificados.')
print('Ejecución:', run_dir.relative_to(ROOT))
evaluation = json.loads((run_dir / 'evaluation.json').read_text(encoding='utf-8'))
manifest = json.loads((run_dir / 'experiment_manifest.json').read_text(encoding='utf-8'))
print('Estado:', manifest['status'], '| Finalizada:', manifest['completed_at_utc'])
'''),
md('## 5.3 Modelos seleccionados y comparación fuera de muestra'),
code('''
selections = pd.DataFrame([{'fold': item['fold'], 'familia elegida por CV interna': item['selected_by_inner_cv']}
                           for item in evaluation['selection_by_fold']])
display(selections)
metrics = pd.DataFrame(evaluation['pooled_oof_metrics']).T.rename_axis('procedimiento')
display(metrics)
print('MAE, RMSE y MedAE: N. R²: sin unidad.')
print('Las métricas por familia son diagnósticas; no se usa el mejor error externo para seleccionar.')
'''),
md('## 5.4 Reajuste para uso posterior y advertencias'),
code('''
final_info = json.loads((run_dir / 'final_model_info.json').read_text(encoding='utf-8'))
display(pd.DataFrame([
    {'artefacto': 'final_model.joblib', 'descripción': 'pipeline reajustado con todos los datos', 'familia': final_info['family']},
    {'artefacto': 'oof_predictions.csv', 'descripción': 'predicciones externas para calcular métricas', 'familia': 'todos los procedimientos'},
    {'artefacto': 'experiment_manifest.json', 'descripción': 'versiones y hashes de la ejecución', 'familia': '—'},
]))
print(final_info['note'])
print('Advertencias durante CV externa:', len(evaluation['warnings']))
print('Advertencias del reajuste final:', len(final_info['warnings']))
if evaluation['warnings']:
    display(pd.DataFrame(evaluation['warnings']).drop_duplicates())
if final_info['warnings']:
    display(pd.DataFrame({'advertencia': final_info['warnings']}))
'''),
md('''
El pipeline reajustado se selecciona por CV sobre todos los datos y sirve como artefacto preliminar
para inferencia. Sus predicciones sobre datos de entrenamiento no constituyen una evaluación nueva.
Las métricas principales pertenecen al procedimiento `selected` fuera de muestra.

Siguiente: [06 · Evaluación e interpretabilidad](06_Evaluacion_Final_e_Interpretabilidad.ipynb).
''')])

save_execute('06_Evaluacion_Final_e_Interpretabilidad', [
md('''
# 6. Evaluación e interpretabilidad de la corrida preliminar

**Entrada:** resultados completos del cuaderno 05. Se comprueba su integridad antes de mostrar
métricas. Esta es la etapa final del flujo de cómputo; la validación científica sigue siendo preliminar.

La cifra principal corresponde al procedimiento `selected`: selección de familia e hiperparámetros
dentro de cada CV interna, seguida de predicción externa. Las cifras por familia sirven para diagnóstico.
'''), code(setup),
md('## 6.1 Recuperar resultados comprobados'),
code('''
RUN = find_completed_run(config, config_hash, require_final=True)
if RUN is None:
    raise FileNotFoundError('Ejecute 05 o python -m src.experimento --run --fit-final')
evaluation = json.loads((RUN / 'evaluation.json').read_text(encoding='utf-8'))
manifest = json.loads((RUN / 'experiment_manifest.json').read_text(encoding='utf-8'))
metrics = pd.DataFrame(evaluation['pooled_oof_metrics']).T.rename_axis('procedimiento')
fold_metrics = pd.read_csv(RUN / 'fold_metrics.csv')
oof = pd.read_csv(RUN / 'oof_predictions.csv')
selected = oof.loc[oof['procedure'] == 'selected'].copy()
print('Corrida verificada:', RUN.relative_to(ROOT))
display(metrics)
'''),
md('## 6.2 Comparación con el baseline y estabilidad entre folds'),
code('''
paired = pd.read_csv(RUN / 'paired_mae.csv')
display(paired[['fold', 'baseline', 'selected', 'baseline_minus_selected_mae']])
gain = metrics.loc['baseline', 'mae'] - metrics.loc['selected', 'mae']
display(Markdown(f"**MAE del procedimiento: {metrics.loc['selected', 'mae']:.2f} N.** "
                 f"Baseline: {metrics.loc['baseline', 'mae']:.2f} N. Reducción: {gain:.2f} N."))
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
metrics['mae'].plot.barh(ax=axes[0], color=['#94a3b8' if c == 'baseline' else '#0f766e' if c == 'selected' else '#2563eb' for c in metrics.index])
axes[0].set(xlabel='MAE OOF [N]', ylabel='', title='Comparación descriptiva')
axes[1].plot(paired['fold'], paired['baseline'], 'o-', label='Baseline', color='#94a3b8')
axes[1].plot(paired['fold'], paired['selected'], 'o-', label='Selección interna', color='#0f766e')
axes[1].set(xlabel='Fold externo', ylabel='MAE [N]', title='Estabilidad por partición', xticks=paired['fold'])
axes[1].legend(); plt.tight_layout(); plt.show()
'''),
md('''
Los entrenamientos de los folds comparten observaciones. Su dispersión describe sensibilidad a
la partición; no equivale a un intervalo de confianza basado en muestras independientes.
'''),
md('## 6.3 Predicciones, residuos y casos con mayor error'),
code('''
selected['error_absoluto_N'] = selected['residual'].abs()
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].scatter(selected['observed'], selected['predicted'], c=selected['fold'], cmap='viridis')
bounds = [min(selected[['observed','predicted']].min()), max(selected[['observed','predicted']].max())]
axes[0].plot(bounds, bounds, '--', color='gray')
axes[0].set(xlabel='Observado [N]', ylabel='Predicho [N]', title='Predicción externa')
axes[1].scatter(selected['predicted'], selected['residual'], color='#2563eb')
axes[1].axhline(0, linestyle='--', color='gray')
axes[1].set(xlabel='Predicho [N]', ylabel='Observado − predicho [N]', title='Residuos')
axes[2].hist(selected['residual'], bins='auto', color='#0f766e', edgecolor='white')
axes[2].axvline(0, linestyle='--', color='gray')
axes[2].set(xlabel='Residuo [N]', ylabel='Frecuencia', title='Distribución del error')
plt.tight_layout(); plt.show()
display(selected[['Case_n','fold','selected_family','observed','predicted','residual','error_absoluto_N']]
        .sort_values('error_absoluto_N', ascending=False).head(8))
print(f"Residuo medio: {selected['residual'].mean():.2f} N; positivo significa subestimación media.")
'''),
md('''
## 6.4 Interpretación y sensibilidad

La permutación mide cuánto aumenta el MAE externo al alterar una variable, después de seleccionar
el modelo. Se muestran los folds por separado porque pueden usar familias distintas. Las variables
correlacionadas y las restricciones geométricas limitan esta interpretación: permutar puede producir
combinaciones poco plausibles. Una importancia pequeña no prueba irrelevancia física.
'''),
code('''
importance_frames, coefficients = [], []
for item in evaluation['selection_by_fold']:
    folder = RUN / f"fold_{item['fold']:02d}"
    imp = pd.read_csv(folder / 'selected_permutation_importance.csv')
    imp['fold'] = item['fold']; imp['familia'] = item['selected_by_inner_cv']
    importance_frames.append(imp)
    coef = pd.read_csv(folder / 'selected_coefficients.csv')
    if not coef.empty:
        coef['fold'] = item['fold']; coef['familia'] = item['selected_by_inner_cv']
        coefficients.append(coef)
importance = pd.concat(importance_frames, ignore_index=True)
fig, axes = plt.subplots(1, len(importance_frames), figsize=(18, 5), sharex=False)
for ax, frame in zip(axes, importance_frames):
    frame = frame.sort_values('mae_increase_mean')
    ax.barh(frame['feature'], frame['mae_increase_mean'], color='#2563eb')
    ax.axvline(0, color='gray', linewidth=.7)
    ax.set_title(f"Fold {frame['fold'].iloc[0]}: {frame['familia'].iloc[0]}", fontsize=9)
    ax.set_xlabel('Aumento de MAE [N]'); ax.tick_params(axis='y', labelsize=6)
plt.tight_layout(); plt.show()
if coefficients:
    display(pd.concat(coefficients, ignore_index=True))
else:
    print('Los modelos seleccionados no exponen coeficientes lineales.')
'''),
md('''
## 6.5 Artefacto ajustado y ejemplo de inferencia

El ejemplo comprueba que el pipeline guardado puede cargarse y predecir. Usa una geometría de la
tabla de entrenamiento; por ese motivo no se incluye como nueva evidencia de desempeño.
'''),
code('''
import joblib
pipeline = joblib.load(RUN / 'final_model.joblib')
final_info = json.loads((RUN / 'final_model_info.json').read_text(encoding='utf-8'))
example = X.iloc[[0]].copy()
prediction = float(pipeline.predict(example)[0])
display(example)
print('Familia reajustada:', final_info['family'])
print(f'Predicción de comprobación: {prediction:.2f} N')
print('Artefacto:', (RUN / 'final_model.joblib').relative_to(ROOT))
print(final_info['note'])
'''),
md('## 6.6 Exportación y conclusiones del avance'),
code('''
REPORTS = ROOT / 'reports'
REPORTS.mkdir(exist_ok=True)
metrics.to_csv(REPORTS / 'metricas_modelos_preliminares.csv', encoding='utf-8')
summary = {
    'status': 'preliminary_completed', 'run_id': RUN.name,
    'configuration_sha256': config_hash,
    'dataset_sha256': manifest['dataset_sha256'],
    'quality_report_sha256': manifest['quality_report_sha256'],
    'completed_at_utc': manifest['completed_at_utc'],
    'protocol': config['protocol'],
    'pooled_oof_metrics': evaluation['pooled_oof_metrics'],
    'selection_by_fold': [{'fold': s['fold'], 'family': s['selected_by_inner_cv']} for s in evaluation['selection_by_fold']],
    'refitted_family': final_info['family'],
    'refit_note': final_info['note'],
    'interpretation': evaluation['interpretation'],
}
(REPORTS / 'resumen_modelado_preliminar.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
display(Markdown(
    f"El procedimiento de selección interna alcanza **MAE {metrics.loc['selected', 'mae']:.2f} N**, "
    f"**RMSE {metrics.loc['selected', 'rmse']:.2f} N** y **R² {metrics.loc['selected', 'r2']:.3f}** "
    f"en las predicciones externas de esta corrida. Se guardó un pipeline de la familia **{final_info['family']}** "
    "reajustado con todos los datos; sus métricas propias no se infieren de esta cifra."
))
print('Exportados: reports/metricas_modelos_preliminares.csv y reports/resumen_modelado_preliminar.json')
'''),
md('''
La evaluación es preliminar y condicionada a la independencia entre casos de esta campaña.
La exploración previa del conjunto y la dependencia geométrica requieren una validación posterior
con datos nuevos o grupos confirmados. No se presenta como desempeño garantizado fuera del dominio.

**Reproducir:** `python -m src.experimento --run --fit-final` y luego ejecutar este cuaderno.
Los reportes CSV/JSON y los HTML quedan disponibles para revisión sin cargar archivos de modelo.
''')])
