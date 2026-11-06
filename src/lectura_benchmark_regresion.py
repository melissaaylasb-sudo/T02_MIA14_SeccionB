"""Lectura reproducible y recomendaciones exploratorias desde predicciones externas.

No ajusta estimadores. La política de adopción precede a la ejecución y separa
la comparación entre los seis regresores de la utilidad frente a la media Y.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NAMES = {"Lineal_multiple":"Regresión lineal múltiple","Arbol_decision":"Árbol de decisión",
         "KNN":"K vecinos","SVR_RBF":"SVR-RBF","XGBoost":"XGBoost","CatBoost":"CatBoost",
         "Ridge_historico":"Ridge histórico","Dummy_mediana":"Mediana de entrenamiento",
         "Media_por_nivel":"Media por nivel Y","Tuning_rango_1":"Primer rango ajustado",
         "Tuning_rango_2":"Segundo rango ajustado","Selector_tuning":"Selector interno ajustado"}
DOMAIN = {"configuracion":"Configuraciones nominales","familia":"Nivel Y completo"}
GROUP = {"SVR_RBF":"A · SVR","XGBoost":"A · boosting","CatBoost":"A · boosting",
         "Lineal_multiple":"B · lineal","KNN":"B · distancias","Arbol_decision":"B · árbol"}
CATEGORY_NAMES = {"lineal":"Lineal (único candidato)", "distancias":"Vecinos (único candidato)",
                  "arboles":"Menor MAE entre los tres modelos de árboles", "grupo_A_literatura":"Menor MAE del grupo A",
                  "menor_MAE_inicial":"Menor MAE de los seis fijos","menor_DE_entre_iniciales":"Menor DE entre los seis fijos",
                  "menor_MAE_procedimientos_incluido_tuning":"Menor MAE incluyendo procedimientos ajustados",
                  "menor_MAE_toda_tabla_incluidos_controles":"Menor MAE incluyendo controles","compromiso":"Compromiso recomendado entre regresores"}
def es(x: float, d: int = 2) -> str:
    """Número de presentación; los CSV conservan precisión completa."""
    return f"{x:.{d}f}".replace(".", ",")
def name(x: str) -> str:
    return NAMES.get(x,x)
def read(p: Path) -> pd.DataFrame:
    return pd.read_csv(p, float_precision="round_trip")
def table(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join(["| "+" | ".join(headers)+" |","|"+"---|"*len(headers),
                      *["| "+" | ".join(map(str,row))+" |" for row in rows]])

def contraste(oof: pd.DataFrame, candidate: str, base: str) -> dict[str, Any]:
    """Errores emparejados y sensibilidad al grupo en un solo protocolo."""
    if oof.protocolo.nunique()!=1:
        raise ValueError("Comparar un protocolo por vez")
    cols=["Case_n","fold","observado_N","predicho_N"]
    a=oof[oof.procedimiento.eq(candidate)][cols].copy()
    b=oof[oof.procedimiento.eq(base)][cols].copy()
    p=a.merge(b,on=["Case_n","fold"],suffixes=("_a","_b"),validate="one_to_one")
    if len(p)!=len(a) or len(p)!=len(b) or not len(p):
        raise ValueError("Cobertura OOF desigual")
    y=p.observado_N_a.to_numpy()
    if not np.allclose(y,p.observado_N_b,atol=1e-12,rtol=0):
        raise ValueError("Respuesta no comparable")
    ea=np.abs(p.predicho_N_a-y);eb=np.abs(p.predicho_N_b-y)
    d=pd.Series(ea-eb,index=p.index)
    gm=d.groupby(p.fold).mean()
    omit=np.array([d[p.fold.ne(f)].mean() for f in gm.index])
    return {"delta_MAE_N":float(d.mean()),"delta_RMSE_N":float(np.sqrt(np.mean(ea**2))-np.sqrt(np.mean(eb**2))),
            "grupos_mejora":int((gm < -1e-10).sum()),"grupos_empate":int((gm.abs()<=1e-10).sum()),
            "grupos":len(gm),"fraccion_grupos_mejora":float((gm < -1e-10).mean()),
            "delta_al_omitir_grupo_min_N":float(omit.min()),"delta_al_omitir_grupo_max_N":float(omit.max())}

def adopta(c: dict[str,Any], margin: float) -> bool:
    return (c["delta_MAE_N"] <= -margin+1e-10 and c["delta_RMSE_N"] <= 1e-10 and
            c["fraccion_grupos_mejora"]>0.5 and c["delta_al_omitir_grupo_max_N"] < -1e-10)

def generar(root: Path = ROOT) -> dict[str, Any]:
    """Escribe tablas de interpretación, conclusiones y la recomendación trazable."""
    root=Path(root);base=root/"reports/benchmark_regresion";td=base/"tables";out=base/"interpretacion"
    out.mkdir(parents=True,exist_ok=True)
    rules=json.loads((root/"config/lectura_benchmark_regresion.json").read_text(encoding="utf-8"))
    cfg=json.loads((root/"config/benchmark_regresion.json").read_text(encoding="utf-8"))
    oof=read(td/"predicciones_oof.csv");metrics=read(td/"metricas_resumen.csv");folds=read(td/"metricas_folds.csv")
    models=rules["modelos_iniciales"];tol=rules["banda_descriptiva_N"]
    nom=oof[oof.protocolo.eq("configuracion")];m=metrics[metrics.protocolo.eq("configuracion")].set_index("procedimiento")
    mf=metrics[metrics.protocolo.eq("familia")].set_index("procedimiento")
    best=float(m.loc[models,"MAE_N"].min())
    chosen=next(x for x in rules["prioridad_parsimonia_en_banda"] if m.loc[x,"MAE_N"]<=best+tol+1e-10)
    check=contraste(nom,"Selector_tuning",chosen)
    principal="Selector_tuning" if adopta(check,tol) else chosen
    ref=contraste(nom,principal,"Media_por_nivel")
    op=principal if adopta(ref,tol) else "Media_por_nivel"
    cmp=[]
    for proto,rows in oof.groupby("protocolo"):
        for model in [*models,"Tuning_rango_1","Tuning_rango_2","Selector_tuning"]:
            for control in ["Ridge_historico","Dummy_mediana","Media_por_nivel"]:
                cmp.append({"protocolo":proto,"procedimiento":model,"referencia":control,**contraste(rows,model,control)})
    pd.DataFrame(cmp).to_csv(out/"comparaciones_pareadas.csv",index=False)
    # Cada familia puede aparecer solo cuando fue priorizada dentro de train.
    conditioned=oof[oof.procedimiento.isin(["Tuning_rango_1","Tuning_rango_2"])].copy()
    if conditioned.duplicated(["protocolo","familia_real","Case_n"]).any():
        raise ValueError("Un caso no puede recibir dos ajustes de la misma familia")
    conditioned["err_inicial"]=(conditioned.predicho_inicial_misma_familia_N-conditioned.observado_N).abs()
    conditioned["err_ajustado"]=(conditioned.predicho_N-conditioned.observado_N).abs()
    partial=conditioned.groupby(["protocolo","familia_real"],as_index=False).agg(
        grupos_reservados=("fold","nunique"),casos_reservados=("Case_n","nunique"),
        MAE_inicial_mismos_casos_N=("err_inicial","mean"),MAE_ajustado_mismos_casos_N=("err_ajustado","mean"))
    partial["delta_MAE_N"]=partial.MAE_ajustado_mismos_casos_N-partial.MAE_inicial_mismos_casos_N
    partial.to_csv(out/"tuning_por_familia_condicional.csv",index=False)
    integrated=metrics.copy()
    integrated["modelo_legible"]=integrated.procedimiento.map(name)
    integrated["optimizado"]=integrated.procedimiento.isin(["Tuning_rango_1","Tuning_rango_2","Selector_tuning"])
    integrated["configuracion"]=integrated.procedimiento.map(lambda k:
        json.dumps(cfg["parametros_iniciales"][k],ensure_ascii=False,sort_keys=True)
        if k in cfg["parametros_iniciales"] else "Adaptativa: ver familia_real, config_id y parametros_json en seleccion_tuning.csv por protocolo y fold")
    integrated.to_csv(out/"tabla_final_integrada.csv",index=False)
    categories={"lineal":"Lineal_multiple","distancias":"KNN","arboles":str(m.loc[["XGBoost","CatBoost","Arbol_decision"],"MAE_N"].idxmin()),
                "grupo_A_literatura":str(m.loc[["SVR_RBF","XGBoost","CatBoost"],"MAE_N"].idxmin()),
                "menor_MAE_inicial":str(m.loc[models,"MAE_N"].idxmin()),"menor_DE_entre_iniciales":str(m.loc[models,"DE_MAE_fold_N"].idxmin()),
                "menor_MAE_procedimientos_incluido_tuning":str(m.loc[models+["Tuning_rango_1","Tuning_rango_2","Selector_tuning"],"MAE_N"].idxmin()),
                "menor_MAE_toda_tabla_incluidos_controles":str(m.MAE_N.idxmin()),"compromiso":principal}
    actual=principal;params=cfg["parametros_iniciales"].get(principal)
    if principal=="Selector_tuning":
        full=json.loads((base/"campana_completa.json").read_text(encoding="utf-8"))
        candidate=full["selector"]
        actual=candidate["familia_real"];params=candidate["parametros"]
    backups=[x for x in m.loc[models].sort_values("MAE_N").index if x!=actual][:2]
    msg=(f"Entre las seis recetas fijas, {name(chosen)} pertenece a la banda de {es(tol)} N y tiene prioridad de parsimonia. "+
         ("El selector ajustado cumple todos los criterios de adopción frente a esta receta." if principal!=chosen else
          "El selector ajustado no satisface conjuntamente los cuatro criterios frente a esa receta."))
    result={"procedimiento_principal":principal,"familia_principal":actual,"parametros_principal":params,
            "respaldos":backups,"referencia_operativa":op,"motivo":msg,"MAE_principal_N":float(m.loc[principal,"MAE_N"]),
            "MAE_media_nivel_N":float(m.loc["Media_por_nivel","MAE_N"]),
            "MAE_referencia_familia_excluida_N":float(mf.loc[op,"MAE_N"]),
            "familia_con_menor_MAE_transferencia":str(mf.loc[models,"MAE_N"].idxmin()),
            "MAE_principal_familia_excluida_N":float(mf.loc[principal,"MAE_N"]),
            "categorias":categories,"procedimiento_fijo_preferido":chosen,"selector_vs_fijo":check,"principal_vs_media_nivel":ref}
    (out/"recomendacion.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    # Salidas por caso: diagnóstico de heterogeneidad, no segunda búsqueda.
    qq=oof[oof.procedimiento.isin(models)].copy()
    qq["error_abs_N"]=(qq.predicho_N-qq.observado_N).abs()
    worst=qq.sort_values(["protocolo","procedimiento","error_abs_N","Case_n"],ascending=[True,True,False,True]).groupby(["protocolo","procedimiento"],as_index=False).head(3)
    worst.to_csv(out/"casos_mayor_error.csv",index=False)
    _document(root,result,metrics,folds,oof,categories,worst,models,cfg)
    return result

def _document(root:Path,r:dict, metrics:pd.DataFrame, folds:pd.DataFrame, oof:pd.DataFrame,
              cat:dict, worst:pd.DataFrame, models:list[str], cfg:dict) -> None:
    """Formar el informe académico con cifras del run actual."""
    m=metrics[metrics.protocolo.eq("configuracion")].set_index("procedimiento")
    f=metrics[metrics.protocolo.eq("familia")].set_index("procedimiento")
    prim=r["procedimiento_principal"];oper=r["referencia_operativa"];ref=r["principal_vs_media_nivel"]
    family=r["familia_principal"];tbest=cat["menor_MAE_inicial"];tree=cat["arboles"];stable=cat["menor_DE_entre_iniciales"]
    td=root/"reports/benchmark_regresion/tables"
    diag=read(td/"diagnosticos_modelos.csv");ols=diag[(diag.procedimiento=="Lineal_multiple")&(diag.protocolo=="configuracion")]
    coef=read(td/"coeficientes_ols.csv");cf=coef[(coef.protocolo=="configuracion")&(coef.variable=="Fiber_height")&(coef.estado=="retenida")]
    svr=diag[(diag.protocolo=="configuracion")&(diag.procedimiento=="SVR_RBF")]
    kn=read(td/"vecinos_knn.csv");kn=kn[(kn.protocolo=="configuracion")&(kn.procedimiento=="KNN")]
    levels=oof[oof.protocolo.eq("configuracion")&oof.procedimiento.eq("KNN")].set_index("Case_n").n_cells_Y
    same=float(kn.Case_n.map(levels).eq(kn.caso_vecino_train.map(levels)).mean())
    selections=read(td/"seleccion_tuning.csv")
    partial=read(td.parent/"interpretacion/tuning_por_familia_condicional.csv")
    full=json.loads((td.parent/"campana_completa.json").read_text(encoding="utf-8"))
    historic=read(root/"docs/benchmark/antecedentes_metricas.csv")
    h=historic[(historic.etapa=="05 · ajuste anidado")&(historic.modelo=="Ridge")&(historic.protocolo=="configuracion")].iloc[0]
    text=["# Resultados comparativos de los seis regresores",
          f"**Para configuraciones de los tres niveles conocidos**, la recomendación entre regresores es **{name(family)}**: MAE {es(m.loc[prim,'MAE_N'])} N, RMSE {es(m.loc[prim,'RMSE_N'])} N y DE del MAE entre grupos {es(m.loc[prim,'DE_MAE_fold_N'])} N. La referencia de media por nivel obtiene {es(m.loc['Media_por_nivel','MAE_N'])} N. La regla previa deja **{name(oper)}** como opción operativa para ese dominio.",
          r["motivo"]+f" El procedimiento principal cambia el MAE frente a la media por nivel en {es(ref['delta_MAE_N'],3)} N y gana en {ref['grupos_mejora']}/{ref['grupos']} configuraciones. Al suprimir un grupo por turno del agregado, el intervalo descriptivo de esa diferencia es [{es(ref['delta_al_omitir_grupo_min_N'],3)}, {es(ref['delta_al_omitir_grupo_max_N'],3)}] N.",
          "## Tabla integrada según la unidad reservada",
          "MAE, RMSE y MedAE en N; R² agrupado fuera de muestra. ΔMAE = procedimiento − Ridge fijo en los mismos casos; DE = desviación estándar de los MAE de los grupos externos. Los rangos son estrategias dependientes del entrenamiento. Se informan tiempos de ajuste y predicción; el coste de búsqueda se consulta por separado en el registro del experimento."]
    order=["Ridge_historico","Dummy_mediana","Media_por_nivel",*cfg["familias"],"Tuning_rango_1","Tuning_rango_2","Selector_tuning"]
    for proto,d in [("configuracion",m),("familia",f)]:
        rows=[]
        for code in order:
            v=d.loc[code];opt=code.startswith("Tuning") or code=="Selector_tuning"
            kind=GROUP.get(code,"Selección interna" if opt else "Control")
            fs="Ninguna" if code=="Dummy_mediana" else "Solo Y" if code=="Media_por_nivel" else "A"
            setting="Anidada" if opt else ("Inicial" if code in models else "Referencia")
            rows.append([name(code),kind,setting,fs,"Sí" if opt else "No",es(v.MAE_N),es(v.RMSE_N),es(v.MedAE_N),
                         es(v.R2_OOF,3),es(v.MAE_N-d.loc["Ridge_historico","MAE_N"]),es(v.DE_MAE_fold_N),
                         es(v.tiempo_ajuste_s,4),es(v.tiempo_prediccion_s,4),es(v.tiempo_busqueda_s,3)])
        text.extend(["### "+DOMAIN[proto],table(["Procedimiento","Grupo","Configuración","Entradas","Optim.","MAE","RMSE","MedAE","R²","ΔMAE","DE","Ajuste (s)","Pred. (s)","Búsqueda (s)"],rows)])
    better=[name(k) for k in models if m.loc[k,"MAE_N"]<m.loc["Ridge_historico","MAE_N"]]
    text.append(f"En configuraciones, {sum(m.loc[k,'MAE_N']<m.loc['Dummy_mediana','MAE_N'] for k in models)} de los seis superan la mediana sin geometría y {len(better)} superan Ridge fijo: "+", ".join(better)+f". El Ridge ajustado histórico sobre A obtuvo {es(h.MAE_N,3)} N: el cambio del recomendado frente a ese antecedente es {es(m.loc[prim,'MAE_N']-h.MAE_N,3)} N. Las particiones son comparables, aunque la búsqueda histórica y su presupuesto son diferentes. No se interpreta una diferencia tan pequeña como un avance confirmado.")
    text.extend(["## Qué cambia según la familia de regresión",
        f"El modelo lineal explica parte de la variación nominal (R² {es(m.loc['Lineal_multiple','R2_OOF'],3)}, MAE {es(m.loc['Lineal_multiple','MAE_N'])} N). El mejor MAE inicial es el de {name(tbest)} ({es(m.loc[tbest,'MAE_N'])} N). La diferencia de cada modelo contra las referencias se conserva en los [contrastes emparejados](../reports/benchmark_regresion/interpretacion/comparaciones_pareadas.csv). El rango deficiente impide interpretar cada pendiente de OLS como efecto mecánico independiente.",
        f"OLS conserva {ols.rango_base.min():.0f} de {ols.n_features_tras_varianza.max():.0f} columnas no constantes en los entrenamientos nominales; la condición de la base estandarizada va de {es(ols.condicion_base.min(),0)} a {es(ols.condicion_base.max(),0)}. La pendiente de altura de fibra varía de {es(cf.coeficiente_N_por_unidad.min())} a {es(cf.coeficiente_N_por_unidad.max())} N/mm entre reservas. El cambio de signo ilustra por qué predecir bien aquí no identifica el efecto aislado de aumentar la altura. Véanse todos los [coeficientes y estados](../reports/benchmark_regresion/tables/coeficientes_ols.csv).",
        f"KNN obtiene {es(m.loc['KNN','MAE_N'])} N frente a {es(m.loc['Dummy_mediana','MAE_N'])} N de mediana sin geometría y {es(m.loc['Media_por_nivel','MAE_N'])} N de media por nivel. El {es(100*same,1)} % de las asignaciones de los tres vecinos corresponde al mismo nivel Y del caso reservado. La proximidad es informativa frente al control ingenuo, pero parte de esa información ya está en el nivel. Al pasar a un nivel no visto su MAE es {es(f.loc['KNN','MAE_N'])} N. Un promedio de vecinos no proporciona una ley de extrapolación.",
        f"SVR-RBF alcanza MAE {es(m.loc['SVR_RBF','MAE_N'])} N, con DE entre grupos {es(m.loc['SVR_RBF','DE_MAE_fold_N'])} N frente a {es(m.loc[stable,'DE_MAE_fold_N'])} N de {name(stable)}, el inicial con menor DE. Esa dispersión describe las configuraciones observadas; no es una banda de confianza del predictor.",
        f"Con ε=5 N, la fracción de casos de entrenamiento que son vectores de soporte varía entre {es(100*svr.fraccion_vectores_soporte.min(),0)} % y {es(100*svr.fraccion_vectores_soporte.max(),0)} %. La receta depende de muchas observaciones y su desempeño es heterogéneo; no se considera preferente frente a OLS para esta decisión.",
        f"Entre árboles, {name(tree)} presenta el menor MAE inicial ({es(m.loc[tree,'MAE_N'])} N). XGBoost y CatBoost arrojan {es(m.loc['XGBoost','MAE_N'])} y {es(m.loc['CatBoost','MAE_N'])} N; el árbol sencillo, {es(m.loc['Arbol_decision','MAE_N'])} N. El antecedente bibliográfico justifica ensayarlos, pero no establece que ensambles más costosos aporten valor en esta campaña.",
        "Las siguientes brechas se calculan con igual peso por grupo exterior (MAE en validación − MAE en entrenamiento). Una brecha positiva es compatible con sobreajuste o con cambio de soporte, especialmente en la prueba por nivel.",
        table(["Escenario","Árbol (N)","XGBoost (N)","CatBoost (N)"],[[DOMAIN[p]]+[es((z.MAE_validacion_N-z.MAE_train_N).mean()) for code in ["Arbol_decision","XGBoost","CatBoost"] for z in [folds[(folds.protocolo==p)&(folds.procedimiento==code)]]] for p in DOMAIN]),
        "## Estabilidad y observaciones influyentes",
        "Mayor error de cada familia en la prueba nominal. Un residual grande describe un caso difícil para el predictor; no justifica excluirlo ni corregir su etiqueta. El [archivo por caso](../reports/benchmark_regresion/interpretacion/casos_mayor_error.csv) conserva los tres mayores de cada familia y escenario.",
        table(["Familia","Caso","Observado (N)","Predicho (N)","Error absoluto (N)"],[[name(x.procedimiento),str(x.Case_n),es(x.observado_N),es(x.predicho_N),es(x.error_abs_N)] for x in worst[worst.protocolo.eq("configuracion")].groupby("procedimiento").head(1).itertuples()]),
        "## Qué aportó el tuning",
        "Cambio externo frente a la configuración inicial de la misma familia que recibió cada observación. Valores negativos indican reducción del MAE. El primer o segundo rango puede representar algoritmos diferentes en pliegues distintos.",
        table(["Escenario / ocasiones que recibió tuning"]+[name(x) for x in cfg["familias"] if x!="Lineal_multiple"],[[DOMAIN[p]]+[str(int(((selections.protocolo==p)&(selections.familia_real==x)).sum())) for x in cfg["familias"] if x!="Lineal_multiple"] for p in DOMAIN]),
        f"Sobre toda la campaña nominal, el cribado posterior selecciona "+", ".join(name(w["familia_real"]) for w in full["candidatas"])+f"; el selector elige {name(full['selector']['familia_real'])} con la receta {full['selector']['config_id']}. Es el reajuste entre las familias elegibles para tuning; OLS no participa en esa búsqueda. Al evaluar el procedimiento en configuraciones reservadas, el tuning del selector cambia su MAE en {es(m.loc['Selector_tuning','delta_MAE_vs_inicial_misma_familia_N'],3)} N frente a sus mismas familias iniciales.",
        table(["Escenario","Primer rango (N)","Segundo rango (N)","Selector (N)"],[[DOMAIN[p]]+[es(float(d.loc[c,"delta_MAE_vs_inicial_misma_familia_N"]),3) for c in ["Tuning_rango_1","Tuning_rango_2","Selector_tuning"]] for p,d in [("configuracion",m),("familia",f)]]),
        "La tabla siguiente conserva los **nombres de las familias ajustadas** y enfrenta su receta inicial y ajustada en exactamente los casos donde el cribado interno las seleccionó. Las coberturas difieren: estas filas no sirven para clasificar algoritmos entre sí. La comparación completa, con todas las reservas, es la tabla de procedimientos anterior.",
        table(["Escenario","Familia ajustada","Grupos","Casos","MAE inicial (N)","MAE ajustado (N)","ΔMAE (N)"],
              [[DOMAIN[v.protocolo],name(v.familia_real),str(v.grupos_reservados),str(v.casos_reservados),
                es(v.MAE_inicial_mismos_casos_N),es(v.MAE_ajustado_mismos_casos_N),es(v.delta_MAE_N,3)] for v in partial.itertuples()]),
        "## Decisión y límites",
        table(["Categoría (configuraciones)","Procedimiento","MAE (N)"],[[CATEGORY_NAMES[k],name(c),es(m.loc[c,"MAE_N"],3)] for k,c in cat.items()]),
        f"Receta recomendada entre regresores: **{name(family)}**, parámetros: `{json.dumps(r['parametros_principal'],ensure_ascii=False)}`. Entradas A, preprocesamiento y semillas del [protocolo congelado](../config/benchmark_regresion.json). Respaldos descriptivos: "+", ".join(f"{name(x)} (MAE {es(m.loc[x,'MAE_N'])} N; DE {es(m.loc[x,'DE_MAE_fold_N'])} N)" for x in r["respaldos"])+".",
        f"El procedimiento principal alcanza MAE {es(f.loc[prim,'MAE_N'])} N y R² {es(f.loc[prim,'R2_OOF'],3)} al retirar un nivel completo. Entre los seis fijos, {name(r['familia_con_menor_MAE_transferencia'])} registra el menor MAE de transferencia ({es(f.loc[r['familia_con_menor_MAE_transferencia'],'MAE_N'])} N). Solo hay tres reservas de este tipo. En una arquitectura ausente la referencia por nivel recurre a la media del entrenamiento; el criterio operativo para niveles observados no constituye una recomendación de diseño para nuevos niveles.",
        "El EDA ya mostraba concentración de la variación de primera falla entre tres niveles y descriptores geométricos relacionados. Las predicciones son compatibles con agrupación por arquitectura, sin identificar el efecto causal del número de celdas. Los rangos externos se describen en una sola campaña ya explorada; no suministran una evaluación independiente de la familia que se acaba de recomendar. El paso siguiente es fijar el uso y un error aceptable con los investigadores, verificar identidades, geometrías y lotes, y contrastar la receta sin nuevos ajustes en probetas o campañas independientes.",
        "[Notebook ejecutado](../notebooks/Comparativa_6_Modelos_Regresion_Pantograficas.ipynb) · [Tabla integrada con parámetros y costes](../reports/benchmark_regresion/interpretacion/tabla_final_integrada.csv) · [Auditoría independiente de datos y selección](../reports/benchmark_regresion/auditoria/revision_independiente.json) · [Fundamento bibliográfico](justificacion_seleccion_modelos.md) · [Antecedentes con su protocolo](benchmark/antecedentes_metricas.csv)"])
    (root/"docs/resultados_comparativos_modelos.md").write_text("\n\n".join(text)+"\n",encoding="utf-8")

if __name__=="__main__":
    print(json.dumps(generar(),ensure_ascii=False,indent=2))
