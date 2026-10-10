# Introducción científica y descripción del experimento

**Universidad Nacional de Ingeniería — Facultad de Ingeniería Industrial y de Sistemas**<br>
**Maestría en Inteligencia Artificial · Trabajo de Investigación II · MIA, 4.º ciclo, sección B · 2026-2**<br>
**Investigadora:** Melissa Dessire Aylas Barranca<br>
**Docente:** Glen Dario Rodríguez Rafael

## Qué se estudia

Un **metamaterial mecánico** es una estructura cuyo comportamiento se diseña mediante su arquitectura interna, además de depender del material que la constituye. Dos piezas fabricadas con el mismo polímero pueden responder de forma distinta si cambian la forma, las dimensiones o la disposición de sus elementos.

Una **estructura pantográfica** es un retículo formado por dos familias de fibras alargadas conectadas por pequeños pivotes. En la descripción de Venditti, las familias son ortogonales, ocupan planos paralelos y se unen mediante conectores cilíndricos elásticos. Las fibras transmiten las acciones mecánicas; los pivotes conectan ambas familias y permiten una interacción deformable entre ellas. No son articulaciones ideales sin resistencia. La literatura distingue extensión y flexión de fibras, y torsión local de pivotes, como mecanismos representados en modelos discretos; identificar sus contribuciones en esta campaña requeriría mediciones o un modelo mecánico adicional. [V, PDF 91; T17, PDF 3–4 / pp. 287–288; T19, PDF 2–4 / pp. 210–212.]

La finalidad del EDA es establecer qué relaciones existen entre **geometría y respuesta experimental**, cuáles se explican por el diseño y cuáles requieren verificación adicional. Una correlación alta no demuestra que un parámetro aislado cause el aumento de resistencia ni demuestra precisión predictiva.

## Procedencia y condiciones verificadas

La campaña se documenta en la tesis doctoral de **Enrico Venditti**, presentada en el **Departamento de Arquitectura, Diseño y Urbanismo de la Università degli Studi di Sassari, Italia**, dentro del doctorado en Arquitectura y Ambiente, bajo supervisión de Emilio Barchiesi. La portada indica año académico **2024/2025**. Esta identificación describe la fuente; no presupone un cargo o afiliación actual. El plan de tesis de Melissa Aylas Barranca, fechado en 2026, señala que Emilio Turco proporcionó la campaña para fines académicos. [V, PDF 1; P, PDF 1, 24 y 26 / pp. impresas 18 y 20 para las dos últimas.]

Los especímenes se diseñaron en SolidWorks y se fabricaron mediante **sinterización láser selectiva de polvo de poliamida (nylon)**. Se buscó mantener una masa idéntica o similar mientras se variaban la altura de las fibras, la altura de los pivotes y el número de celdas en Y. Las dimensiones externas nominales fueron **210 × 70 mm**. Por tanto, las familias no representan automáticamente especímenes de mayor longitud exterior: representan diferentes arquitecturas dentro de dimensiones exteriores comunes. La condición de masa similar es un criterio de diseño documentado; el archivo no contiene masas medidas para verificarla caso por caso. [V, PDF 94–95.]

La utilidad de estudiar esta arquitectura en una investigación peruana reside en desarrollar procedimientos de análisis y predicción para diseño y fabricación de estructuras; la nacionalidad del investigador no altera las leyes mecánicas. La transferencia a fabricación local, otro polímero, otra impresora o una aplicación estructural exige nuevas verificaciones experimentales. Esta campaña no demuestra por sí misma aptitud para construcción, protección sísmica ni disponibilidad industrial local.

## Qué ocurre durante el ensayo

El protocolo documenta **tracción con ciclos de carga y descarga**, una velocidad de desplazamiento impuesta de **15 mm/min** y niveles nominales de desplazamiento espaciados cada **10 mm**. El espécimen se fija al equipo y se registra la respuesta fuerza–desplazamiento. Los valores de velocidad en mm/min describen desplazamiento por tiempo, no una tasa de deformación unitaria. Los detalles instrumentales de otros artículos no se atribuyen a esta campaña. [V, PDF 96.]

La tracción tiende a alargar la estructura. Durante la fase de carga aumenta el desplazamiento impuesto; la fuerza puede caer si se produce una rotura. Durante la descarga se reduce la solicitación y se registra el desplazamiento que permanece: descargar no implica necesariamente comprimir. La flexión de fibras y los giros de las conexiones son mecanismos de respuesta y no acreditan otros ensayos externos.

La descripción consultada identifica el equipo de ensayo de tracción y el registro de fuerza–desplazamiento, pero no especifica fabricante, modelo, capacidad ni sensores concretos. También documenta grabación del sonido de las roturas para identificar sus instantes. Esto no acredita un sistema de emisión acústica ni un modelo de micrófono específico. [V, PDF 96–97.]

La **fuerza** expresa la acción mecánica medida, en newtons (N). El **desplazamiento** expresa el cambio de posición impuesto o registrado, en milímetros (mm). La **deformación** describe el cambio de forma o dimensiones; una deformación unitaria requeriría definir una longitud de referencia apropiada. Las columnas de desplazamiento disponibles no deben renombrarse como deformación unitaria.

Durante el ensayo pueden producirse roturas locales de fibras o pivotes. La estructura puede conservar capacidad de carga después del primer daño porque aún existen elementos conectados capaces de transmitir acciones. La fuente documenta esa capacidad y secuencias de rotura; el EDA no identifica por sí solo qué elemento se rompe en cada caso ni cuantifica la redistribución de esfuerzos. [V, PDF 96–97.]

## Definición del objetivo predictivo

La **carga de primera falla**, F_FF, es la fuerza experimental asociada al **primer evento de rotura identificado de una fibra o un pivote** conforme a la campaña. Se representa mediante la columna **First_Failure_Load_N**, expresada en N. El análisis conserva el resumen numérico de la fuente. La revisión documental detectó pequeñas diferencias entre ese resumen y el primer pico tabulado en algunas fichas; se registran en trazabilidad y no se reemplazan automáticamente. [V, PDF 96, 115, 119, 149 y 164.]

No corresponde necesariamente a la fuerza máxima del ensayo. **Ultimate_Load** se interpreta como la carga del evento terminal reportado, coherente con la fuerza de rotura total/final descrita en el protocolo; tampoco se calcula como el máximo de toda la curva. Puede haber picos mayores entre el primer y el último evento. **Maximum_Displacement**, los desplazamientos residuales y las energías describen otras respuestas.

El problema futuro de regresión es:

**F_FF = f(X_geometría, X_estructura) + ε**

Las X son dimensiones, volúmenes y descriptores de arquitectura conocidos antes del ensayo. La función f representa una relación estimada a partir de ejemplos experimentales. El término ε reúne variación no explicada, incertidumbre experimental y limitaciones del modelo. El objetivo es estimar una fuerza numérica antes del ensayo, por lo que ninguna carga, desplazamiento residual, energía o número de eventos observado después puede formar parte de las entradas.

## Esquema conceptual original

El siguiente esquema representa el flujo de investigación; no reproduce un espécimen ni simula su deformación.

    GEOMETRÍA Y ARQUITECTURA
    Celdas, fibras y pivotes; dimensiones y volúmenes previos
                         ↓
    ENSAYO MECÁNICO
    Tracción cíclica documentada; registro de fuerza y desplazamiento
                         ↓
    RESPUESTA OBSERVADA
    Primera falla, evento terminal, residuales y energías
                         ↓
    AUDITORÍA Y ANÁLISIS ESTADÍSTICO
    Trazabilidad, calidad, asociaciones, redundancia e incertidumbre
                         ↓
    FUTURO MODELADO PREDICTIVO
    Solo entradas previas; validación acorde con el dominio geométrico

## Qué puede aportar el análisis y qué queda fuera

El EDA compara distribuciones, diferencia asociaciones entre familias y dentro de ellas, identifica geometrías redundantes, revisa casos influyentes y estudia respuestas repetidas por ciclo o evento. Estas observaciones ayudan a elegir representaciones de entrada, definir hipótesis y orientar mediciones futuras.

Las energías están expresamente rotuladas en **mJ** en el apéndice. La energía por fractura se denomina **energía tabulada asociada al evento**: no se dispone del algoritmo ni de los límites de integración necesarios para identificarla inequívocamente como energía liberada en una caída, trabajo acumulado o propiedad material. Los valores negativos iniciales de disipación y los extremos se preservan y se auditan. [V, PDF 117, 121 y 131.]

Los ciclos de un mismo espécimen son mediciones relacionadas. El cociente residual/amplitud nominal y su complemento describen pérdida y recuperación nominales respecto a esa amplitud; no son deformación unitaria, rigidez o una ley de fatiga. El documento contiene imágenes de curvas fuerza–desplazamiento, pero el archivo MATLAB disponible contiene resúmenes, no las series completas. No se reconstruyen curvas ni se integran áreas de histéresis a partir de esos resúmenes.

La conciliación documental forma parte del resultado: hay diferencias entre el archivo digital, la tabla general y algunas fichas. Las fuentes originales se conservan intactas, y la interpretación distingue evidencia calculada sobre el archivo digital de hipótesis pendientes de revisión experimental. Véanse [trazabilidad](trazabilidad_datos.md), [diccionario](diccionario_datos.md) e [informe de curaduría](informe_curaduria.md).

## Referencias verificadas

- **V:** Venditti, E. *Analisi strutturale per lo studio del danneggiamento in materiali e strutture complesse*. Tesis doctoral, Università degli Studi di Sassari, año académico 2024/2025. Copia privada suministrada; capítulo 9 y apéndice A. Las páginas son posiciones del PDF desde la portada, sin numeración impresa visible.
- **T17:** Turco, E., Golaszewski, M., Giorgio, I. y Placidi, L. (2017). *Can a Hencky-Type Model Predict the Mechanical Behaviour of Pantographic Lattices?* En *Mathematical Modelling in Solid Mechanics*, pp. 285–311. DOI: [10.1007/978-981-10-3764-1_18](https://doi.org/10.1007/978-981-10-3764-1_18).
- **T19:** Turco, E., Misra, A., Sarikaya, R. y Lekszycki, T. (2019; publicación en línea 2018). *Quantitative analysis of deformation mechanisms in pantographic substructures: experiments and modeling*. *Continuum Mechanics and Thermodynamics*, 31, 209–223. DOI: [10.1007/s00161-018-0678-y](https://doi.org/10.1007/s00161-018-0678-y).
- **P:** Aylas Barranca, M. D. (2026). *Análisis predictivo e interpretable de la respuesta mecánica de estructuras pantográficas mediante aprendizaje automático supervisado aplicado a datos experimentales*. Plan de tesis, Universidad Nacional de Ingeniería. Copia privada suministrada.
