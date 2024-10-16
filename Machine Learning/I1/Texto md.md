## **Regularización L<sup>2</sup> en Machine Learning**
###### Por Carolina Vera Vásquez, 16 de octubre, 2024

En *Machine Learning*, un desafío principal es la construcción de un modelo que se adapte correctamente a los datos y pueda predecir correctamente datos nuevos, de modo que el modelo no sea muy rígido (subajuste o *underfitting*) ni muy flexible (sobreajuste u *overfitting*).

Para lo mencionado se utilizan las regularizaciones, estas ayudan a la capacidad del modelo para generalizarse acorde a los datos evitándose el sobreajuste.

# **¿Qué son las regularizaciones?**
Entrenar un modelo busca la estabilidad entre el error de entrenamiento y el de testeo, para esto entran las regularizaciones [\[1\]](#ref1), donde se tiene en cuenta:

- Sesgo (*bias*): diferencia entre la predicción esperada y el valor real.
- Varianza (*variance*): variabilidad esperada de la predicción del modelo.
- Error irreducible: error proveniente de la problemática, tales como factores no considerados o errores de mediciones.

La regularización es un conjunto de métodos que, como se dijo anteriormente, ayudan a evitar el sobreajuste del modelo, modificando el algoritmo de aprendizaje para reducir el error de generalización, pero no el error de entrenamiento. El objetivo de estos métodos es la reducción de la complejidad del modelo de manera controlada, permitiendo la generalización con un mejor resultado frente a nuevos datos o datos de testeo, manteniendo la precisión de los datos de entrenamiento al reducir la varianza. Las regularizaciones normalmente agregan un término de regularización, valga la redundancia, a la función de costo que se busca minimizar en el modelo [\[2\]](#ref2).

Existen distintos tipos de regularizaciones, las dos más comunes son:

- Regularización L<sup>1</sup> o regresión Lasso.
- Regularización L<sup>2</sup>, regresión Ridge o regularización Tikhonov.

# **El problema del sobreajuste**
Se le denomina sobreajustado a un modelo que se adapta tan bien a los datos de entrenamiento que no generaliza correctamente, de modo que no se obtiene un patrón que permita predecir o inferir nuevos datos. Este fenómeno puede pasar al sobre entrenar el modelo o se entrena con datos anómalos, de modo que aprende de forma específica y no genérica [\[3\]](#ref3), un ejemplo de esto se ve en la [Imagen 1](#Imagen1).

<div style="text-align: center;" id="Imagen1">
    <img src="https://raw.githubusercontent.com/CarolinaVeraV/Publicos/refs/heads/main/Machine%20Learning/I1/Imagen1.png" alt="Underfitting, Appropiate, Overfitting" width="300"/>
    <p>Imagen 1: ejemplo de un modelo con subajuste, ajuste apropiado y sobreajuste <a href="#ref4">[4]</a>.</p>
</div>

Las formas de evitar el sobreajuste son:

- Aumentando la cantidad de datos de entrenamiento.
- Cambiando los parámetros del algoritmo, al simplificarlo le es más difícil al modelo ajustarse tanto a los datos.
- Aplicando un método de regularización.

# **Cómo funciona la regularización L<sup>2</sup>**
La regularización L<sup>2</sup>, también conocida como regresión Ridge, regularización Tikhonov o decaimiento de peso, es la más utilizada y común dentro del *Deep Learning*. Este método lleva los pesos más cerca del origen por medio de la adición del término de regularización $\Omega\left(\theta\right)=\left|\left|w\right|\right|^2=w^Tw=\sum_i w_i^2$ a la función objetivo del modelo [\[4\]](#ref4).

La principal diferencia con la regularización L<sup>1</sup>, es que esta última resulta en una solución más dispersa, es decir, da valores óptimos iguales a cero para algunos parámetros, dando un comportamiento cualitativamente diferente a la regresión Ridge.

Para realizar una regresión lineal con el decaimiento de peso, se minimiza la función de costo total ($J(w)$) que considera el error cuadrático medio en el entrenamiento (${MSE}_{train}$) y un criterio que expresa la preferencia para los pesos que obtengan una menor norma L<sup>2</sup> cuadrática ($\lambda\in\mathbb R^+_0$) en función de los pesos o parámetros ($w$) del modelo, denominada penalización. De este modo y considerando que se suele excluir el sesgo de las regularizaciones:

$$J(w)={MSE}_{train}+\lambda\Omega(\theta)$$

Expandiendo para la regresión Ridge:

$$ J\left(w\right)=\frac{1}{m_{train}}\sum_{i=1}^{m_{train}}\left({\hat{y}}_i^{train}-y_i^{train}\right)^2+\lambda\sum_{j=1}^{n}w_j^2 $$

Cuando el valor $\lambda$ es 0, se trata de la no preferencia, por lo que no hay regularización y la función será simplemente el error cuadrático medio. Por el otro lado, para valores grandes de $\lambda$, es decir, una mayor penalización, el impacto de la regularización es mayor, provocando que los pesos se vuelven más pequeños y estables, encontrando una solución más equilibrada que para λ menores.

Lo mencionado se puede observar en la [Imagen 1](#Imagen1):

- En el primer cuadro se ve un subajuste, debido a un $\lambda$ muy grande, provocando una estabilización excesiva en los datos, obteniéndose así un modelo muy simplificado.
- En el tercer cuadro se observa el caso opuesto, debido a que el valor de $\lambda$ es 0 (o próximo a este), prácticamente no existe la regularización de los parámetros, quedando con un sobreajuste.

# **Beneficios de la regularización L<sup>2</sup>**
Los beneficios destacables de este método son [\[5\]](#ref5):

- Agrega una penalización a la función de costo en base a los pesos del modelo, mitigando aquellos con un gran efecto en el modelo, impidiendo que se ajuste a detalles específicos de los datos de entrenamiento.
- Reduce los pesos de los parámetros menos importantes, usualmente aproximándolos a cero sin llegar a este, así no se elimina por completo ninguna característica del modelo y no se deja de considerar cada una.
- Uniformiza y mejora la estabilidad de la importancia de los datos, evitando eficazmente el sobreajuste al penalizar los pesos grandes y distribuyendo la influencia de estos de forma más equitativa.
- Reduce la complejidad del modelo al utilizar parámetros más pequeños y controlados, sin el descarte de aquellas características que no contribuyen significativamente.
- Mejora la capacidad del modelo para la generalización en datos de testeo.
# **Comparación entre las regularizaciones L<sup>1</sup> y L<sup>2</sup>**
Si bien ambas regularizaciones ayudan a evitar el sobreajuste en los modelos, tienen una ligera diferencia en la fórmula que utilizan, de modo que generan cambios distintos al modelo y cómo manejan los datos.

La regularización L<sup>1</sup> se diferencia en el término de regularización $\Omega\left(\theta\right)=\left|\left|w\right|\right|_1=\sum_i \left|w_i\right|$, entonces se calcula del siguiente modo [\[4\]](#ref4):

$$ J\left(w\right)=\frac{1}{m_{train}}\sum_{i=1}^{m_{train}}\left({\hat{y}}_i^{train}-y_i^{train}\right)^2+\lambda\sum_{j=1}^{n}\left|w_j\right| $$

Se puede observar como el término de regularización usa los valores absolutos para la regresión Lasso, mientras que en la regresión Ridge se usan los cuadrados de los pesos.

Como se mencionó anteriormente, la principal diferencia entre estos dos métodos es que la regularización L<sup>1</sup> resulta en una solución más dispersa, de modo que este puede igualar algunos parámetros a cero.

Esta regulación ayuda a seleccionar aquellas características importantes y excluye (igualando a 0) las que no lo sean, mientras que la L<sup>2</sup> conserva un efecto de todos los parámetros presentes en el modelo [\[6\]](#ref6).

Ambas regresiones crearán vectores de pesos distintos, estos resultados a modo de ejemplo serían:

- L<sup>1</sup>: $\left(0;\ 1;\ 1;\ 0;\ 1\right)$, se observa que en el vector se anulan por completo algunos parámetros.
- L<sup>2</sup>: $\left(0,5;\ 0,3;\ -0,2;\ 0,4;\ 0,1\right)$, en este caso, ningún parámetro se anula, pero se obtienen variados coeficientes para los pesos.

En la [Imagen 2](#Imagen2) e [Imagen 3](#Imagen3), se puede observar gráficamente el efecto de ambas regularizaciones en un mismo modelo. Las elipses sólidas representan los contornos para mismos valores de la función objetivo sin regularizar. Las líneas punteadas representan contornos de igual valor para los regularizadores L<sup>1</sup> y L<sup>2</sup>, respectivamente. Los puntos $\widetilde{w}$ indican donde se alcanza el equilibrio entre la función de costo y los regularizadores. Se ve como la función objetivo no varía mucho en $w_2$ al modificar harto el $w_1$ (al alejarse de $w^\ast$), indicando que este parámetro vertical no influye tanto como lo hace el horizontal, por lo que la regularización afecta de mayor manera a $w_1$, llevándolo a 0 en el caso de L<sup>1</sup> y a un valor cercano a cero para L<sup>2</sup> [\[4\]](#ref4).

<div style="text-align: center;" id="Imagen2">
    <img src="https://raw.githubusercontent.com/CarolinaVeraV/Publicos/refs/heads/main/Machine%20Learning/I1/Imagen2.png" alt="Efecto regularización L1" width="250"/>
    <p>Imagen 2: ilustración del efecto de la regularización L<sup>1</sup> en el valor óptimo de <i>w</i>, modificación de la imagen original por la autora del blog <a href="#ref4">[4]</a>.</p>
</div>

<div style="text-align: center;" id="Imagen3">
    <img src="https://raw.githubusercontent.com/CarolinaVeraV/Publicos/refs/heads/main/Machine%20Learning/I1/Imagen3.png" alt="Efecto regularización L2" width="250"/>
    <p>Imagen 3: ilustración del efecto de la regularización L<sup>2</sup> en el valor óptimo de <i>w</i> <a href="#ref4">[4]</a>.</p>
</div>

En resumen, la regresión Lasso realiza una selección de características, quedándose únicamente con aquellas más relevantes, reduciendo la cantidad de parámetros del modelo, siendo útil para aquellos problemas con parámetros irrelevantes. Por el otro lado, la regresión Ridge distribuye los pesos entre todas las características, estabilizando las influencias, ideal para problemas en los que todos los parámetros son necesarios y que se requiera limitar el impacto de aquellos con mayor peso.

Adicionalmente, se pueden combinar ambas regulaciones en una misma, llamada *Elastic net*, que aprovecha los beneficios de ambas penalizaciones, utilizando un parámetro de control $\alpha\in\left[0,\ 1\right]$ para controlar el uso de las regresiones, cuando este parámetro es 0 se utiliza únicamente la regulación L<sup>2</sup> y al ser 1 queda únicamente la regulación L<sup>1</sup>. Esta combinación usualmente provee buenos resultados, puesto que conserva el uso de las características más relevantes, mientras regula el peso de estas. La fórmula que las une es [\[7\]](#ref7):

$$ J\left(w\right)=MSE_{train}+\lambda\left(\alpha\Omega_1\left(\theta\right)+\left(1-\alpha\right)\Omega_2\left(\theta\right)\right) $$

$$ J\left(w\right)=\frac{1}{m_{train}}\sum_{i=1}^{m_{train}}\left({\hat{y}}_i^{train}-y_i^{train}\right)^2+\lambda\left(\alpha\sum_{j=1}^{n}\left|w_j\right|+\left(1-\alpha\right)\sum_{j=1}^{n}w_j^2\right) $$

# **Desafíos de la regresión Ridge**
En la regularización L<sup>2</sup> se pueden encontrar algunos desafíos y problemáticas, las más destacables son [\[8\]](#ref8):

- Parámetro $\lambda$: este parámetro necesita ser determinado, pero si es muy bajo el modelo se sobreajustará y si es muy alto no se ajustará bien a los datos, por lo que se debe encontrar el valor adecuado, incluyendo varias iteraciones.
- Selección de regularización: si bien este método funciona bien, la elección de si usar esta u otra se debe basar en los datos utilizados y la problemática que se busca resolver, donde los efectos de la regularización no afecten negativamente los resultados esperados. Además de considerar la posibilidad de necesitar utilizar otras métricas, como precisión, matriz de confusión, entre otras.
- Características irrelevantes: a pesar de que se reduce los pesos de las características, pueden considerarse algunas que realmente sean irrelevantes o redundantes para la predicción, siendo estas una problemática en caso de modelos muy extensos y complicados.

# **Conclusiones**
El uso de regularizaciones es fundamental en *Machine Learning*, puesto que ayuda a que el modelo se ajuste adecuadamente a los datos, disminuyendo el error de generalización. Dentro de los métodos de regularización, el más utilizado es la regularización L<sup>2</sup> o regresión Ridge.

La regresión Ridge mejora la generalización del modelo por medio de regular los pesos de los datos sin anularlos, haciendo el modelo más robusto frente a nuevos datos. Sin embargo, tiene sus desafíos, los cuales incluyen escoger el parámetro $\lambda$ adecuado para evitar el subajuste y sobreajuste, seleccionar el método de regularización correcto para el objetivo de la problemática y que se incluyan datos que sean completamente irrelevantes para el modelo.

En resumen, aunque la regularización L<sup>2</sup> no sea una solución universal, es capaz de reducir el sobreajuste y mejorar la estabilidad del modelo, siendo una opción confiable para considerar en *Machine Learning*.

# **Referencias**
1. <div id="ref1"></div>Ramírez Sánchez, J. (2022). <i>Regularización de redes neuronales artificiales para la clasificación de imágenes de retinopatía diabética.</i> Universidad Nacional de Colombia. <a href="https://repositorio.unal.edu.co/handle/unal/81945" target="_blank">https://repositorio.unal.edu.co/handle/unal/81945</a>

2. <div id="ref2"></div><i>Regularización</i>. (s. f.). Interactive Chaos. Recuperado 12 de octubre de 2024, de <a href="https://interactivechaos.com/es/manual/tutorial-de-machine-learning/regularizacion#:~:text=La%20regularizaci%C3%B3n%20es%20un%20conjunto,en%20los%20datos%20de%20entrenamiento." target="_blank">https://interactivechaos.com/es/manual/tutorial-de-machine-learning/regularizacion#:~:text=La%20regularizaci%C3%B3n%20es%20un%20conjunto,en%20los%20datos%20de%20entrenamiento.</a>

3. <div id="ref3"></div>Alvaro. (2020, 25 mayo). <i>¿Qué es el sobreajuste u overfitting y por qué debemos evitarlo?</i> MachineLearningParaTodos.com. Recuperado 12 de octubre de 2024, de <a href="https://machinelearningparatodos.com/que-es-el-sobreajuste-u-overfitting-y-por-que-debemos-evitarlo/#:~:text=Cu%C3%A1ndo%20se%20produce%20el%20sobreajuste,los%20patrones%20generales%2C%20el%20concepto" target="_blank">https://machinelearningparatodos.com/que-es-el-sobreajuste-u-overfitting-y-por-que-debemos-evitarlo/#:~:text=Cu%C3%A1ndo%20se%20produce%20el%20sobreajuste,los%20patrones%20generales%2C%20el%20concepto</a>

4. <div id="ref4"></div>Goodfellow, I., Bengio, Y., & Courville, A. (2016). <i>Deep learning</i>. MIT Press. <a href="http://www.deeplearningbook.org" target="_blank">http://www.deeplearningbook.org</a>

5. <div id="ref5"></div>Code Labs Academy. (2024, 5 junio). <i>Comprender la regularización L1 y L2: Estrategias clave para evitar el sobreajuste en los modelos de aprendizaje automático</i>. Recuperado 15 de octubre de 2024, de <a href="https://codelabsacademy.com/es/blog/the-role-of-l1-and-l2-regularization-in-preventing-overfitting-and-enhancing-model-generalization" target="_blank">https://codelabsacademy.com/es/blog/the-role-of-l1-and-l2-regularization-in-preventing-overfitting-and-enhancing-model-generalization</a>

6. <div id="ref6"></div>Miguel. (2021, 5 junio). <i>Regularización L1 vs L2 y ¿Cuándo usar cuál?</i> ManualesTutor. Recuperado 15 de octubre de 2024, de <a href="https://manualestutor.com/aprendizaje-automatico/regularizacion-l1-vs-l2-y-cuando-usar-cual/" target="_blank">https://manualestutor.com/aprendizaje-automatico/regularizacion-l1-vs-l2-y-cuando-usar-cual/</a>

7. <div id="ref7"></div>Dhumne, S. (2023, 12 marzo). <i>Elastic Net Regression detailed guide !</i> Medium. <a href="https://medium.com/@shruti.dhumne/elastic-net-regression-detailed-guide-99dce30b8e6e" target="_blank">https://medium.com/@shruti.dhumne/elastic-net-regression-detailed-guide-99dce30b8e6e</a>

8. <div id="ref8"></div><i>¿Cómo explica la diferencia entre la regularización L1 y L2 a una audiencia no técnica?</i> (2023, 30 marzo). Redes neuronales. Recuperado 15 de octubre de 2024, de <a href="https://www.linkedin.com/advice/3/how-do-you-explain-difference-between-l1-l2-regularization?lang=es&originalSubdomain=es" target="_blank">https://www.linkedin.com/advice/3/how-do-you-explain-difference-between-l1-l2-regularization?lang=es&originalSubdomain=es</a>

