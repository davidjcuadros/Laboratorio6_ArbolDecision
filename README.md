# Laboratorio 6: Clasificación de Riesgo Crediticio con Árboles de Decisión

## Objetivo
Catalogar correctamente el riesgo de los préstamos solicitados por los clientes en tres categorías: alto, medio y bajo, facilitando la toma de decisiones de las entidades financieras respecto a la aprobación de créditos, mediante el uso de árboles de decisión.

## Metodología: CRISP-DM
Se sigue la metodología CRISP-DM, la cual consta de las siguientes fases:
<img width="578" height="564" alt="image" src="https://github.com/user-attachments/assets/53c266f7-15e3-4c16-968b-02507ff478fa" />

1. Entendimiento del negocio  
2. Entendimiento de los datos  
3. Preparación de los datos  
4. Modelado  
5. Evaluación  
6. Despliegue  

## 1. Entendimiento del negocio

Las entidades financieras necesitan evaluar el riesgo asociado a cada solicitud de préstamo para minimizar pérdidas y optimizar la asignación de recursos.

Actualmente, estos procesos suelen ser:
- Manuales o semi-automatizados  
- Costosos en tiempo  
- Dependientes de múltiples fuentes de información  

El objetivo es construir un modelo que permita:
- Automatizar la evaluación del riesgo  
- Clasificar clientes en niveles de riesgo  
- Apoyar decisiones de aprobación o rechazo de crédito  

## 2. Entendimiento de los datos

Se dispone de un dataset con las siguientes características:

- Número de registros: 39,717  
- Número inicial de variables: 111  

Los datos incluyen información sobre:
- Perfil del cliente (ingresos, empleo, vivienda)  
- Historial crediticio  
- Características del préstamo  
- Estado actual del préstamo  

La variable más relevante identificada es:

- loan_status: indica el estado del préstamo  

## 3. Preparación de los datos

Esta fase incluyó limpieza, transformación y selección de variables relevantes.

### 3.1 Eliminación de columnas sin información

Se eliminaron las siguientes variables debido a que no contienen datos en ninguno de los registros (valores nulos):

annual_inc_joint, dti_joint, verification_status_joint, acc_now_delinq, 
tot_coll_amt, tot_cur_bal, open_acc_6m, open_il_6m, open_il_12m, open_il_24m, 
mths_since_rcnt_il, total_bal_il, il_util, open_rv_12m, open_rv_24m, max_bal_bc, 
all_util, total_rev_hi_lim, inq_fi, total_cu_tl, inq_last_12m, acc_open_past_24mths, 
avg_cur_bal, bc_open_to_buy, bc_util, chargeoff_within_12_mths, delinq_amnt, 
mo_sin_old_il_acct, mo_sin_old_rev_tl_op, mo_sin_rcnt_rev_tl_op, mo_sin_rcnt_tl, 
mort_acc, mths_since_recent_bc, mths_since_recent_bc_dlq, mths_since_recent_inq, 
mths_since_recent_revol_delinq, num_accts_ever_120_pd, num_actv_bc_tl, 
num_actv_rev_tl, num_bc_sats, num_bc_tl, num_il_tl, num_op_rev_tl, num_rev_accts, 
num_rev_tl_bal_gt_0, num_sats, num_tl_120dpd_2m, num_tl_30dpd, num_tl_90g_dpd_24m, 
num_tl_op_past_12m, pct_tl_nvr_dlq, percent_bc_gt_75, pub_rec_bankruptcies, 
tax_liens, tot_hi_cred_lim, total_bal_ex_mort, total_bc_limit, 
total_il_high_credit_limit, collections_12_mths_ex_med, 
mths_since_last_major_derog, mths_since_last_record

### 3.2 Eliminación de columnas constantes

Se eliminaron variables con valores constantes en todos los registros, ya que no aportan información al modelo:

policy_code, application_type, initial_list_status, pub_rec

### 3.3 Eliminación por alta ausencia de datos

Se eliminó la variable:

next_pymnt_d

Debido a que solo el 2.741% de los registros contiene información.

### 3.4 Eliminación de variables irrelevantes

Se eliminó la variable:

url

Por no aportar valor analítico al modelo.

### 3.5 Prevención de data leakage

Se excluyeron variables que contienen información posterior al otorgamiento del préstamo:

total_pymnt, last_pymnt_d, recoveries, collection_recovery_fee

Estas variables no deben ser usadas porque introducen información del futuro en el modelo.

### 3.6 Transformación de la variable objetivo

Se transformó la variable loan_status en una nueva variable categórica llamada riesgo:

| loan_status | riesgo | interpretación |
|------------|--------|----------------|
| Charged Off | Alto  | cliente incumplió el pago |
| Current     | Medio | cliente en proceso de pago (riesgo incierto) |
| Fully Paid  | Bajo  | cliente cumplió completamente |
<img width="528" height="200" alt="image" src="https://github.com/user-attachments/assets/83c583f9-c3e0-4447-a83e-83b81177d219" />

### 3.7 Reducción del dataset

Variables iniciales: 111  
Variables finales: 42  

## 4. Modelado

Se implementará un modelo de árbol de decisión para clasificar el riesgo crediticio.

Implementación en código.

Se ejecuta python train.py, programa que nos ayudará a entrenar el árbol de decisión:
<img width="1139" height="349" alt="image" src="https://github.com/user-attachments/assets/40609a50-3085-4262-b6f5-6185733f8dec" />

Se ejecuta python predict.py, programa que predice dos clientes que fueron evaluados:
<img width="365" height="183" alt="image" src="https://github.com/user-attachments/assets/956101ec-366f-4e21-8563-86ff692e5e4c" />


## 5. Evaluación
<img width="451" height="323" alt="image" src="https://github.com/user-attachments/assets/dd458660-70e8-4267-b669-668f43564db7" />

Dado el desbalance en la distribución de clases del dataset donde la clase Bajo representa la gran mayoría de los registros frente a Alto y Medio, fue necesario incorporar el parámetro class_weight="balanced" en el DecisionTreeClassifier. Sin este ajuste, el modelo tendía a clasificar casi todos los casos como Bajo, obteniendo un accuracy artificialmente alto del 83% pero fallando completamente en detectar las clases minoritarias (recall de 0.01 para Alto y 0.00 para Medio). Al aplicar el balanceo, el algoritmo penaliza proporcionalmente los errores en las clases con menor representación, forzando al modelo a prestarles atención durante el entrenamiento.

Con este ajuste, el accuracy global descendió al 60.38%, lo cual es esperado y no debe interpretarse como un deterioro del modelo, simplemente ya no se beneficia del sesgo hacia la clase mayoritaria. Los resultados muestran mejoras sustanciales donde más importa: Medio pasó de un recall de 0.00 a 0.95, logrando identificar correctamente 217 de 228 casos, y Alto mejoró de 0.01 a 0.28, acertando 318 de 1,125. La clase Bajo redujo su recall de 1.00 a 0.65, lo cual es un sacrificio necesario y aceptable para lograr un modelo más equilibrado y útil en la práctica. En un contexto de riesgo crediticio, no detectar un cliente de riesgo Alto o Medio es mucho más costoso que clasificar incorrectamente uno de riesgo Bajo, por lo que este trade-off está justificado.

## 6. Despliegue
- No realizado.
