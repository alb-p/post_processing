# Fairness metrics computed using division operator
def fairness_metrics_division(TP_discr, TN_discr, FP_discr, FN_discr, len_discr,
                              TP_priv, TN_priv, FP_priv, FN_priv, len_priv, threshold):
    
  GroupFairness_discr = (TP_discr+FP_discr)/len_discr
  GroupFairness_priv = (TP_priv+FP_priv)/len_priv
  if GroupFairness_priv == 0:
    GroupFairness = 2  #max value
  else:
     GroupFairness = GroupFairness_discr/GroupFairness_priv

  if TP_discr+FP_discr == 0:
    PredictiveParity_discr = 0
    PredictiveParity = 0  #min value
  else:
    PredictiveParity_discr = (TP_discr)/(TP_discr+FP_discr)
  if TP_priv+FP_priv == 0:
    PredictiveParity_priv = 0
    PredictiveParity = 2  #max value
  else:
    PredictiveParity_priv = (TP_priv)/(TP_priv+FP_priv)
  if PredictiveParity_discr != 0 and PredictiveParity_priv != 0:
    PredictiveParity = PredictiveParity_discr/PredictiveParity_priv
  elif PredictiveParity_priv == 0:
    PredictiveParity = 2  #max value
  else:
    PredictiveParity = 0  #min value

  if TN_discr+FP_discr == 0:
    PredictiveEquality_discr = 0
    PredictiveEquality = 0  #min value
  else:
    PredictiveEquality_discr = (FP_discr)/(TN_discr+FP_discr)
  if TN_priv+FP_priv == 0:
    PredictiveEquality_priv = 0
    PredictiveEquality = 2  #max value
  else:
    PredictiveEquality_priv = (FP_priv)/(TN_priv+FP_priv)
  if PredictiveEquality_discr != 0 and PredictiveEquality_priv != 0:
    PredictiveEquality = PredictiveEquality_discr/PredictiveEquality_priv
  elif PredictiveEquality_priv == 0:
    PredictiveEquality = 2  #max value
  else:
    PredictiveEquality = 0  #min value

  if FN_priv+TP_priv == 0:
    EqualOpportunity_priv = 0
    EqualOpportunity = 2  #max value
  else:
    EqualOpportunity_priv = (FN_priv)/(TP_priv+FN_priv)
  if FN_discr+TP_discr == 0:
    EqualOpportunity_discr = 0
    EqualOpportunity = 0  #min value
  else:
    EqualOpportunity_discr = (FN_discr)/(TP_discr+FN_discr)
  if EqualOpportunity_priv != 0 and EqualOpportunity_discr != 0:
    EqualOpportunity = EqualOpportunity_discr/EqualOpportunity_priv
  elif EqualOpportunity_priv == 0:
    EqualOpportunity = 2  #max value
  else:
    EqualOpportunity = 0  #min value

  if FN_discr+TP_discr == 0:
    EqualizedOdds1 = 0
    EqualizedOdds = 0 #min value
  elif FN_priv+TP_priv == 0:
    EqualizedOdds1 = 0
    EqualizedOdds = 2 #max value
  elif (TP_priv/(TP_priv+FN_priv)) == 0:
    EqualizedOdds1 = 2 #max value
  else:
    EqualizedOdds1 = ((TP_discr/(TP_discr+FN_discr)) / (TP_priv/(TP_priv+FN_priv))) # (1-equalOpportunity_discr)/(1-equalOpportunity_priv)
  if TN_priv+FP_priv == 0:
    EqualizedOdds2 = 0
    EqualizedOdds = 2 #max value
  elif TN_discr+FP_discr == 0:
    EqualizedOdds2 = 0
    EqualizedOdds = 0 #min value
  elif (FP_priv/(TN_priv+FP_priv)) == 0:
    EqualizedOdds2 = 2 #max value
  else:
    EqualizedOdds2 = ((FP_discr/(TN_discr+FP_discr)) / (FP_priv/(TN_priv+FP_priv))) # = PredictiveEquality
  if EqualizedOdds1 != 0 and EqualizedOdds2 != 0:
    EqualizedOdds = and_function(EqualizedOdds1, EqualizedOdds2, threshold)
  else:
    EqualizedOdds = 2 #max value

  if TP_discr+FP_discr == 0 or TN_discr+FP_discr == 0:
    ConditionalUseAccuracyEquality1 = 0
    ConditionalUseAccuracyEquality= 0 #min value
  elif (TP_priv/(TP_priv+FP_priv)) == 0:
    ConditionalUseAccuracyEquality1 = 2 #max value
  else:
    ConditionalUseAccuracyEquality1 = ((TP_discr/(TP_discr+FP_discr)) / (TP_priv/(TP_priv+FP_priv)))
  if TN_discr+FN_discr == 0 or TN_priv+FN_priv == 0:
    ConditionalUseAccuracyEquality2 = 0
    ConditionalUseAccuracyEquality = 2 #max value
  elif (TN_priv/(TN_priv+FN_priv)) == 0:
    ConditionalUseAccuracyEquality2 = 2 #max value
  else:
    ConditionalUseAccuracyEquality2 = ((TN_discr/(TN_discr+FN_discr)) / (TN_priv/(TN_priv+FN_priv)))
  if ConditionalUseAccuracyEquality1 != 0 and ConditionalUseAccuracyEquality2 != 0:
    ConditionalUseAccuracyEquality = and_function(ConditionalUseAccuracyEquality1, ConditionalUseAccuracyEquality2, threshold)
  else:
    ConditionalUseAccuracyEquality = 2 #max value

  OAE_discr = (TP_discr+TN_discr)/len_discr
  OAE_priv = (TP_priv+TN_priv)/len_priv
  if OAE_priv != 0 and OAE_discr != 0:
    OverallAccuracyEquality = OAE_discr/OAE_priv
  elif OAE_priv == 0:
    OverallAccuracyEquality = 2  #max value
  else:
    OverallAccuracyEquality = 0 #min value

  if FP_priv == 0:
    TreatmentEquality_priv = 0
    TreatmentEquality = 2  #max value
  else:
    TreatmentEquality_priv = (FN_priv/FP_priv)
  if FP_discr == 0:
    TreatmentEquality_discr = 0
    TreatmentEquality = 0 #min value
  elif (FN_discr/FP_discr) == 0:
    TreatmentEquality_discr = 0 #max value
    TreatmentEquality = 0 #min value
  else:
    TreatmentEquality_discr = (FN_discr/FP_discr)
  if TreatmentEquality_priv != 0 and TreatmentEquality_discr != 0:
    TreatmentEquality = TreatmentEquality_discr/TreatmentEquality_priv
  elif TreatmentEquality_priv == 0:
    TreatmentEquality = 2 #max value
  else:
    TreatmentEquality = 0 #min value

  if TN_priv+FN_priv == 0:
    FORParity_priv = 0
    FORParity = 2 #max value
  else:
    FORParity_priv = (FN_priv)/(TN_priv+FN_priv)
  if TN_discr+FN_discr == 0:
    FORParity_discr = 0
    FORParity = 0  #min value
  elif (FN_discr)/(TN_discr+FN_discr) == 0:
    FORParity_discr = 0
    FORParity = 0 #min value
  else:
    FORParity_discr = (FN_discr)/(TN_discr+FN_discr)
  if FORParity_priv != 0 and FORParity_discr != 0:
    FORParity = FORParity_discr/FORParity_priv
  elif FORParity_priv == 0:
    FORParity = 2 #max value
  else:
    FORParity = 0 #min value


  FN_P_discr = (FN_discr)/len_discr
  FN_P_priv = (FN_priv)/len_priv
  if FN_P_priv == 0:
    FN_metric = 2  #max value
  else:
    FN_metric = FN_P_discr/FN_P_priv


  FP_P_discr = (FP_discr)/len_discr
  FP_P_priv = (FP_priv)/len_priv
  if FP_P_priv == 0:
    FP_metric = 2  #max value
  else:
    FP_metric = FP_P_discr/FP_P_priv


  #RecallParity = (TP_discr/(TP_discr+FN_discr))/(TP_priv/(TP_priv+FN_priv))

  metrics = {}
  metrics['GroupFairness'] = [GroupFairness, GroupFairness_discr, GroupFairness_priv]
  metrics['PredictiveParity'] = [PredictiveParity, PredictiveParity_discr, PredictiveParity_priv]
  metrics['PredictiveEquality'] = [PredictiveEquality, PredictiveEquality_discr, PredictiveEquality_priv]
  metrics['EqualOpportunity'] = [EqualOpportunity, EqualOpportunity_discr, EqualOpportunity_priv]
  metrics['EqualizedOdds'] = [EqualizedOdds, EqualizedOdds1, EqualizedOdds2]
  metrics['ConditionalUseAccuracyEquality'] = [ConditionalUseAccuracyEquality, ConditionalUseAccuracyEquality1 , ConditionalUseAccuracyEquality2]
  metrics['OverallAccuracyEquality'] = [OverallAccuracyEquality, OAE_discr, OAE_priv]
  metrics['TreatmentEquality'] = [TreatmentEquality, TreatmentEquality_discr, TreatmentEquality_priv]
  metrics['FORParity'] = [FORParity, FORParity_discr, FORParity_priv]
  metrics['FN'] = [FN_metric, FN_P_discr, FN_P_priv]
  metrics['FP'] = [FP_metric, FP_P_discr, FP_P_priv]

  for k in metrics.keys():
    value = standardization(rescale(metrics[k][0]))
    discr = metrics[k][1]
    priv = metrics[k][2]
    metrics[k] = {
      'Value': value,
      'Discr_group': discr,
      'Priv_group': priv
    }

  return metrics

# # Standardize and scale the metrics
#     for key, value in metrics.items():
#         metrics[key] = {
#             'Value': standardize_and_rescale(value[0]),
#             'Discr_group': value[1],
#             'Priv_group': value[2]
#         }

#     return metrics

def rescale(metric):
  metric = metric - 1
  return metric

def standardization(metric):
  if metric > 1:
    metric = 1
  elif metric < -1:
    metric = -1
  return metric

def valid(metric, th):
  if metric > 1-th and metric < 1+th:
    return True
  return False

def sub_valid(metric, th):
  if metric > -th and metric < th:
    return True
  return False

def sub_and_function2(m1, m2):
  return m1 if abs(m1) > abs(m2) else m2

def and_function(m1,m2, th):
  m1_new = rescale(m1)
  m2_new = rescale(m2)
  m = sub_and_function(m1_new, m2_new, th)
  return m+1

def sub_and_function(m1, m2, th):
  max_value = max(abs(m1), abs(m2))
  if max_value == abs(m1):
    return m1
  else:
    return m2