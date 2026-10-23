"""Registered conjunctive decision; no threshold depends on observed outcomes."""
def verdict(metrics, completeness, integrity=True):
    if not integrity:
        return 'SC_D_PDR_INTERFACE_NOT_ESTABLISHED'
    if min(completeness.values()) < 45:
        return 'INCONCLUSIVE_DUE_TO_SC_D_MISSINGNESS'
    j, l, e, g = [metrics[k] for k in ('J_exact','log_odds','weight_error','gradient_cosine')]
    behavior = j['mean']>0 and j['ci95'][0]>0 and j['positive']>=40 and l['mean']>0 and l['ci95'][0]>0
    signal = e['mean']<0 and e['ci95'][1]<0 and g['mean']>0 and g['ci95'][0]>0
    reliability = all(metrics[k]['lower_one_sided_95']>-.01 for k in ('gate_sequence_success','safe_route_success','full_task_success'))
    if behavior and not reliability:
        return 'SC_D_PDR_EARLY_GAIN_WITH_RELIABILITY_HARM'
    if signal and not behavior:
        return 'SC_D_PDR_SIGNAL_RECOVERY_PRESENT_BEHAVIOR_GAIN_NOT_CONFIRMED'
    if behavior and not signal:
        return 'SC_D_PDR_BEHAVIOR_GAIN_WITHOUT_REGISTERED_SIGNAL_BRIDGE'
    if behavior and signal and reliability:
        return 'SC_D_PDR_CLOSED_LOOP_REALIZATION_CONFIRMED'
    return 'SC_D_PDR_GAIN_NOT_CONFIRMED'
