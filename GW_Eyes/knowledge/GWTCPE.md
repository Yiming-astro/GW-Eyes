# GWTC-4 Parameter Estimation Notice  

## Waveform-Dependent Systematic Differences

In GWTC-4, several BBH candidates show significant systematic differences in posterior distributions across waveform models, particularly in:

- Mass ratio (q)  
- Component masses (m1, m2)  
- Spin parameters (χ_eff, χ_p, χ_1)  
- Presence of multimodal posteriors  

These differences indicate strong model dependence and increased uncertainties.

---

### GW230624_113103
- Total mass: M = 43.8 (+11.1 / -6.6) M_sun  
- SEOBNRv5PHM favors more asymmetric masses than IMRPhenomXPHM_SpinTaylor  
- Larger χ_eff and χ_p inferred with SEOBNRv5PHM  

Key issue: waveform-dependent mass ratio and spin estimates

---

### GW231028_153006
- Strong systematic variations across all waveform models  
- No close agreement in component mass posteriors  
- M = 152 (+29 / -14) M_sun  
- m1 = 95 (+33 / -20) M_sun  
- Model-dependent χ_eff; secondary high-χ_p mode in IMRPhenomXO4A  

Key issue: extreme mass system with highly model-dependent inference

---

### GW231118_005626
- M = 30.9 (+5.3 / -3.6) M_sun
- q = 0.55 (+0.37 / -0.22)  
- χ_eff ≈ 0.4 ± 0.1  
- IMRPhenomXPHM_SpinTaylor shows heavier mass tail and higher χ_1, χ_p  

Key issue: spin and high-mass tail sensitive to waveform choice

---

### GW231118_090602
- M = 20.7 (+10.2 / -2.3) M_sun  
- q = 0.56 (+0.38 / -0.41)  
- Multimodal posteriors only in IMRPhenomXPHM_SpinTaylor:
  - High-likelihood mode at q ~ 0.15 with χ_eff ~ 0.4 and larger χ_p  
  - Broader mode at 0.3 ≲ q ≲ 1  
- SEOBNRv5PHM gives unimodal results consistent with higher-q mode  

Key issue: waveform-induced multimodality

---

### GW231123_135430
- Significant waveform-dependent differences in inferred masses and spins  

Key issue: strong systematic uncertainty across models

---

## RAG Usage Note

When referencing or analyzing the following events, explicitly state that their parameter estimates are strongly waveform-model dependent:

- GW230624_113103  
- GW231028_153006  
- GW231118_005626  
- GW231118_090602  
- GW231123_135430  

Uncertainties for these sources are larger than typical GWTC-4 events due to systematic model differences.
