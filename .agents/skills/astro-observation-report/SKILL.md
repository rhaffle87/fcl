---
name: astro-observation-report
description: Skill for writing gravitational-wave observational results papers for compact binary coalescence events (BNS, BBH, NSBH) detected by LIGO, Virgo, or KAGRA. Covers paper architecture, first-page layout, parameter-estimation figures, uncertainty notations, credible-interval plots, statistical conventions, page headers, replicating Physical Review X two-column journal style.
---

# Gravitational-Wave Observational Results Report (LIGO / Virgo / KAGRA)

## Overview

The `astro-observation-report` skill provides specialized architectural, statistical, and typesetting guidance for reporting astrophysical discoveries and observational detections of compact binary coalescences (binary black hole [BBH], binary neutron star [BNS], and neutron star-black hole [NSBH] mergers) detected by gravitational-wave observatories (LIGO, Virgo, KAGRA). It replicates the layout, uncertainty conventions, and two-column typesetting standards of *Physical Review X* and *Physical Review D*.

---

## Trigger Phrases

Invoke this skill when requested to:
- "draft gravitational wave observation paper"
- "format LIGO/Virgo/KAGRA observational report"
- "typeset PRX / PRD astrophysics results"
- "format parameter estimation uncertainties and corner plots"

---

## 1. Paper Architecture (Physical Review X Style)

1. **Title & Front Matter**:
   - Descriptive event designation: *"GW[YYMMDD]_[HHMMSS]: Observation of a [Source Class] at Redshift $z \sim X$"*
   - Author block: Collaboration list with institutional affiliations and author footnotes.
   - Abstract: Terse summary containing trigger timestamp (UTC), network matched-filter signal-to-noise ratio ($\rho$), false alarm rate (FAR), chirp mass ($\mathcal{M}$), component masses ($m_1, m_2$), luminosity distance ($d_L$), effective inspiral spin ($\chi_\mathrm{eff}$), and sky localization area.
2. **§I. Introduction**:
   - Discovery context, detector network operating states (O3, O4 runs), calibrated strain data availability, and environmental monitoring vetoes.
3. **§II. Detection & Data Quality**:
   - Trigger pipelines: Search algorithms (GstLAL, PyCBC, cWB).
   - Significance assessment: Matched-filter SNR ($\rho_\mathrm{mf}$), $\chi^2$ signal consistency checks, background time-slide estimation, and FAR in units of $\mathrm{yr}^{-1}$.
4. **§III. Parameter Estimation**:
   - Bayesian inference models (LALInference, Bilby, RIFT).
   - Waveform approximants: e.g. `IMRPhenomXPHM`, `SEOBNRv4PHM`.
   - Priors: Uniform in detector frame masses, isotropic spins, volumetric distance.
5. **§IV. Astrophysical Implications**:
   - Compact object classification, remnant mass and spin ($M_f, a_f$), peak luminosity, gravitational-wave radiated energy ($E_\mathrm{rad}$), and remnant recoil velocity.
   - Electromagnetic counterpart search summary (kilonova, gamma-ray bursts).
6. **§V. Tests of General Relativity**:
   - Residual test, inspiral-merger-ringdown consistency check, parameterized deviations ($\delta \hat{\varphi}_i$), dispersion relation bounds ($m_g$).
7. **§VI. Summary & Outlook**:
   - Event catalog release, open data release links (GWOSC).

---

## 2. Statistical & Notation Conventions

### Uncertainty Formatting (90% Credible Intervals)
- Report median values accompanied by 90% credible intervals as sub/superscripts:
  $$m_1 = 35.6^{+4.8}_{-3.2}\,M_\odot, \quad m_2 = 28.4^{+3.1}_{-4.0}\,M_\odot$$
  $$d_L = 1240^{+380}_{-410}\,\mathrm{Mpc}, \quad z = 0.24^{+0.07}_{-0.08}$$
- Chirp mass in detector and source frames:
  $$\mathcal{M} = \frac{(m_1 m_2)^{3/5}}{(m_1 + m_2)^{1/5}}, \quad \mathcal{M}^\mathrm{source} = \frac{\mathcal{M}}{1+z}$$
- Spin parameters:
  $$\chi_\mathrm{eff} = \frac{\frac{m_1}{M} \mathbf{S}_1 + \frac{m_2}{M} \mathbf{S}_2}{M} \cdot \hat{\mathbf{L}}, \quad \chi_p \in [0, 1]$$

### Figure Standards
- **Corner Plots**: Posterior probability density functions for coupled parameters ($(m_1, m_2)$, $(\mathcal{M}, q)$, $(d_L, \chi_\mathrm{eff})$) showing 50% and 90% credible contours.
- **Strain Spectrograms**: Q-transform time-frequency spectrograms centered on merger time $t_0$, calibrated in normalized energy.
- **Sky Localization Skymap**: Mollweide projection in equatorial coordinates showing 50% and 90% credible sky areas in $\mathrm{deg}^2$.
