# Revisión de estilo externa de paper/main.tex (2026-11-03)

Informe de un agente revisor independiente (sin contexto del proyecto), encargado de
evaluar sólo la escritura: naturalidad, claridad, coherencia e interés. No toca la física.
Números de línea referidos a main.tex en el commit 65622cb.

## Veredicto global

Escala 1 (prosa claramente generada) a 5 (científico cuidadoso que escribe bien): **4**.
Sin tics léxicos de LLM (0 "Moreover", 0 "Importantly", 0 "Indeed", 0 "It is worth noting",
0 "Furthermore"; grep), sin tripletes mecánicos, inglés correcto. Lo que fatiga:
(a) párrafos-bloque de resultados numéricos (431–467, 653–684, 742–763) con apartes
defensivos y rastro de las revisiones; (b) exceso de "therefore" (13) y "the same" (21), y
98 de ~300 frases empiezan por "The …"; (c) frases con sujeto al final o dobles relativas,
y algún calco ("Already without the CMB", "as it should", "then … then").

Tres cambios de mayor efecto:
1. Reestructurar 431–467 (diamante) en tres párrafos: exponente local; ajuste de r y cota
   de δ; comprobaciones (bootstrap, estimadores, rango). Ruta del JSON al data statement.
2. Cerrar la Discusión con una idea, no con la reivindicación de prioridad (1040–1042):
   mover ésa al principio y terminar diciendo qué zanjaría lo abierto.
3. Quitar del cuerpo las huellas de revisión: "additional test (see the methodological
   note)" ×2, "(below)" ×3, aparte 436–441. La nota metodológica ya lo documenta.

## Hallazgos (por impacto)

1. **431–467** párrafo sobrecargado (37 líneas, ~20 números, 3 apartes). Dividir en tres:
   - P1 exponente: "Figure 1 shows … The local exponent of the total variance decreases with ρV like that of J¹, from 1.24±0.04 at ρV≈35 to 0.72±0.05 at ρV≈4500 and 0.71±0.07 between 12 800 and 25 600. The last two values lie 2.6σ and 1.5σ below those of J¹ (0.85 and 0.81); local exponents from neighbouring densities are noisy, however, and the fit of the remainder is the more robust statement."
   - P2 remainder: "Combining all independent sprinklings (…), r = 0.321±0.012, i.e. (ΔS)² ≃ 1.32 J¹, with slope +0.002±0.007 (0.3σ; figure 1(c)). The two new densities give r = 0.38±0.10 and 0.19±0.08, within 0.6σ and 1.7σ of the earlier mean. The fit has χ² = 16.0 for 9 degrees of freedom … Scaling the slope error by √(χ²/dof), −0.05 < δ < 0.06 at 2σ, i.e. p ≤ 0.78. An exponent as low as the Poisson value would require r → −1, correlations that cancel J¹; we see no trend in that direction."
   - P3 comprobaciones: estimadores combinados, bootstrap (sin ruta de fichero), estabilidad del rango.
2. **1040–1042** la Discusión termina con "to our knowledge this is the first confrontation…". Moverla tras la línea 992 y cerrar: "Two results would settle what remains open. A calculation of the connected term J⁶_c for ε → 0 at fixed ερV would decide whether the exponent 3/4 survives strong smearing; and a measurement of the temporal memory of the smeared action would remove the main assumption of the data test. If both confirm the local-action behaviour, the conclusion is sharp: the BD action inherits the boost range of the region, and any mechanism for an everpresent Λ must find a way to discard it."
3. **139–147** ítem 3 de resultados, frase anidada, "They" ambiguo. → "Only the strongly smeared action has an acceptable amplitude. We therefore assume that two quantities measured for the local action, the exponent and the temporal correlation along nested past cones, carry over to the smeared one, and on that assumption we confront stochastic Λ(t) histories with supernova, BAO and CMB data. Over the measured range of the memory these histories are excluded, as is Model 1 in our implementation; within a wider family of dynamics, rare good histories appear only when Λ stays close to a slowly wandering constant (section 6)."
4. **85–88** "then … and … then" (calco). → "Write the fluctuation of the action as ΔS ∝ V^p (this exponent p is used throughout). Then δΛ ∝ V^{p−1} and, since V ∼ H^{−4}, δΛ/H² ∝ H^{2−4p}: only p = 1/2 gives an \emph{everpresent} Ω_Λ."
5. **734–736** sujeto al final. → "The volume of the causal past of a comoving observer today, back to the big bang in a Planck 2018 ΛCDM background [Planck2018], is V ≈ 6.4×10²⁴² ℓ_p⁴ (computed by quadrature in conformal time)."
6. **841–847** "good" usado antes de definirlo. → "A seed is \emph{good} if χ² < χ²_ΛCDM + 14.7. In every run, the 50th-best grid seed lies further above this threshold than the largest gain that refinement produced among the 50, so no seed outside the 50 could become good unless it gained more than all of them."
7. **917–918** "Already without the CMB" → "Even without the CMB".
8. **710–713** "its δΛ … as it should" → "i.e. the δΛ = 8πα/√V of Model 1, as it must, since Model 1 defines its fluctuating term through the same relation."
9. **675–679** doble relativa "which leaves … but which". → "For ε = 0.1 it decreases: the correlations grow more slowly than J¹. If the decrease saturates, or r decays to zero, p = 3/4 survives; if instead the linear trend in ln ρV continued, the (negative) correlations would eventually cancel part of J¹."
10. **653–657** criterio elíptico ("and the new points with the earlier mean"). → "We tested this explanation with a new criterion, adding 500 sprinklings at each of ρV = 12 800 and 25 600 (which extend ερV to 2560) to both earlier sets: the slope of the unprojected r must be compatible with zero at 4σ, and the new points must be compatible with the earlier mean."
11. **979–983** "the number of … whose number" → "…nearly null intervals, and that number is set by the volume of the boost group that fits inside the region".
12. **436–441** aparte defensivo. → "The last two values lie 2.6σ and 1.5σ below those of J¹ (0.85 and 0.81), as a slowly decreasing r would produce; local exponents from neighbouring densities are noisy, however, and the bound on δ from the fit of the remainder, which covers this deviation, is the more robust statement."
13. **462–463** ruta `figures/goal6_bootstrap.json` en el cuerpo → al data statement ("including the bootstrap check of the variance errors").
14. **443, 573, 422** "additional test (see the methodological note)" ×2 y "the tests called additional below" → quitar la etiqueta del cuerpo; que la nota diga qué puntos se añadieron.
15. **111–112** "Efficient codes compute the BD action of large sprinkled causal sets [Cunningham2018]." huérfana → mover a 4.1 (tras 415) o eliminar.
16. **180–195** normalización de MYZ intercalada; "they" lejos; "r" usado antes de definirlo → mover la frase de MYZ al final del párrafo, entre paréntesis, con "leaves every ratio and exponent of this paper unchanged"; "In two dimensions the coefficients are".
17. **278–280** "dominate asymptotically \emph{exactly}" → "we compute \emph{exactly} the terms that dominate asymptotically, and measure the remaining terms on sprinklings."
18. **495–501** "and not Σf(n)² alone" colgando → "The coefficients do not add up (23.2 + 1.76 ≠ 24.43) because J¹ is still curving towards its limit and the two fits weight the densities differently. Our estimate of the asymptotic coefficient is therefore the asymptotic Σ_n f(n)² = 24 plus the correlation coefficient, b ≈ 25.8, i.e. r₂ ≈ 0.07±0.01, rather than Σ_n f(n)² alone."
19. **562–568** "For instance" ilustra la frase equivocada → intercambiar frases o "The prefactor G is also tested directly:".
20. **866–880** "This holds" ambiguo; paréntesis final suelto → dividir en dos párrafos; "Distance priors assume standard early physics. The BD model satisfies this (Λ/3H² ∼ 4×10⁻⁵ at z = 1100); Model 1 does not (∼0.5), so we fit it to supernovae and BAO only. … For completeness, with the CMB included its minimum is +359."
21. **921–923** "here by an independent route" → "…which found Δχ² > 150; our route is independent and does not use the CMB."
22. **742–755** rango y relativa encajados → "This is below the bound ℓ_k ≲ 10⁻¹⁹ m … from 8 TeV LHC data [Biswas2015, as quoted in Belenchia2016], and inside the window 10⁻²⁹–2×10⁻²² m that proposed optomechanical experiments could probe [Belenchia2016]."
23. **767** frase que empieza por símbolo → "The quantity ΔS is a standard deviation over sprinklings, whereas…".
24. **283–290** sección 3 sin frase puente → anteponer "We now compute the term of (6) that dominates asymptotically, J¹, and show that it is controlled by the range of boosts that fits inside the region."
25. **165–175** sección 2 sin propósito → "This section fixes notation, recalls the MYZ decomposition of the variance, and explains why we do not evaluate it in full."
26. **72–83** gancho de libro de texto; el gancho real está en 81–88 → abrir con Λ: "If spacetime is a causal set, the cosmological constant may fluctuate. …" o fusionar las dos primeras frases.
27. **349–350** título "Why the correlation terms should follow J¹" seguido de negación → "Should the correlation terms follow J¹?" y "We have no proof …, but there is a reason to expect it."
28. **413–414** "unlike Markov-chain sampling, no thermalisation is needed" → eliminar o "Each sprinkling is an independent sample of the ensemble."
29. **597** "(2000 to 500 sprinklings per density)" → "(2000 sprinklings at the lowest density, 500 at the highest)".
30. **539, 549** "⟨N_k⟩/(8)" → definir ⟨N_k⟩_∞ y usar "⟨N_k⟩/⟨N_k⟩_∞".
31. **575–576** "two dimensional" → "two-dimensional".
32. **40** "causal-set inspired proposal" → "a proposal inspired by causal sets".
33. **52–54** "rather than the Poisson value" tras dos proporcionalidades → "rather than the Poisson exponents 1/2 and −1/2".
34. **692–695** tres avisos con punto y coma → "To connect the action with Λ we follow the logic of Everpresent Λ. The relation we adopt, (12), is a working hypothesis rather than a consequence of a causal-set dynamics; the comparison with Model 1 below only checks its normalisation."
35. **804–808** "For it we assume" → "For that regime we assume".
36. **968–973** dos niveles de paréntesis y "that concerns" → dividir en dos frases: "… [DESI2025]). That preference concerns a smooth drift of the mean; the models tested here are stochastic processes whose typical histories wander far more than the data allow."
37. **1005** "Our conclusions have the following limitations." → "Four limitations qualify these conclusions."
38. **Tiempo verbal**: "We evaluated" (268), "We sprinkled" (597), "We tested" (653), "We measured" (774) frente a "We use/evaluate/model/follow"; 1079 mezcla ("We evaluate … and checked"). Presente para el método; pasado sólo en la historia de los criterios.

## Problemas sistémicos (grep)

- "therefore": 13 (135, 318, 333, 378, 515, 650, 663, 680, 720, 740, 774, 874, 965). Dejar 5–6 (333, 650, 663, 740, 965).
- "the same …": 21; 6 en 320–368 y 2 en 598–600. Variar.
- "only": 28; "the geometry only fixes/sets the amplitude" en 54, 581 y 342.
- Frases que empiezan por "The": 98 de ~300 ("For" 32, "In" 24, "We" 22). Ritmo plano en la sección 4.
- "compatible with" 9 / "consistent with" 5, intercambiables (136, 446). Fijar: compatible = contraste estadístico; consistent = acuerdo entre valores.
- Referencias relativas "(below)", "above" en 436, 440, 605, 749, 821 → referencias explícitas.
- Huellas de revisión: "additional" ×5, "pre-registered" ×4, "methodological note" ×2, "earlier drafts" ×1, "a posteriori" ×1, "test suite" ×2, "repository" ×3. Reducir a las indispensables (604 y 793).
- Repetición: "rare good histories appear only when Λ stays close to a slowly wandering constant" ×3 (64–65, 146–147, 965–966); "a constant fraction of" ×4.
- Ortografía británica consistente (0 formas americanas en prosa). No tocar.
- Terminología consistente (light cone, spacetime, non-local, big bang, same-pair, boost range). Única excepción: "two dimensional" (576).
- "i.e.\" ×7; en 448, 454 y 500 seguido de otro número entre paréntesis.

## Lo que ya está bien (no tocar)

- 341–347 (enunciado del resultado: "The region enters only through G…").
- 149–162 ("The mechanism behind the first result is well known in another guise…").
- 286–290 (arranque físico de la sección 3).
- 466–467 ("An exponent as low as the Poisson value would require r → −1…").
- 514–515 ("Everpresent Λ involves the causal past of an observer, not a diamond.").
- 604–609 y 649–651 (criterio prerregistrado que falló y su explicación).
- 994–1003 (mejor párrafo del artículo: "This does not refute Everpresent Λ as such…").
- 417–427 (nota metodológica).
- 1007–1038 (lista de limitaciones).
- Ortografía británica, terminología, agradecimientos y data statement.
