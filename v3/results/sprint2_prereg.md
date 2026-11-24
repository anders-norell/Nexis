# Sprint 2 – förregistrering (låst 24 nov 2026, före all data för schemat)

Godkänd plan: projektet `claude/nexis_v3_sprint2_plan.md` (Anders 24 nov 12:10). Precisering nedan är [Claudes förslag]
och ändras inte efter att schemadata setts. Ändringar kräver Anders godkännande och redovisas.

**Mått A (låst):** summa över meningens tokens, medel över lager 9–17, huvuden och svarspositioner (rad n_p−1 … L−2),
normerat över de fyra meningarna (`nexis3/attention.py`, commit vid denna fil). A_first och A_mean redovisas.

**Data:** testtexter v2. Träning frö 2000 (1 200 texter), test frö 3000 (300 texter). Inga gemensamma texter (kontrollerat: 0).
Schemats indata: dolda tillstånd i lager 9, 13, 17, bara promptens positioner (medel per mening + sista prompttoken).

**Kriterium G2a (schemat förutsäger A):** på testtexterna, andel där argmax S = argmax A. Jämförs parat per text mot
två nollmodeller: (i) senaste meningen (nr 4) och (ii) ordöverlapp med frågan (`overlap_choice`). Godkänt om skillnaden
> 2 SE (parat SE över texterna) mot båda.

**Kriterium G2b (styrning, tvångsprov):** S ersätts med en punktfördelning på en slumpvis vald icke-målmening
(frö 4000). Mått: A på den valda meningen med styrning β > 0 minus med β = 0, parat per text på testtexterna.
Godkänt om > 2 SE. β väljs på träningstexter innan testtexterna används.

**Redovisas oavsett:** ablation (β = 0 mot β > 0: A på målet, uppgiften rätt), korrelation S–A, kalibrering.
