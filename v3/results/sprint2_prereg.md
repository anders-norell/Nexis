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

## Tillägg 24 nov ca 14:50 (före all styrning på riktiga modellen; G2a redan bedömd, se `sprint2_schema.json`)

**Styrningens form:** bias β·log(4·S_s) på attention-poängen mot mening s tokens, lager 9–17, rader från sista
prompttoken och framåt (`nexis3/steer.py`). Tvingat S = 0,97 på vald mening, 0,01 på övriga.
A mäts som förut (lager 9–17). Eftersom biasen läggs i samma lager som A mäts i, redovisas också A_late
(lager 18–27, ej direkt styrda) och om svaret ändras. Det är de mått som visar att styrningen får följder.

**Val av β (regel låst nu):** på 40 valideringstexter (träningsfröets texter 1000–1039, ej använda för schemats
vikter) prövas β ∈ {0,5; 1; 2; 4}. β* är det minsta β där (a) A på den tvingade meningen ökar med minst 0,20 i medel
jämfört med β = 0, och (b) uppgiften med schemats egen S är rätt minst lika ofta som med β = 0, minus 5 procentenheter.
Uppfyller inget β båda villkoren väljs det största β som uppfyller (b). Testtexterna används inte för valet.

## Granskningen 24 nov (`sprint2_review.md`) och FÖRSLAG till ändring – väntar på Anders, inget kört på riktiga modellen

Granskaren visade att **G2b och β-villkor (a) är sanna per konstruktion**. Biasen läggs i samma lager och rader som A mäts i,
så A flyttar redan av aritmetiken (+0,46 vid β = 0,5). Förslag [Claudes förslag, efter granskarens]:
- **G2b′ (ersätter G2b):** när schemat tvingas peka på en annan mening blir svaret fel oftare än utan styrning.
  Mått: andel fel svar i det tvingade villkoret minus vid β = 0, parat på testtexterna. Godkänt om skillnaden > 2 SE.
  Redovisas: om svaret nämner den tvingade meningens värde (bara fyllnadsmeningar), samt A_late (lager 18–27).
- **β-villkor (a′):** ersätter (a). Den tvingade styrningen gör svaret fel i minst 20 % av de valideringstexter som
  besvarades rätt vid β = 0. Villkor (b) och regeln i övrigt är oförändrade.
- **Kontroll "perm" (ny, beskrivande):** schemats egen S med värdena flyttade ett till tre steg mellan meningarna, med
  samma β. Uppgift rätt med egen S minus med perm, parat. Kontrollen visar om schemats innehåll spelar roll.
- **Begränsning (redovisas):** schemat generaliserar inte till ämnen utanför träningen. Med fyra av tolv ämnen
  undanhållna träffar det 33 %, mot 69 % för "alltid målmeningen". G2a gäller de kända mallarna. Det är viktigt för
  kärntestet ("fall modellen inte tränats på") och tas upp i planen för sprint 3.
- Starkare nollmodeller redovisas i efterhand: "alltid målmeningen" 69 %, ämne + position 77 %. Schemat slår båda
  (z 5,4 och 3,5). Det ändrar inte G2a.
