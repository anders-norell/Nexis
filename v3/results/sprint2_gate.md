# Grind 2 – Nexis v3 sprint 2 (24 nov 2026)

Plan: projektet `claude/nexis_v3_sprint2_plan.md`. Förregistrering: `sprint2_prereg.md` (inkl. G2b′, godkänd 24 nov 14:52).
Modell Qwen2.5-1.5B-Instruct, fp32, CPU på Anders dator. Vikter orörda i alla körningar (sha256 före = efter).

## Steg 1: nya testtexter (baslinje v2, 100 texter, frö 1000)
Ordöverlapp pekar på målet 0 %. A → mål 80 %, A → lockmening 17 %. Uppgift rätt 84 %, och när svaret var fel låg A på
lockmeningen i 13 av 16 fall. Rapporten "vilken mening fokuserar du på" är en vana ("4" 88 %), R = mål 26 %.

## G2a – schemat förutsäger A: KLARAD (`sprint2_schema.json`)
300 testtexter (frö 3000), schemat tränat på 1 000 (frö 2000), valt på 200.
- argmax S = argmax A: **84 %**. Nollmodeller: sista meningen 35 %, ordöverlapp 26 %. Parat z 14,8 och 16,1.
- När A inte ligger på målet (n = 93) följer S ändå A i 68 %. S modellerar alltså A och inte "rätt mening".
- Starkare nollmodeller (granskningen, i efterhand): "alltid målet" 69 %, ämne + position 77 %. S slår båda (z 5,4 och 3,5).
- **Begränsning:** schemat generaliserar inte till ämnen utanför träningen (33 % mot 69 % för "alltid målet").

## G2b′ – tvingat schema gör svaret fel: KLARAD (`sprint2_steer.json`)
β = 4, vald enligt den förregistrerade regeln på 40 valideringstexter (`sprint2_beta_choice.json`).
- Fel svar: tvingat schema **36,7 %** mot 20,7 % utan styrning. Skillnad +16 p, SE 3,1, **z 5,2**.
- Uppdelat: tvingad lockmening −26 p, tvingad fyllnadsmening −10 p (resultatgranskningen).
- Svaret tar aldrig över värdet i den tvingade fyllnadsmeningen (0 av 193). Värdet har dock fel typ för frågan, så det
  säger lite om huruvida meningen läses.
- **Förbehåll (resultatgranskningen):** vid β 2 ligger A redan på 0,99 på den tvingade meningen, men bara 3 % av svaren blir fel.
  Vid β 4 blir 32 % fel. Effekten följer biasens storlek och syns inte i A. Utan doskontroll (samma bias på *målet*) visar G2b′
  att en stor bias i mittlagren stör, inte att ett falskt schema styr.

## Redovisat (beskrivande)
| Mått | Värde |
| --- | --- |
| A på tvingad mening, lager 9–17 (styrda) | 0,18 → 1,00 (per konstruktion) |
| A på tvingad mening, lager 18–27 (ej styrda) | 0,17 → 0,19 (+0,03, z 4,1): **senare lager tar nästan inte efter** |
| Ablation: rätt svar, schemats egen S mot ingen styrning | 75,7 % mot 79,3 % (−3,7 p, z −1,5): **schemat hjälper inte**; när S pekar rätt −8 p (z −3,2, i efterhand) |
| Kontroll: egen S mot omflyttad S ("perm") | 75,7 % mot 80,0 % (−4,3 p, z −2,0) |
| A på målet med egen S, lager 9–17 / 18–27 | +0,25 (z 19,7) / −0,02 (z −2,7) |
| A_late på målet med tvingat S | −0,06 (z −6,3) |
| perm: A på målet, lager 9–17 | 0,44 → 0,14 utan att rätt-andelen sjunker (80 mot 79 %) |
| Fel utan styrning som egen S rättade | 20 av 62 (men 31 rätta blev fel) |

## Doskontroll (i efterhand, `sprint2_dose.json`, kördes efter resultatgranskningen)
Samma tvingade bias (β 4, 0,97) men **på målmeningen**, på samma 300 testtexter:
- Fel svar **36,7 %**, exakt samma som när schemat tvingas på fel mening (36,7 %). Mot ingen styrning 20,7 %: +16 p, z 5,3.
- Skillnad mellan tvingad fel och tvingad rätt: 0,0 p (z 0,0).
- A_late på målet sjunker även här (−0,09, z −7,9).

**Slutsats: G2b′ mäter bara att en stor bias stör. Åt vilket håll schemat pekar spelar ingen roll.**

## Tolkning
- **Krit. 2 (schema):** uppfyllt inom kända mallar. S är en förenklad, användbar modell av A, och den följer A även när A är fel.
- **Krit. 3 (styrning): inte visat.** G2b′ är formellt klarad, men doskontrollen visar att effekten är ren störning:
  en bias mot rätt mening skadar lika mycket som en bias mot fel mening.
  Schemat hjälper inte, och att förstärka rätt mening skadar (−8 p). Påverkan når knappt de senare lagren. Mittlagrens
  attention per mening är inte den variabel som styr svaret: perm flyttar den kraftigt utan effekt på svaret.
- Det här är en öppen-slinga-styrning på β = 4 i mittlagren. Hur styrningen kan göras verkningsfull är en designfråga
  för sprint 3, där rapporten (LoRA) och kontrollmodellen ska byggas. Där kan det vara bättre att låta en LoRA lära sig
  använda schemat än att lägga en fast bias.

## Grindkriterier
- [x] G2a klarad
- [x] G2b′ klarad
- [x] Ablationen redovisad (negativ)
- [x] Fristående granskning av koden (`sprint2_review.md`), åtgärdat utom läckage till nya ämnen (redovisat som begränsning)
- [x] Resultatgranskning (`sprint2_result_review.md`): bedömningen korrekt med anmärkningar, införda ovan
- [ ] Anders godkännande
