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
- Svaret använder aldrig den tvingade meningens värde (0 av 193 fyllnadsmeningar). Modellen tappar målet och faller
  tillbaka på lockmeningen eller hittar på ("The antique volume was produced in 1850"). Den läser alltså inte den
  mening schemat pekar på.

## Redovisat (beskrivande)
| Mått | Värde |
| --- | --- |
| A på tvingad mening, lager 9–17 (styrda) | 0,18 → 1,00 (per konstruktion) |
| A på tvingad mening, lager 18–27 (ej styrda) | 0,17 → 0,19 (+0,03, z 4,1): **senare lager tar nästan inte efter** |
| Ablation: rätt svar, schemats egen S mot ingen styrning | 75,7 % mot 79,3 % (−3,7 p, z −1,5): **schemat hjälper inte** |
| Kontroll: egen S mot omflyttad S ("perm") | 75,7 % mot 80,0 % (−4,3 p, z −2,0) |
| A på målet med egen S, lager 9–17 / 18–27 | +0,25 (z 19,7) / −0,02 (z −2,7) |
| Fel utan styrning som egen S rättade | 20 av 62 (men 31 rätta blev fel) |

## Tolkning
- **Krit. 2 (schema):** uppfyllt inom kända mallar. S är en förenklad, användbar modell av A, och den följer A även när A är fel.
- **Krit. 3 (styrning):** delvis uppfyllt. Styrning via schemat påverkar beteendet: ett falskt schema stör svaret
  (G2b′). Men schemat *hjälper* inte (ablationen), och påverkan når knappt de senare lagren. Effekten är störning
  snarare än styrning: när mittlagrens attention flyttas förlorar modellen informationen, men den hämtar inte in
  den nya meningen.
- Det här är en öppen-slinga-styrning på β = 4 i mittlagren. Hur styrningen kan göras verkningsfull är en designfråga
  för sprint 3, där rapporten (LoRA) och kontrollmodellen ska byggas. Där kan det vara bättre att låta en LoRA lära sig
  använda schemat än att lägga en fast bias.

## Grindkriterier
- [x] G2a klarad
- [x] G2b′ klarad
- [x] Ablationen redovisad (negativ)
- [x] Fristående granskning av koden (`sprint2_review.md`), åtgärdat utom läckage till nya ämnen (redovisat som begränsning)
- [ ] Resultatgranskning (`sprint2_result_review.md`)
- [ ] Anders godkännande
