# Grind 1 – Nexis v3 sprint 1 (24 nov 2026)

Modell: Qwen/Qwen2.5-1.5B-Instruct, float32, CPU (Anders laptop, Windows), torch 2.15.0+cpu, transformers 5.24.0.
A = attention per mening, lager 9–17 av 28, under stegen som väljer svarets tokens. Engelska testtexter, frö 0, n = 100.
Målposition balanserad (25 per plats). Körtid 15,9 min. Vikter orörda (sha256 före = efter).

## Baslinje (`baseline_s0.json`)

| Mått | Andel | 95 % (Wilson) |
| --- | --- | --- |
| Uppgiften rätt besvarad | 100/100 | 0,96–1,00 |
| argmax A = målmening | 98/100 | 0,93–0,99 |
| argmax A_mean (per token) = mål | 98/100 | 0,93–0,99 |
| argmax A_first (första svarssteget) = mål | 45/100 | 0,36–0,55 |
| Rapport R = mål | 41/100 | 0,32–0,51 |
| R = argmax A | 43/100 | 0,34–0,53 |
| R ej tolkbar | 0/100 | 0,00–0,04 |

- A på målmeningen i medel: 0,571 ± 0,011 (SE). Andel av all attention som går till de fyra meningarna: 0,149 ± 0,003.
- **Rapporten är ett positionsbeteende, inte en rapport.** Modellen svarar "4" i 74 fall och "1" i 26, aldrig "2" eller "3".
  R = mål är 100 % när målet står sist, 64 % när det står först och 0 % på plats 2 och 3.
- argmax A ≠ mål i bara 2 fall, för få för att säga något om hur R följer A.
- Invarianttesterna på den riktiga modellen: 48 klarade, 1 överhoppat (I3-mutanten, körs bara på testmodellen),
  inklusive den positiva kontrollen (`tests_real_model.txt`).

## Tolkning

- A pekar ut rätt mening nästan alltid, men frågan delar ord med målmeningen (granskningen, punkt 2). Det visar
  att mätningen fungerar, inte att den fångar något mer än ordmatchning.
- A_first (45 %) visar att första svarssteget ofta tittar någon annanstans. Valet av A-variant låses före kärntestet.
- Nollpunkten för kärntestet: basmodellen kan inte rapportera var den fokuserar. R säger "sista meningen" eller
  "första meningen" oavsett vad den faktiskt gjorde. En schemabaserad rapport har alltså allt att vinna, men
  kontrollmodellen (LoRA utan schema) kommer att lära sig svarsformatet, och det är skillnaden mot den som räknas.

## Grindkriterier

- [x] Baslinjen redovisad med antal och spridning.
- [x] Fristående granskning av koden mot invarianterna (`sprint1_review.md`, `sprint1_review2.md`), åtgärdat utom punkt 2.
- [x] Chatten fungerar och visar attention per mening. Anders skärminspelning 24 nov: 17 meddelanden på svenska och engelska,
  attention per mening visas för varje svar.

## Iakttagelser från chatten (beskrivande)

- Senaste meddelandet får mest attention ("ok tack" 19,7 %), och därefter går den till nyare meningar. Attention är alltså
  starkt präglad av närhet.
- På frågan "vet du att du är en del i ett test jag gör tillsammans med Claude?" svarar modellen "Ja, jag vet att jag är en del av
  ett test …". Den bekräftar det som står i frågan, och det är just den svaghet som kontrollmodellen i kärntestet ska fånga.
- Meningsindelningen delar "(Hello! Hello! Hello!)" i tre meningar med parenteserna på var sin ända. Det är korrekt enligt
  reglerna men ser udda ut.

**Grind 1 godkänd av Anders 24 nov 2026 12:07 ("yes").**
