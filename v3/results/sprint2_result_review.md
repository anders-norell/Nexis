# Resultatgranskning sprint 2 (Nexis v3), 24 nov 2026

Fristående granskare. Underlag: `sprint2_prereg.md`, `sprint2_gate.md`, `sprint2_steer_test.jsonl` (300 rader),
`sprint2_steer.json`, `analyze_steer.py`, `sprint2_schema.json`, `sprint2_beta*.jsonl`, `sprint2_beta_choice.json`,
`steer_eval.py`, `nexis3/`. Inga befintliga filer har ändrats. Alla siffror nedan är omräknade av mig ur rådata
(egen kod, inte `analyze_steer.py`). Analyser märkta *post hoc* är mina och ingår inte i något låst kriterium.

## Dom: bedömningen är korrekt med anmärkningar

Siffrorna stämmer och förregistreringen följdes. G2b′ är klarad så som kriteriet var formulerat. Men tolkningen i
gate-rapporten håller bara delvis. Data visar att svaret störs av **biasens storlek** och inte av att schemat pekar
fel. Det mittlagrens attention mäter (A) förklarar inte effekten. Det kan G2b′ inte skilja på, eftersom en
doskontroll saknas. Tre av de fyra tolkningspåståendena behöver skrivas om (se 3).

## 1. Siffrorna

Alla siffror i gate-rapporten återskapas exakt ur `sprint2_steer_test.jsonl` (300 unika texter, β = 4 på alla rader):

| Mått | Gate | Omräknat |
| --- | --- | --- |
| Fel, tvingat mot bas | 36,7 % mot 20,7 %, +16 p, SE 3,1, z 5,2 | 0,367 / 0,207, +0,160, SE 0,0307, z 5,22 |
| Värdet från den tvingade meningen nämns (fyllnad) | 0 av 193 | 0 av 193 |
| A tvingad, lager 9–17 | 0,18 → 1,00 | 0,176 → 1,000 |
| A_late tvingad, lager 18–27 | +0,03, z 4,1 | +0,027, z 4,11 |
| Egen S mot bas (rätt) | 75,7 mot 79,3, z −1,5 | −0,037, z −1,54 |
| Egen S mot perm | 75,7 mot 80,0, z −2,0 | −0,043, z −2,04 |
| A på målet med egen S, mitt/sent | +0,25 (z 19,7) / −0,02 (z −2,7) | +0,253 (19,67) / −0,022 (−2,67) |
| Fel vid bas som egen S rättade | 20 av 62, 31 rätt blev fel | 20 av 62, 31 av 238 |

β-tabellen (`sprint2_beta_choice.json`) återskapas också ur `sprint2_beta.jsonl` och `sprint2_beta_base.jsonl`:
40 texter (1000–1039), 34 rätt vid β = 0. Andelen som blir fel med tvingat S är 1/34 vid β 0,5, 1 och 2, och 11/34 = 32 %
vid β 4. Villkor (b) är uppfyllt för alla β. Den tvingade meningen är densamma för en text vid alla β.

**Bedömningen av svaren.** `answer_correct` söker målvärdet med ordgräns. Jag har läst alla felbedömda svar i de fyra
villkoren (62/73/60/110) och de rätta svar som innehåller "not/no":
- Falskt fel: "Birches" (egen S, mål birch) och "Bagel" (tvingat, mål bagels). Singular och plural känns inte igen.
- Falskt rätt: "Peugeot is not a brand name." (egen S) och "Volvo is not an answer to the question." (tvingat).
- Avhuggna svar (max 24 tokens): 23 av de tvingade felsvaren har minst 14 ord. Jag hittade inget där målvärdet
  verkar vara på väg. De flesta är "not provided", omskrivna frågor eller upprepade lockmeningar.
- Med rättad bedömning blir alla nyckeltal oförändrade på tre decimaler (G2b′ z 5,22, egen − bas z −1,54, egen − perm z −2,04).
Bedömningen är alltså inte källan till något resultat. Långa svar bedöms rätt när de innehåller värdet
("The young musician in the family performs on the clarinet." räknas som rätt, och det stämmer).

## 2. Förregistreringen

- **β-valet:** regeln (a′) + (b) tillämpad rätt. β = 4 är det minsta β som uppfyller båda villkoren.
  `steer_eval.py test` vägrar starta utan `models/sprint2_beta_choice.json`. Filtiderna stämmer: förregistreringens
  tillägg 14:52, β-filer 15:55, testfil 17:43. `steer.log` visar valideringsfasen (160 rader) och sedan testfasen.
- **Testtexterna användes inte vid valet:** valideringen gjordes på frö 2000, texterna 1000–1039, och testet på
  frö 3000. *Anmärkning:* förregistreringen kallar 1000–1039 "ej använda för schemats vikter". Det stämmer, men de
  ingår i de 200 valideringstexter som schemats hyperparametrar valdes på (`train_schema.py`, rad 4). Det påverkar inte
  testtexternas oberoende men bör stå så.
- **G2b′ bedömd som formulerad:** andel fel tvingat minus bas, parat SE, > 2 SE. Det som skulle redovisas finns med:
  om värdet nämns (bara fyllnadsmeningar) och A_late. perm-kontrollen finns med (beskrivande).
- Frö 4000 för den tvingade meningen och 1–3 steg för perm följer planen.

## 3. Håller tolkningen?

### 3a. "Störning snarare än styrning": riktigt som beskrivning, men förklaringen ("förlorar informationen") har inte stöd

Tre fynd talar emot att det är förlorad attention mot målet i mittlagren som gör svaret fel:

1. **Mellan β 2 och β 4 ändras A nästan inte, men felandelen ökar tiofalt.** I valideringen är A på den tvingade
   meningen 0,994 vid β 2 och 1,000 vid β 4, och A på målet 0,006 och 0,000. Ändå blir 3 % fel vid β 2 och 32 % vid β 4.
   Effekten på svaret går alltså inte via fördelningen över meningar som A mäter. Den följer storleken på biasen:
   med S = 0,97/0,01 får den tvingade meningen +2,7 vid β 2 och +5,4 vid β 4, och övriga meningar −6,4 och −12,9 logit.
   Trolig mekanism [granskarens hypotes]: en stor positiv bias drar massa från tokens utanför meningarna
   (systemprompten, frågan, den första token som ofta tar emot mycket attention). A normeras över de fyra meningarna
   och kan inte se det. `attention.py` räknar ut `mass` (meningarnas andel av all attention), men den sparas inte i
   styrningsraderna.
2. **perm tar bort attention från målet utan att svaret blir sämre.** Med perm sjunker A på målet i mittlagren från
   0,435 till 0,139. Rätt svar: 80,0 % mot 79,3 % utan styrning. I de 28 texter där perm ger målet en bias mellan −8 och
   −4 (A på målet 0,015) blir 89 % rätt, mot 75 % utan styrning (post hoc). Svaret klarar sig alltså nästan utan
   mittlagrens attention mot målet. Svaret hämtas troligen via lager 0–8 och 18–27 och via promptens egna positioner,
   som inte styrs (biasen läggs bara från sista prompttoken och framåt).
3. **Att peka rätt skadar också (post hoc).** I de 219 texter där schemats egen S har sin topp på målet sjunker
   andelen rätt med egen S från 90,9 % till 82,6 % (parat −8,2 p, z −3,15). Biasen mot målet är där i median +2,3.

Det finns också en effekt av innehållet, men den gäller lockmeningen. Att tvinga mot lockmeningen ger −26 p
(n 107, z −5,0) och mot en fyllnadsmening −10 p (n 193, z −2,8). Där lockmeningen tvingas upprepar svaret den ofta.
G2b′:s effekt kommer alltså till stor del från de 107 texter där lockmeningen tvingades. Det står inte i rapporten.

**Följd:** G2b′ visar att *en stor bias i mittlagren gör svaret fel*. Det visar inte att *ett schema som pekar fel
gör svaret fel*. För att skilja de två åt behövs en doskontroll med samma biasstorlek och 0,97 på **målet**
(eller jämnt på alla meningar). Det är samma lärdom som i CLAUDE.md ("Doskontroll ... grind 4"). perm är ingen
doskontroll, eftersom dess största bias ligger kring +2,6 och den tvingade är +5,4/−12,9.

### 3b. "Schemat hjälper inte": riktigt, men för milt

Egen S mot bas är −3,7 p (z −1,5), alltså inget stöd för att schemat hjälper. Att innehållet spelar roll åt fel håll
(perm är *bättre* än egen S, z −2,0) står i tabellen men tolkas inte. Tillsammans med 3a.3 tyder det på att det skadar
att förstärka det schemat förutsäger, oftast målet. Mer korrekt vore: "styrning i den här formen hjälper inte och
skadar när den förstärker rätt mening". Det är post hoc och bör märkas så.

### 3c. "Senare lager tar nästan inte efter": riktigt som beskrivning

A_late på den tvingade meningen ökar bara med +0,027. Rapporten saknar två saker. (i) A_late på **målet** sjunker
med det tvingade villkoret, från 0,476 till 0,416 (−0,061, z −6,3, post hoc), och i valideringen bara vid β 4
(0,49 → 0,40). Lager 18–27 påverkas alltså, men genom att tappa målet och inte genom att följa den tvingade meningen.
(ii) A mäts över svarets positioner, och svaren är olika långa i villkoren (tvingat i medel 7,3 ord, bas 4,4).
Jämförelserna av A blandar därför olika positioner.

### 3d. "Modellen läser inte den tvingade meningen": inte belagt

- Fyllnadsmeningarna kommer från **andra ämnen**, och deras värde har en annan typ än det frågan efterfrågar
  (frågan gäller ett instrument, fyllnadsmeningen ett hundnamn). Att svaret aldrig innehåller fyllnadsvärdet (0/193)
  väntas lika mycket om modellen läser meningen och förkastar den som irrelevant. Måttet kan inte skilja de två fallen.
- I mittlagren *läser* svarspositionerna per konstruktion bara den tvingade meningen (A = 1,00). Vad modellen sedan gör
  med det syns inte i svarets värde.
- Det finns spår av fyllnadsmeningarna i svaren: "Maria visited the city hospital." (en fyllnadsmening om jobb som
  tvingades, mål Munich) och "...put in the ground ... is a dog" (två svar). Ord som bara finns i den tvingade meningen
  förekommer i 33 tvingade svar mot 14 vid bas, men de flesta gäller fall där lockmeningen tvingades.
- Att svaret faller tillbaka på lockmeningen (29 av 63 felsvar där en fyllnadsmening tvingades) är anmärkningsvärt,
  eftersom lockmeningen också trycks ner med −12,9 i mittlagren. Det stöder att svaret hämtas via vägar som styrningen
  inte når.

Föreslagen formulering: "Svaret tar inte över den tvingade fyllnadsmeningens värde, men eftersom värdet har fel typ
för frågan säger det lite om huruvida meningen läses."

## 4. Det som saknas och måste redovisas

1. G2b′ uppdelad på tvingad lockmening (−26 p) och tvingad fyllnadsmening (−10 p).
2. β-språnget: A är mättad redan vid β 2, men svaret ändras först vid β 4. Effekten följer biasens storlek, inte A.
3. Att ingen doskontroll finns (0,97 på målet eller jämn bias med samma storlek vid β 4). Utan den kan G2b′ inte tolkas
   som en effekt av att schemat pekar fel. Bör köras innan styrresultatet används som grund i sprint 3, eller redovisas
   som begränsning.
4. perm: målets attention i mittlagren sjunker till 0,14 utan att svaret blir sämre. Det motsäger förklaringen
   "förlorar informationen".
5. Att `mass` (meningarnas andel av all attention) inte sparas. Den skulle visa förskjutningen från tokens utanför
   meningarna.
6. Att valideringstexterna för β ingick i hyperparameterurvalet för schemat (rätta ordalydelsen i förregistreringen).
7. Bedömningen av svaren: 2 falskt fel (singular/plural) och 2 falskt rätt ("X is not ..."). Utan betydelse för
   resultatet, men bör nämnas.
8. A_late på målet sjunker med tvingat S (−0,06, z −6,3).

## Kort lista

- Siffror: alla rätt, återskapade ur rådata. Svarsbedömningen har 4 fel av 1 200 och påverkar ingenting.
- Förregistreringen: följd (β-regeln, testtexterna orörda före valet, G2b′ som formulerat). Liten ordalydelse-
  anmärkning om valideringstexterna.
- G2b′: formellt klarad, men visar att en stor bias stör, inte att ett falskt schema styr. Doskontroll saknas.
- "Störning": stämmer som beskrivning. "Förlorar informationen" motsägs av perm och av β 2 mot 4.
- "Schemat hjälper inte": stämmer, men för milt. Det skadar att förstärka rätt mening (post hoc).
- "Senare lager tar nästan inte efter": stämmer. A_late på målet sjunker, och det bör nämnas.
- "Läser inte den tvingade meningen": inte belagt. Fyllnadsvärdet har fel typ för frågan.
- Effekten drivs till stor del av texterna där lockmeningen tvingades. Redovisa uppdelningen.
