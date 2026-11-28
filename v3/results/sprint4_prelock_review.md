# Sprint 4 – granskning före låsning (del 2b: rapportmodell och K4a–c)

Fristående granskare, 28 nov 2026. Blind: inga test4-data finns i arbetsytan (bara `test4_done.txt`) och inga har hämtats.
Underlag: förregistreringen del 2b, `nexis3/report.py`, `report_train.py`, `report_eval.py`, `nexis3/kmap.py`,
`train_kmap.py`, `extract_v4.py`, `textgen.make_items_v4`, `tests/test_report.py`, `sprint4_f1c.json`, `sprint4_f2.json`.
Körda: `pytest tests` (97 godkända, 1 överhoppad). Beskrivande analys på **train4** (1 500 texter, 3 658 innehållstoken,
samma urval som `report_eval.test_tokens`, kartan `models/kmap.pt`), skript i `/tmp/rev/ana.py`, `/tmp/rev/dose.py`
(inte i repot). Ingen rapportmodell tränad eller körd.

## Slutsats: **ej låsbar**

Koden gör i stort sett vad förregistreringen säger, men tre av kriterierna mäter något annat än de ska, och det syns
redan på träningsdata: K4a sammanfaller till stor del med positionsvanan, K4c är nästan sant per konstruktion, och K4b
har en dos ungefär tre gånger så stor som skillnaden S–A och ligger utanför kartans fördelning. Dessutom finns en stark
textväg till A, och den gör tolkningsregeln ("kontrollen slår vanan → inre väg finns") missvisande.

## Nyckeltal från train4 (innehållstoken, beskrivande)

| Mått | Värde |
| --- | --- |
| argmax S = argmax A, alla | 60,5 % (observerade 100 %, oobserverade 50,1 %) |
| L1(S, A) vid observerade token | 0,0017 (kartans förstärkning k ≈ 1) |
| Andel observerade token i D (S ≠ A) / i E (S = A) | 0 % / 35 % |
| argmax S = mening 4 (vanan), i D | **69 %** (1 002 av 1 445) |
| argmax A = målmeningen (den som innehåller ordet som skrivs), alla / i D / i E | 70 % / **65 %** / 73 % |
| argmax S = målmeningen, i D | 9 % |
| D där A ≠ målmeningen | 506 token i 465 texter (i valideringsämnena 58) |
| S = A: kartans träningstexter / valideringsämnen / övriga uteslutna | 62,8 % / 58,7 % / 52,0 % |
| L1(S, manipulerad S) i D / L1(S, A) i D | **1,27 / 0,46** |
| Andel token med max S ≥ 0,85 | 0,08 % (99:e percentilen 0,78) |
| Manipulerad mening j = argmax A, i D / j = målmeningen, i D | 33 % / 31 % |
| Vanligaste argmax A (innehållstoken) | mening 4, 38 % (HABIT = 3 stämmer) |

## Anmärkningar

1. **Stor. K4a sammanfaller med positionsvanan.** I D pekar kartan på mening 4 i 69 % av fallen (kartan driver mot
   mening 4 mellan observationerna). En rapport som bara säger "4" (som R i sprint 1–2: "4" 74–88 %) får alltså
   K4a = P(R = S) − P(R = A) kraftigt positivt utan att läsa S. Tolkningsvillkoret jämför bara kontrollen med vanan,
   inte rapporten i D. *Förslag:* räkna K4a även med vanan som nollmodell (vanans K4a-värde i D, parat per text), kräv
   att schemamodellens K4a > vanans med > 2 SE, och redovisa K4a i delmängden argmax S ≠ 4 (huvudmått eller
   förregistrerad känslighetsanalys).

2. **Stor. Det finns en textväg till A som inte är introspektion, och den förvränger tolkningsreglerna.** Rapportmodellen
   ser texten och sitt kapade svar. Det sista ordet är svarsordet, som per konstruktion (`make_items_v4`) bara finns i
   målmeningen. "Den mening som innehåller ordet" är rätt om A i 70 % av innehållstoken, mot vanans 38 %. Följder:
   (a) Att kontrollen slår vanan visar sannolikt bara att Qwen kan slå upp ett ord, så tolkningsvillkoret "en inre väg
   finns" uppfylls trivialt. (b) I D är A = målmeningen i 65 % och S = målmeningen i 9 %. Där pekar texten alltså mot A.
   Eftersom facit = A belönar träningen att texten får gå före S, och om K4a faller kommer det att läsas som "rättar
   kartan med egen tillgång" fast det är ordmatchning. (c) Det enda stället där inre tillgång kan skiljas från texten är
   D med A ≠ målmeningen (≈ 35 % av D; på test4 grovt räknat 100–150 token, oklart hur många texter).
   *Förslag:* (i) förregistrera en textbaslinje: målmeningen för delen (eller den mening som har störst ordöverlapp med
   det senast skrivna ordet), redovisad i D, E och totalt. (ii) Byt villkoret "kontroll > vana" mot "kontroll > textbaslinje
   och > vana", gärna mot en extern kombination (logistisk regression på vana + textbaslinje, tränad på samma
   träningsexempel). (iii) Dela K4a och kontrollens träffar på A efter A = målmeningen / A ≠ målmeningen. Det AST-relevanta
   påståendet ("rapporten följer S trots att en inre väg finns") får bara göras i delmängden A ≠ målmeningen, och bara om
   kontrollen där slår textbaslinjen.

3. **Stor. K4a mäter inte det AST förutsäger på rätt sätt.** "Ingen inre tillgång" ger också R = S i D, och en
   regelbunden kombinerare som väger S mot textledtråden gör detsamma när S är pålitligare. *Förslag:* huvudjämförelsen i
   D bör vara inom samma token: P(R = A | D) för schemamodellen mot P(Rc = A | D) för kontrollen (sänker kartan träffarna
   jämfört med Qwen utan karta?), och i synnerhet andelen R = S bland D-token där kontrollen har rätt (Rc = A). Det är
   "schemat tar över en tillgänglig väg". Behåll gärna den nuvarande K4a som beskrivande mått.

4. **Stor. K4c är nästan sant per konstruktion.** E väljs på villkoret argmax S = argmax A. En rapport som bara läser
   kartans argmax träffar A till 100 % i E. 35 % av E är observerade token, där S = A exakt (L1 0,0017). Kontrollen och
   vanan har inget sådant urval. K4c visar alltså bara att rapporten läser S, vilket K4b redan prövar. *Förslag:* räkna
   K4c på alla innehållstoken (eller bara oobserverade): rapportens träffar på A mot kontrollen, mot vanan, mot
   textbaslinjen och mot argmax S. "Meningsfull rapport" bör betyda "bättre än textbaslinjen", annars är det ordmatchning.

5. **Stor. K4b: fel jämförelse, för stor dos och utanför fördelningen (doskontrollen, projektregeln).**
   (a) Mot kontrollen klarar varje modell som läser sin indata K4b, eftersom kontrollen aldrig ser S. Rätt kausal
   jämförelse är inom schemamodellen: andelen R = j med manipulerad S minus med sann S, parat per token.
   (b) Dosen: L1(S, Sm) ≈ 1,27 mot L1(S, A) ≈ 0,46 i D, alltså 2,8 gånger större. Max S ≥ 0,85 förekommer i 0,08 % av
   token. K4b kan klaras med en ingreppsstorlek som K4a aldrig erbjuder.
   (c) j = argmax A i 33 % och j = målmeningen i 31 % av D, så R = j blandar ihop kartan med text och attention.
   *Förslag:* en doskontrollerad manipulation, t.ex. S flyttad mot j tills argmax byts, med L1 dragen ur fördelningen för
   L1(S, A) i D (eller S lånad från en annan token med argmax j). Behåll 0,85 som beskrivande "stark dos". Dra j bland
   meningar ≠ argmax S, ≠ argmax A och ≠ målmeningen, eller redovisa per stratum. Kör doskontrollen (L1-fördelningar) på
   train4 och skriv in den i förregistreringen före låsning.

6. **Stor. K4a ska räknas på oobserverade token, och det ska stå så.** I D finns inga observerade token på train4 (k ≈ 1).
   Det sker automatiskt nu, men K4c, `rates` och `manip_follow` blandar in observerade token där S = A exakt.
   *Förslag:* förregistrera att alla K4-mått räknas på oobserverade innehållstoken, och redovisa observerade separat
   (`lag_part2` gör det inte heller, trots att prereg lovar "observerade och oobserverade var för sig").

7. **Liten–medel. Kartan är bättre på rapportens träningsdata än på test.** `report_train.examples` använder alla 1 500
   train4-texter. 938 av dem tränades kartan på (S = A 62,8 %), valideringsämnena ger 58,7 %, och test4 har nya ämnen
   (sprint 3: schemat generaliserade inte till nya ämnen). Rapporten lär sig därför lita på S mer än S förtjänar på test,
   vilket drar K4a uppåt. *Förslag:* korsanpassning (k-faldiga kartor, S till varje träningstext från en karta som inte sett
   den), eller åtminstone redovisa S = A på test4 mot på träningsexemplen.

8. **Liten. Fördelningen skiljer mellan träning och test.** Var tredje träningsexempel är en "annan" token (struktur,
   där A ofta pekar på lockmening eller sista meningen), medan test bara har innehållstoken. Test vägs dessutom mot långa
   ord (flera token). Det kan förstärka vanan. *Förslag:* redovisa rapportens fördelning över 1–4 i test, och överväg att
   väga per text i träningen som i testet.

9. **Medel. "Inre tillgång" är inte väldefinierad.** LoRA ligger på q_proj och v_proj i alla lager och är påslagen även
   över prefixet (text och kapat svar). Rapportmodellens aktiveringar vid token t är alltså inte de som gav A_t, utan en
   ny beräkning med ändrade vikter. *Förslag:* kör prefixet med LoRA avstängd (frusen modell, KV-cache) och slå på LoRA
   bara för de mjuka token och frågan. Då är prefixets tillstånd exakt det som mättes, och "inre väg" betyder läsning av
   just det. Kontrollera också på hans dator att `pre`-tokeniseringen av `"".join(answer_tokens[:t+1])` ger samma id som
   de genererade (avkodning token för token och omtokenisering kan skilja, och assistentvändan stängs med `<|im_end|>`).

10. **Medel. Låsningen täcker inte allt som bestämmer facit och prompten.** `LOCKED` saknar `nexis3/compress.py`
    (`caricature` = definitionen av A, `load`), `nexis3/textgen.py` (prompt, `make_items_v4`, `SYSTEM_PROMPT`),
    `nexis3/topics_v3.py`, `nexis3/attention.py` och `extract_tokens.py`. Rapportmodellerna hashas först vid körning, så
    inget hindrar omträning eller val mellan flera tränade modeller. *Förslag:* lägg till filerna. Träna båda modellerna
    före låsning (de rör inte test4) och lås deras sha256 och träningsloggar, eller gör ett andra låssteg `lock-models`
    som `run` kräver. Lås också `HABIT` som beräknat värde (räknas ur train4 och skrivs i låsfilen; nu hårdkodat, men det
    stämmer: mening 4, 38 %).

11. **Liten. bf16-risk.** `add_lora` skapar LoRA-parametrarna i basmodellens dtype. Om `load_model` väljer bfloat16
    (RAM < 14 GB) uppdateras A-matriserna med AdamW i bf16 vid lr 1e-4, och små steg kan avrundas bort. *Förslag:* håll
    LoRA och projektor i float32 (kasta i forward), eller kräv float32 och skriv dtype i modellfilen och resultatet.

12. **Liten. Statistik.** Parning per text med medel per text inom delmängden och SE över texter är rimligt. Texter utan
    D-token faller bort i K4a (rätt, men redovisa n texter, inte bara n token). Med sex jämförelser bör kraven "> 2 SE"
    stå per test, och grind 4 = alla tre, utan sammanvägning i efterhand.

## Kodkontroll (punkt 6 i uppdraget): inga egentliga fel

- `report_train.examples` och `report_eval.test_tokens` bygger prompten identiskt: `SYSTEM_PROMPT`, `it.prompt` (som
  finns på `Item4`) och `"".join(answer_tokens[:t+1])`, med samma `Reporter.logits`. Uppslaget av items via `SPLITS` och
  `make_items_v4(n, s0, pool)` med `it.idx` = postens `idx` stämmer. `content_rows` används likadant (test = unionen av
  båda delarna, sorterad).
- Index och argmax är rätt: `aS = S[t].argmax()`, `aA = H[t].argmax()`, där H = `seq(r)[0]` = samma karikatyr som facit
  i träningen. `j` dras med frö 4242 i deterministisk ordning (sorterade filer). `paired` och `cmp` är rätt.
- Låsning: `lock` vägrar om låsfilen finns, `run` kontrollerar sha256 och vägrar om resultatfilen finns (hederssystem: går
  att kringgå genom att radera den, men det är acceptabelt). `run` måste köras från `v3/`.
- Mindre: `lag_part2` delar inte på observerade och oobserverade token (se anm. 6). Rapportmodellernas hash skrivs bara
  i resultatet (anm. 10).

## Det som är bra

- Facit = faktisk A och en kontroll med samma exempel och ordning, där bara S skiljer, är rätt grundupplägg. Kontrollen
  utan karta förhindrar att "rapporten följer S" tas som AST-stöd när det inte finns någon annan väg. Den tankegången är
  redan skriven i förväg.
- Kartan ser aldrig texten. F2 klarades mot tre nollmodeller på undanhållna ämnen, och observationen var fjärde token
  ger verkliga S ≠ A-fall (40 % av innehållstoken).
- Basvikterna är frysta och kontrolleras med checksumma, LoRA är identitet vid start eller avstängd (testat), och test4
  har inte rörts i arbetsytan.
- Allt redovisas parat per text och resultatet skrivs en gång.

## Minsta ändringar för "låsbar"

1. K4a: vanan som nollmodell, delmängden argmax S ≠ 4, och jämförelsen inom D mellan schemamodell och kontroll (anm. 1, 3).
2. Textbaslinje förregistrerad. Villkoret för inre väg = kontroll > textbaslinje. AST-tolkningen bara i A ≠ målmeningen
   (anm. 2).
3. K4c över alla oobserverade innehållstoken mot kontroll, vana och textbaslinje (anm. 4, 6).
4. K4b inom schemamodellen (manipulerad mot sann S), doskontrollerad manipulation, j skild från A och målmeningen.
   Doskontrollen körd på train4 och införd (anm. 5).
5. Låslistan kompletterad och modellerna låsta före test (anm. 10).

Anm. 7 (korsanpassning), 9 (LoRA bara efter prefixet) och 11 (float32) rekommenderas men kan ersättas av redovisning.

---

# Andra omgången (28 nov 2026, efter commit 7d3f355)

Granskat: "Del 2b, reviderad" i `sprint4_prereg.md`, nya `report_eval.py`, `kmap_crossfit.py`, `models/kmap_fold{0,1}.pt`,
`nexis3/report.py`, `report_train.py`, `tests/test_report.py`, `tests/test_report_eval.py`. Fortfarande blint: inga
test4-data i arbetsytan. `pytest tests`: 99 godkända, 1 överhoppad. Analys på train4 med de korsanpassade kartorna
(S för varje text från foldkartan som inte sett den), samma urval som nya `test_tokens`: 2 894 oobserverade
innehållstoken i 1 266 texter, D = 1 621 token i 992 texter. Skript `/tmp/rev/r2.py` (inte i repot). Jag körde
`report_eval.summarize` på syntetiska rapporter byggda av dessa train4-rader. Ingen Qwen-körning.

## Slutsats: **ej låsbar** (men nära; tre små ändringar räcker)

Åtgärderna från första omgången är i huvudsak rätt gjorda, och koden är korrekt. Två fel finns kvar. Den förregistrerade
tolkningen av den inre vägen klaras per konstruktion (anm. R2-2). K4a klaras om schemamodellen bara har mer vana än
kontrollen (anm. R2-1). En tredje sak bör skrivas in: en rapport utan introspektion, som bara väger S mot text och
position, klarar alla tre kriterierna (anm. R2-3).

## Nyckeltal (train4, oobserverade innehållstoken, korsanpassad S)

| Mått | Värde |
| --- | --- |
| S = A / vana = A / text (T) = A | **44 %** / 38 % / 65 % |
| I D: S = 4 / S = T / A = T | 71 % / 8 % / 68 % |
| Dos i D: L1 vid byte topp↔j (medel, kvartiler) | 0,32 (0,21–0,43) |
| L1(S, A) i D (medel, kvartiler) | 0,48 (0,26–0,70) |
| S-fördelning foldkartor mot `kmap.pt` på valideringsämnen: argmax = 4 / median max S | 0,69 mot 0,70 / 0,36 mot 0,36 |
| D med S ≠ 4 | 468 token i 378 texter (på test4 grovt 90 token) |
| D med A ≠ T | 518 token i 478 texter |

## Syntetiska rapporter genom `summarize` (train4-rader, 992 D-texter; på test4, ca 200 D-texter, blir z ungefär hälften)

| Schemamodell R | Kontroll Rc | K4a | K4b | K4c |
| --- | --- | --- | --- | --- |
| alltid 4 | T | **+1,15 (z 42)** | 0 | 0 |
| T, men 4 i 30 % | T | **+0,34 (z 17)** | 0 | +0,18 (z 14) |
| följer S | alltid 4 | +0,43 (z 21) | – | +0,05 (z 5,6) |
| logistisk kombinerare: log S + T + positionsbias per del, tränad på A (korsvis) | samma utan S | **+0,09 (z 4,4)** | **+0,05 (z 11)** | **+0,32 (z 21)** |

Kombineraren: andel R = S där kontrollen har rätt 10 %. Kontrollen mot T totalt +0,01 (z 1,3), mot T i D där A ≠ T
**+0,43 (z 19,5)**.

## Anmärkningar

R2-1. **Stor. K4a klaras av vanskillnad mellan modellerna.** Kontrasten tar bort vanan bara om båda modellerna har lika
mycket vana. I D är S = 4 i 71 % av fallen, så en schemamodell som säger "4" oftare än kontrollen får stort K4a utan att
läsa S: 30 % extra vana ger z 17 på train4-skala. K4b skyddar grinden (den kräver kausal effekt av S), men K4a kan då
inte tolkas som "rapporten följer kartan där den skiljer sig från A". *Förslag:* gör nollmodellen till samma
schemamodell med S utbytt mot S från en slumpad annan token i test4 (i fördelningen, frö låst):
[1(R(S) = S) − 1(R(S) = A)] − [1(R(S_donor) = S) − 1(R(S_donor) = A)]. Vana och textväg tar då ut varandra exakt, eftersom
det är samma modell. Behåll kontrastmot kontrollmodellen som redovisning, och redovisa K4a i D med S ≠ 4 samt andelen
R = 4 för båda modellerna. (Ett donor-S med samma argmax som S bör dras om; annars blir kontrasten 0 per token.)

R2-2. **Stor. Tolkningsvillkoret för inre väg klaras per konstruktion.** `control_vs_text_Dsub` jämför kontrollen med T i
D där A ≠ T. Där är T = A noll per definition, så varje kontroll som någon gång träffar A "slår texten". Vanan ensam ger
redan +0,2–0,4, och kombineraren +0,43 (z 19,5). Formuleringen "klarar kontrollen inte T där A ≠ T finns ingen påvisad
inre väg" leder alltså alltid till slutsatsen "inre väg finns". *Förslag:* den inre vägen prövas mot en extern
kombinerare utan Qwen: logistisk regression på T och positionsbias (per del), tränad på samma träningsexempel ur train4,
parametrarna låsta före test. Kontrollen ska slå den, totalt och i D där A ≠ T, > 2 SE. Kombineraren kan räknas på test4
utan Qwen och hör hemma i `report_eval.py` och i låsningen.

R2-3. **Stor (tolkning, inte kod). Grinden skiljer inte AST från rationell vägning av ledtrådar.** En kombinerare som
väger S, T och position, tränad med facit A och utan inre tillgång, klarar K4a, K4b och K4c på train4. Det är väntat:
när S bär information utöver text och vana är det rätt att delvis följa S. Att K4a–c klaras visar alltså att rapporten
använder kartan som en ledtråd bland flera. Det var det F2 och rapportträningen gjorde möjligt, och det är inte unikt
för AST. *Förslag:* skriv det i tolkningen. Räkna samma kombinerare med S (låst, ur train4) på test4 och redovisa dess
K4a, K4b och andel R = S där kontrollen har rätt bredvid schemamodellens. Det AST-specifika påståendet blir då
"schemamodellen följer S mer än en rationell kombinerare gör, och mer än kontrollen har inre väg för (R2-2)". Det kan
redovisas utan att vara grindkriterium.

R2-4. **Medel. K4c är oinformativt, och motiveringen använder fel siffra.** På de token som räknas är S = A 44 %, inte
61–62 % (det är över alla oobserverade token, `sprint4_kmap_crossfit.json`). K4c mot vanan klaras av kontrollen via
texten och av en ren S-följare (z 5,6 på train4, grovt 2,5 på test4). Ett krav som alla rimliga rapporter klarar säger
inget om schemat. *Förslag:* rätta siffran i prereg. Kalla K4c förutsättning ("rapporten är inte degenererad"), inte
belägg, eller ersätt med R = A för schemamodellen mot R = A för samma modell med donor-S (som R2-1). Det mäter om kartan
gör rapporten sannare.

R2-5. **Bra, med en liten anmärkning: K4b och doskontrollen.** Jämförelsen inom modellen och bytet topp↔j är rätt.
Bytet bevarar S-värdenas fördelning, så det ligger i fördelningen. Dosen är 0,32 mot 0,48 i D (medel), i samma
storleksordning och snarare konservativ. Att j utesluter S, A och T tar bort sammanblandningen. Liten anmärkning:
prereg anger 0,36 (alla token) mot 0,48 (D); ange båda i samma delmängd. Redovisa också dosfördelningen (kvartiler),
inte bara medel, så att projektregeln går att kontrollera.

R2-6. **Bra: KV-cache-vägen stämmer.** Prefixet körs med `input_ids` och standardpositionerna 0…len(pre)−1. Andra
anropet med `past_key_values` och `inputs_embeds` får `cache_position` och positionerna len(pre)… från cachelängden, och
masken av ettor har rätt längd n = len(pre) + len(tail). Testet visar att cachevägen med LoRA avstängd ger samma logits
som en hel körning (atol 1e-4), och att gradient når LoRA och projektorn. `DynamicCache` muteras av andra anropet
(update lägger till svansens nycklar), men en ny cache byggs i varje `logits`-anrop, så inget läcker mellan R, Rm och
olika token. Kostnaden är att prefixet räknas två gånger per token i schemavillkoret. Det är ofarligt, men cachen kan
återanvändas med `copy.deepcopy` om tiden blir ett problem. LoRA och projektorn är i float32. Kvar från första omgången
(liten): kontrollera på Anders dator att omtokeniseringen av `"".join(answer_tokens[:t+1])` ger de genererade id:na,
till exempel med en kontroll på ett par texter i `report_train.py time`.

R2-7. **Bra: korsanpassningen fungerar.** Foldkartornas S liknar `kmap.pt`:s S på valideringsämnen (argmax = 4: 0,69 mot
0,70; median max 0,36 mot 0,36; S = A 47 % mot 49 % på valideringsämnena). Kvarvarande skillnad: test4 har nya ämnen.
Redovisa S = A på test4 bredvid träningsvärdet.

R2-8. **Liten. Låsning.** Låslistan är nu komplett för det som bestämmer A, prompten och modellerna, och `run` kräver att
listan stämmer. Två saker saknas. `extract_tokens.py` och `nexis3/segment.py` bestämmer inte test4 i efterhand, så det är
ingen brist. Däremot finns inget som visar hur många gånger rapportmodellerna tränats före låsning. *Förslag:* committa
träningsloggarna (förlustkurvor, `base_weights_unchanged`) tillsammans med låsfilen. Lägg kombinerarens parametrar
(R2-2, R2-3) och donorfröet (R2-1) i låsningen. `HABIT = 3` stämmer på de token som räknas (mening 4, 38 %).

## Minsta ändringar för "låsbar"

1. K4a med donor-S i samma schemamodell som nollmodell. Kontrollkontrasten redovisas (R2-1).
2. Villkoret för inre väg mot en extern T + position-kombinerare, låst ur train4, inte mot T i A ≠ T (R2-2).
3. Tolkningen kompletteras: grinden visar användning av S som ledtråd. Kombineraren med S redovisas som rationell referens
   (R2-3). Siffran 61 % rättas och K4c omformuleras (R2-4).
