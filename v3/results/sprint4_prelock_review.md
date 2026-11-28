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
