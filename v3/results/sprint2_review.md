# Sprint 2 – fristående granskning (24 nov 2026)

Granskare: separat agent utan bygghistorik. Granskat: `nexis3/schema.py`, `train_schema.py`, `extract.py`,
`nexis3/steer.py`, `steer_eval.py`, `beta_choice.py`, `nexis3/attention.py` (A och tillstånd), `nexis3/textgen.py`
(`make_items_v2`), `results/sprint2_prereg.md`, `results/sprint2_schema.json`, testerna. Schemadatan i
`/mnt/user-data/uploads/nexis3/results/schema_data` (1 200 + 300 poster) analyserades direkt. Inga befintliga filer
ändrade. Egna analysskript låg utanför repot (/tmp) och är inte incheckade.

Tester: `python3 -m pytest -q` → **68 passed, 1 skipped** (liten slumpmodell, transformers 5.24.0, torch 2.15).

## Dom: **godkänd med anmärkningar**

Koden gör det den säger. Inget läckage hittat, G2a återskapas exakt (0,8433 ur `models/schema_s.pt`) och
styrningen läggs på rätt rader, kolumner och lager, även med KV-cache och i bf16. Men två saker måste hanteras
**innan `steer_eval.py test` körs**, annars blir grind 2 tom:

1. G2b och villkor (a) i β-regeln är sanna per konstruktion (anmärkning 1).
2. Schemat generaliserar inte till ämnen det inte tränats på (anmärkning 2). G2a står som förregistrerat, men
   påståendet "S förutsäger A" gäller bara de tolv mallarna.

---

## 1. Läckage, validering och triviala vägar till G2a

**Inget läckage.**
- Schemats indata är dolda tillstånd bara i promptens positioner (`attention.py:216`, `[0, :n_p]`). Modellen är
  kausal, så svaret påverkar dem inte. `test_extract.py::test_states_independent_of_answer` prövar det.
- Inga fält som `target`, `lure`, `answer` eller `task_correct` används som indata (`schema.py:24-30`).
- Normeringen (mu/sd) räknas bara på träningsdelen (`schema.py:54-56`).
- Testdelen (frö 3000) används först efter valet av hyperparametrar (`train_schema.py:64-74`).
- Posterna i datan stämmer med `make_items_v2` (idx, target och lure kontrollerade för alla 1 500). Inga
  gemensamma prompter mellan träning och test. Exakt en testtext har samma fyra meningar som en träningstext i
  annan ordning, vilket saknar betydelse.

**Valideringen är ren men något optimistisk.** Samma 200 valideringstexter (1000–1199) används både för tidig
stopp och för val av hyperparametrar (`train_schema.py:68-73`). Valideringssiffran blir därför för hög (87,5 %
mot 84,3 % på test). Testet påverkas inte. β-valets 40 texter (1000–1039) ligger inom valideringsdelen. De har
inte påverkat vikterna, men de har påverkat tidig stopp. Det är en liten sak.

**Kan G2a klaras trivialt? Delvis. Nollmodellerna i förregistreringen är svaga.**

| Prediktor på testtexterna (n = 300) | argmax = argmax A | schema minus prediktor (parat) |
| --- | --- | --- |
| Schemat S | **84,3 %** | – |
| Senaste meningen (förregistrerad) | 34,7 % | +0,50, z 14,8 |
| Ordöverlapp (förregistrerad) | 26,3 % | +0,58, z 16,1 |
| Alltid målmeningen | 69,0 % | +0,15, z 5,4 |
| Ämne → mål eller lock (tabell ur träningen) | 69,0 % | +0,15, z 5,4 |
| Ämne + målposition + lockposition (tabell) | 77,0 % | +0,07, z 3,5 |
| Roll + position + "uppgiften rätt" (använder svaret, bara jämförelse) | 86,0 % | – |

- `overlap_choice` pekar **alltid** på lockmeningen (1 200 av 1 200 och 300 av 300). Nollmodellen "ordöverlapp" är
  alltså samma sak som "alltid lockmeningen". Det är så texterna är byggda, men den nollmodellen säger nästan
  inget.
- Argmax A ligger på målet i 69 %, på lockmeningen i 26 % och på en fyllnadsmening i 5 %. Uppgiften handlar alltså
  nästan bara om "mål eller lock". Lock sammanfaller med fel svar: 56 av 62 felsvar har argmax A på lockmeningen.
- Schemat slår också den starkaste textbaserade nollmodellen (ämne och positioner, 77 %) med z 3,5. När A inte
  pekar på målet träffar schemat 68 % (63 av 93). Schemat bär alltså information per text utöver ämnet och
  positionen. Positionsvana förklarar det inte heller: argmax A fördelar sig 59/77/60/104 över platserna.

## 2. Ny analys: schemat generaliserar inte till nya ämnen (allvarlig för tolkningen)

Jag tränade om schemat med samma hyperparametrar (h 64, wd 0,1, dropout 0,3). Varje gång hölls fyra av tolv
målämnen utanför träningen, och schemat testades på testtexter med de ämnena som mål (tre veck).

| | argmax S = argmax A | alltid mål |
| --- | --- | --- |
| Nya ämnen, sammanlagt (n = 300) | **33,3 %** | 69,0 % (schema −0,36, z −8,3) |
| Kontroll: samma storlek (670), slumpvis delning, kända ämnen | 80,7 % | – |

För nya ämnen pekar schemat oftast på lockmeningen (38 av 89 i ett veck) eller på en fyllnadsmening (31), sällan
på målet (20). Det har alltså lärt sig vilka av de tolv mål- och lockmallarna som hör ihop med vilken fråga. Det
har inte lärt sig att läsa av var modellen riktar uppmärksamheten utifrån tillståndet. Det är inget läckage, och
G2a är uppfyllt som det förregistrerades. Men slutsatsen bör skrivas som "S förutsäger A för texter byggda av
kända mallar". Det påverkar också styrningen: S_own på nya texter vore nästan slumpmässigt.
**Förslag [granskarens]:** redovisa delningen med nya ämnen som beskrivande mått i grind 2. Fler mallar per ämne
(flera målmeningar och frågor per ämne) behövs om schemat ska användas i chatten eller på fria texter.

## 3. Styrningen

**Biasen läggs rätt.** Jag kontrollerade koden och körde egna prov på den lilla modellen i fp32 och bf16.
- Kolumner: bara promptens nyckeltokens som ingår i en mening får bias. Mallmarkörer, systemprompt, fråga och
  svarstokens får 0 (`steer.py:57-59`, `make_bias`).
- Rader: `pos >= start` med `start = len(ids) − 1` och absoluta `position_ids` (`steer.py:53,60`). Under
  generering med KV-cache har förifyllningen raderna 0…n_p−1, och bara den sista får bias. Varje avkodningssteg har
  en rad med `pos ≥ n_p`. Mätningen utan cache biasar samma rader som A mäter (n_p−1 …). Mitt prov visade att
  styrd girig generering med cache gav exakt samma token som argmax ur en styrd körning utan cache, i fp32 och bf16.
- Lager: hookar bara på `self_attn` i `meas.layers` = 9–17. Det prövas (`test_steer.py`, lager före oförändrade,
  kvot = exp(bias) i första styrda lagret).
- Mask: med eager och transformers 5.24 är masken ett flyttal i modellens dtype (fp32 −3,40e38, bf16 −3,39e38,
  alltså min och inte −inf). Formen är (1, 1, Q, K) och under avkodning (1, 1, 1, K) med nollor. Att lägga till en
  bias på högst cirka ±12 ger varken spill till −inf eller NaN. Grenen för bool-mask (`steer.py:55-56`) används inte
  med eager.
- β = 0 och jämnt S: `run()` hoppar över styrningen helt vid β = 0 (`steer_eval.py:83`). Med aktiv hook och
  β = 0, eller med S = 0,25 och β = 3, blev attentionvikterna **bitidentiska** (torch.equal) med basmodellen i fp32
  och bf16, eftersom log(4 · 0,25) = 0 exakt.

**Designval att redovisa:** styrningen är öppen slinga. S räknas en gång ur ostyrda tillstånd
(`steer_eval.py:96`) och räknas inte om medan svaret skrivs.

## 4. β-regeln och G2b: sanna per konstruktion

Biasen läggs på exakt de lager (9–17) och rader som A mäts över. I första styrda lagret ändrar den den tvingade
meningens odds mot de andra meningarna med faktorn r = exp(β · (ln 3,88 − ln 0,04)) = exp(4,58 β). Det ger 9,8
vid β = 0,5 och 97 vid β = 1. Jag tog A i testdatan som baslinje, valde en icke-målmening slumpvis och räknade bara
med den aritmetiken. Den förväntade ökningen av A på meningen blir då **0,46 vid β = 0,5**, och i 100 % av texterna
≥ 0,20. Villkor (a) i β-regeln (`beta_choice.py:15`) och G2b (> 2 SE) klaras alltså av algebran, oavsett vad
modellen gör efteråt. Regeln väljer minsta β som klarar (a) och (b) (`beta_choice.py:17`), så β = 0,5 är nästan
säkert. Det ger den svagaste effekten på beteendet.

Villkor (b) har låg styrka. 40 texter ger SE ≈ 6 procentenheter, mot en tolerans på 5. Villkoret skyddar alltså
bara mot grov skada.

**Vilka mått visar följder:**
- *Svaret ändras* under tvingad styrning jämfört med basen, och *uppgiften blir fel* (parat). Det är det
  starkaste beviset för följder.
- *Svaret nämner den tvingade meningens värde.* Det fungerar bara för fyllnadsmeningar, eftersom
  lockmeningen saknar värde, och kräver rättning (anmärkning 3).
- *A_late* (lager 18–27). Biasen läggs inte där direkt, men lagren får styrd indata via residualströmmen. Måttet är
  bättre än A men fortfarande delvis mekaniskt.
- *Kontroll som saknas:* S_own jämförd med ett S som inte hör till texten (permuterat S eller S från en annan text)
  med samma β. Först då visas att det är *schemats innehåll* som spelar roll och inte att en bias läggs över
  huvud taget. Med anmärkning 2 i åtanke bör kontrollen göras.

**Förslag [granskarens]:** innan testfasen körs låser Anders ett beteendekriterium för G2b. Ett exempel: andelen
där svaret ändras, eller andelen fel, under tvingad styrning minus bas, > 2 SE parat. A-förskjutningen redovisas
då som manipulationskontroll och inte som resultat. Det är en ändring av förregistreringen och kräver Anders
godkännande.

## 5. Testerna

Testerna kan fallera på rätt sätt. Kvottestet `exp(bias)` i första styrda lagret fångar fel kolumner, fel bias och
fel rad vid `start`. Testet "lager före oförändrade" fångar fel lager före 9, och bitidentitet vid inaktiv hook är
ett hårt krav. Luckor:
- Inget test av styrd generering med KV-cache. Jag provade det själv och det fungerar, men det borde finnas som test.
- `test_steer.py` körs aldrig på riktiga modellen. Steget `test` i `setup_windows.ps1:17` kör bara
  `tests/test_invariants.py` med `NEXIS_REAL_MODEL`.
- Inga tester för `beta_choice.py` eller för `evaluate`/`paired` i `train_schema.py`.
- `test_forced_steering_moves_A` (β = 4, krav +0,1) klaras per konstruktion och visar bara att hooken körs.
- Inget test kräver att lager *efter* 17 saknar direkt bias. Inget test kontrollerar rader > start under generering.
- `test_splits_disjoint_from_baseline_seed` tillåter ≤ 2 gemensamma texter. Det är rimligt.

## 6. Windows CPU med riktiga modellen

Datan togs fram med torch 2.15.0+cpu, transformers 5.24.0 och fp32 (ur `model_info`), alltså samma versioner som
här. Risker:
- `requirements.txt` har `transformers>=4.56` utan övre gräns. Om en ny installation drar in en annan version kan
  anropet till `self_attn` byta nyckelord. Ett saknat `position_ids` eller en mask som är None ger ett tydligt fel.
  Om hooken inte anropas alls blir styrningen däremot **tyst avslagen**: `run()` kontrollerar inte att
  `steerer.calls` ökat (`steer_eval.py:86-87`). Lås transformers till 5.24.0 och kontrollera `calls > 0`.
- dtype "auto" väljer fp32 vid ≥ 14 GB RAM, annars bf16. Schemat tränades på fp32-tillstånd. `steer_eval.py`
  kontrollerar inte att dtype stämmer med schemadatans. Kör med `--dtype float32` eller kontrollera dtype.
- Återupptagning: `sprint2_beta_base.jsonl` skrivs bara om filen saknas (`steer_eval.py:141`). Avbryts körningen
  mitt i filen blir den aldrig färdig. Då beräknas `beta_choice.py` på färre texter eller stoppar med KeyError
  (`beta_choice.py:11`, `base[r["idx"]]`). En avhuggen sista rad i en jsonl-fil ger JSONDecodeError vid
  återstart (`steer_eval.py:127,154`).
- `beta_choice.py` tar sökvägarna som argument men saknas i `setup_windows.ps1`. Skriptet skriver till
  `results/` relativt den aktuella katalogen.
- Tiden: varje testtext körs i tre villkor, och varje villkor består av generering med 24 tokens och ett pass
  med attention i alla lager. Det bör gå på CPU, men räkna med timmar för 300 texter. Loggarna går via
  `Tee-Object`, och PowerShell 5.1 skriver dem i UTF-16 (som `tests_real_model.txt`). Det är ofarligt.

## Anmärkningar efter allvar

| # | Allvar | Fil:rad | Anmärkning |
| --- | --- | --- | --- |
| 1 | Allvarlig | `nexis3/steer.py:60-61`, `beta_choice.py:15,17`, `results/sprint2_prereg.md` (G2b, tillägg) | G2b och β-villkor (a) är sanna per konstruktion: förväntad ökning 0,46 vid β = 0,5 bara av aritmetiken. Lås ett beteendekriterium (svaret ändras eller blir fel, parat) innan `steer_eval.py test` körs. Kräver Anders godkännande. |
| 2 | Allvarlig (tolkning) | `nexis3/textgen.py` (TOPICS_V2, en mall per ämne), `train_schema.py:64` | Schemat generaliserar inte till nya ämnen: 33 % mot 69 % för "alltid mål" (z −8,3), med kontroll 81 % vid samma datamängd. G2a gäller bara kända mallar. Redovisa och begränsa påståendet. |
| 3 | Medel | `steer_eval.py:102` | `mentions_forced_value` matchar delsträngar ("red" i "hundred", "oak" i "soaked", "four" i "fourteen") och missar siffror ("three" och "3"). Lockmeningen ger alltid False. Använd ordgräns och siffermappning som i `answer_correct`, och analysera bara fyllnadsmeningar. |
| 4 | Medel | `steer_eval.py` (villkor) | Kontroll saknas: S_own mot permuterat S eller S från en annan text, med samma β. Utan den visas inte att schemats innehåll spelar roll. |
| 5 | Medel | `steer_eval.py:86-87`, `requirements.txt` | Styrningen kan bli tyst avslagen med en annan transformers-version. Kontrollera `steerer.calls > 0` och lås `transformers==5.24.0`. |
| 6 | Medel | `steer_eval.py:141-147`, `beta_choice.py:11` | Baslinjefilen för β-valet blir aldrig färdig om körningen avbryts, vilket ger KeyError eller fel acc0. En avhuggen jsonl-rad stoppar återstarten (`:127`, `:154`). |
| 7 | Liten | `steer_eval.py` (`--dtype auto`) | Ingen kontroll att dtype stämmer med schemadatans (fp32). |
| 8 | Liten | `beta_choice.py:15` | Villkor (b) har låg styrka (40 texter, SE ≈ 6 p mot tolerans 5 p). Minsta β väljs, vilket ger svagast effekt på beteendet. |
| 9 | Liten | `train_schema.py:68-73` | Samma valideringsdel används för tidig stopp och för val av hyperparametrar. Valideringssiffran är optimistisk. Testet påverkas inte. |
| 10 | Liten | `results/sprint2_prereg.md` | Nollmodellen ordöverlapp är identisk med "alltid lockmeningen". Redovisa också "alltid mål" (69 %) och ämne + position (77 %). Schemat slår båda (z 5,4 och 3,5). |
| 11 | Liten | `tests/test_steer.py`, `setup_windows.ps1:17` | Inget test av KV-cache, styrtesterna körs inte på riktiga modellen, inga tester av `beta_choice` eller `evaluate`. |
| 12 | Info | `steer_eval.py:96` | Styrningen är öppen slinga: S räknas ur ostyrda tillstånd och räknas inte om. Det bör redovisas som designval. |

Allt märkt [granskarens] är förslag, inte beslut. Ändringar av förregistreringen beslutar produktägaren.
