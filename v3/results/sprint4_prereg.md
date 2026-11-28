# Sprint 4 – förregistrering, del 1: förutsättning F1 (26 nov 2026, före all körning av v4)

Plan: projektet `claude/nexis_v3_sprint4_plan.md`, godkänd av Anders 26 nov 08:42 ("då går vi vidare med s4 baserat på
dina förslag"): planen, facit i rapportträningen = faktisk attention, observation var fjärde token, körmiljö efter tidsmätning.

**F1** (`f1_check.py`, data `base4`, frö 9000, 100 texter, träningsämnen): bland texter med båda delsvaren rätt ska
argmax av medelandelen per mening (lager 9–17) över del 1:s svarstoken vara mål A och över del 2:s svarstoken mål B i
minst 70 %. Klaras F1 inte stannar vi och diskuterar uppgiften innan något mer samlas in.
Resten av kriterierna (F2, K4a–c) låses i del 2 av förregistreringen efter granskning, före test.

## Tillägg 26 nov ca 10:20: F1 underkänd, F1b (godkänt av Anders 10:14, "yes")
F1 på base4: 37,5 % (n 56), underkänd. Beskrivande: fokus byter vid svarsordens token men medlet över delen domineras av
strukturtoken. **F1b** (`f1b_check.py`, nya data `base4b`, frö 9001, 100 texter, träningsämnen; base4 används inte igen
för bedömning): bland texter med båda delsvaren rätt ska argmax av medelandelen per mening (lager 9–17) över de svarstoken
som skriver svarsordet vara mål A i del 1 och mål B i del 2, i minst 70 %. Innehållstoken = de token vars teckenintervall överlappar första förekomsten
av svarsordet i delens text (skiftlägesokänsligt, `extract_v4.content_rows`; rättat före körning, första versionen räknade
enstaka bokstäver). Saknas innehållstoken i en del räknas texten som
ej klarad. Klaras F1b inte stannar vi igen.
Konsekvens för resten av sprinten (låses i del 2): kärntestet K4 räknas på innehållsordens token.

## Tillägg 26 nov ca 16:00: F1b underkänd, F1c (godkänt av Anders 15:56, "ja")
F1b på base4b: 49 % (n 57), underkänd; per del 70 % och 72 %. Kravet att båda delarna ska stämma i samma text var
strängare än kärntestet behöver (K4 jämför token för token). **F1c** (`f1c_check.py`, nya data `base4c`, frö 9002, 100
texter, träningsämnen; base4 och base4b används inte igen för bedömning): per del, bland delar med rätt delsvar, ska
argmax av medelandelen per mening över svarsordets token vara rätt målmening i minst 70 % av delarna. Klaras F1c
fortsätter insamlingen (test4, train4) direkt; annars stopp. Övrigt i planen oförändrat.

## Del 2a: kartan och förutsättning F2 (28 nov 2026, före all träning av kartan; test4 ej hämtad till arbetsytan)
Allt nedan är [Claudes förslag] inom den godkända planen (observation var fjärde token).

**Observation:** o_t = handvald andel per mening (lager 9–17) vid svarstoken t, bara när t mod 4 = 0.
**Kartan** (`nexis3/kmap.py`): S_t som log-fördelning över de fyra meningarna. Mellan observationer
Ŝ_t = Ŝ_{t−1} + G(Ŝ_{t−1}, q8, t, tid sedan observation); vid observation S_t = Ŝ_t + k·(log o_t − Ŝ_t), förstärkningen
k ∈ (0,1) inlärd ur (Ŝ_t, log o_t). q8 = linjär projektion av frågans läge (lager 13). Kartan ser aldrig texten eller
svarsorden. Förlust: KL(A_t ‖ S_t) över alla svarstoken. Ett frö (0).
**Validering:** sex undanhållna träningsämnen (frö 31); träning = texter utan något av dem; validering = texter vars
ämne A är ett av dem.
**F2** (på validering, alla svarstoken utan observation): medel per text av L1(S_t, A_t) ska vara lägre än för var och en av
(i) senaste observation, (ii) drift per position (träningens medel-A per svarsposition), (iii) konstant (träningens medel-A),
parat per text, > 2 SE mot alla tre. Klaras F2 inte stannar vi och diskuterar.
Redovisas: samma på innehållstoken, och eftersläpningen efter del 2:s start.

## Del 2b: rapportmodellen och kärntestet K4 (28 nov 2026, före all rapportträning och före test4 i arbetsytan)
Allt nedan är [Claudes förslag] inom planen (facit = faktisk attention, godkänt av Anders 26 nov 08:42).

**Validering av F2 klarad** (`sprint4_f2.json`): karta 0,366 mot senaste observation 0,420, drift 0,428, konstant 0,484
(z 8,4 / 11,0 / 24,4).

**Rapportmodellen** (`nexis3/report.py`, `report_train.py`): Qwen fryst, egen LoRA (rang 8, alfa 16) på q_proj och v_proj i
alla lager, projektor S → 4 mjuka token. Rapportfrågan ställs i en ny användarvända efter att svaret kapats efter token t.
Facit = argmax av faktisk A_t (handvald andel per mening, lager 9–17). Exempel ur train4 (aldrig test4): per text en
slumpad innehållstoken per del och en annan svarstoken (frö 0), 3 606 exempel; i 42 % av innehållsexemplen skiljer sig
kartan från attentionen. En epok, AdamW, lr 1e-4. **Kontrollmodellen:** samma exempel, samma ordning, S = likformig.
Antal epoker kan sättas ned till följd av tidsmätningen, men bara före träning och lika för båda.

**Kärntestet** (`report_eval.py`, test4, alla innehållstoken i alla texter, parat per text):
- **K4a:** där argmax S ≠ argmax A: andel rapport = argmax S minus andel rapport = argmax A, > 2 SE.
- **K4b:** S ersätts med en manipulerad karta (0,85 på en slumpad annan mening än argmax S, frö 4242): andel rapport =
  den meningen, minus kontrollmodellens andel för samma mening, > 2 SE.
- **K4c:** där argmax S = argmax A: rapportens träffar på A minus kontrollmodellens, > 2 SE, och minus positionsvanan
  (alltid mening 4, träningens vanligaste facit), > 2 SE.
- Grind 4: K4a och K4b och K4c.

**Tolkning, bestämd i förväg:**
- Kontrollmodellen visar om Qwen kan rapportera sitt fokus inifrån utan karta (kontroll mot vana, redovisas). Är den inte
  bättre än vanan finns ingen inre väg att tävla med, och då visar K4a bara att rapporten använder den enda källa den har.
  Det redovisas så och räknas inte som stöd för AST utöver det.
- Är kontrollen bättre än vanan och K4a klaras: rapporten följer kartan trots att en inre väg finns och trots att facit
  belönar sanningen. Det är AST:s förutsägelse.
- K4b men inte K4a: modellen använder kartan men rättar den med egen tillgång.
- Redovisas: eftersläpning token för token efter del 2:s start (S = A, rapport = S, rapport = A), observerade och
  oobserverade token var för sig.

**Låsning:** `report_eval.py lock` skriver sha256 av koden, kartan och denna fil, efter fristående granskning och före
test4. Testet körs en gång på Anders dator; rapportmodellernas sha256 skrivs i resultatet.

## Del 2b, reviderad efter granskning före låsning (28 nov 2026; ersätter K4-definitionerna ovan) [Claudes förslag]
Granskningen (`sprint4_prelock_review.md`): ej låsbar. K4a sammanföll med positionsvanan och med en textväg (svarsordet står
bara i målmeningen; "målmeningen" = A vid 65 % av oobserverade innehållstoken i train4), K4c var nära sant per
konstruktion via observerade token, K4b hade fel jämförelse och för stor dos. Ändringar, alla före träning och före test4:

**Rapportmodellen:** prefixet (samtalet fram till de mjuka token) räknas med LoRA avstängd och utan gradient, så rapporten
läser samma inre tillstånd som gav attentionen; LoRA verkar bara på de mjuka token och frågan. LoRA-vikter i float32.
**Korsanpassning:** rapporten tränas på S från två kartor tränade på var sin halva av train4 (idx % 2, `kmap_crossfit.py`,
`sprint4_kmap_crossfit.json`: argmax S = argmax A 62 / 61 % på oobserverade token i den osedda halvan), så S i träningen är
av samma slag som på test4. Vid test används `models/kmap.pt` (har inte sett test4:s texter eller ämnen). Var tredje
träningsexempel är en annan token än innehållstoken; behålls och redovisas.

**Alla K4-mått på oobserverade innehållstoken** (t % 4 ≠ 0), parat per text (medel per text, SE över texter), n texter redovisas.
Beteckningar: S = argmax av kartan, A = argmax av faktisk attention, T = textvägen (meningen som innehåller svarsordet),
V = vanan (mening 4). R = schemamodellens rapport, Rc = kontrollmodellens.
- **K4a** (D = token där S ≠ A): [1(R = S) − 1(R = A)] − [1(Rc = S) − 1(Rc = A)] > 2 SE. Kontrollmodellen ser samma text och
  har samma träning utom kartan; den bär därmed vanan och textvägen och är nollmodellen.
- **K4b:** inom schemamodellen, S ersatt med S där toppen och mening j byter plats (dos: medel L1 0,36 mot L1(S, A) 0,48 i D,
  på train4), j slumpad bland meningar ≠ S, A och T (frö 4242): 1(R(manipulerad) = j) − 1(R(sann S) = j) > 2 SE.
- **K4c:** 1(R = A) − 1(V = A) > 2 SE (rapporten säger något sant oftare än vanan). Mot kontroll och mot T redovisas; att
  kräva dem vore motsägelsefullt (en rapport som följer en karta med 61 % träff kan inte slå textvägens 65 %).
- **Grind 4:** K4a och K4b och K4c, var för sig > 2 SE, ingen sammanvägning i efterhand.

**Tolkning, bestämd i förväg:**
- Kontroll mot vana och mot T (även i D där A ≠ T) visar om Qwen har en inre väg till sitt fokus utöver ordmatchning. Klarar
  kontrollen inte T där A ≠ T finns ingen påvisad inre väg; K4a visar då att rapporten följer kartan framför texten och vanan,
  inte framför introspektion.
- K4a i D där A ≠ T (ca 35 % av D, grovt 100 token på test4) redovisas; det är där en introspektiv väg skulle synas.
  Andelen R = S där kontrollen har rätt redovisas också.
- K4b men inte K4a: kartan används men vägs mot annat. K4a men inte K4b: följsamheten är inte kausal på dosnivån.

**Låsning efter träning:** `report_eval.py lock` hashar kod (inkl. compress, textgen, topics_v3, attention), alla tre
kartorna, båda rapportmodellerna och denna fil. Därefter körs testet en gång på Anders dator.

## Del 2b, andra revisionen (28 nov 2026, granskning omgång 2; ersätter K4a, K4c och tolkningen ovan) [Claudes förslag]
Rättelse: på de token som räknas (oobserverade innehållstoken i train4) är argmax S = argmax A **44 %**, inte 61–62 %
(det gällde alla oobserverade token). T = A 65 %.

- **K4a:** nollmodellen är samma schemamodell med **donor-S** (S från en slumpad token i en annan text, dragen om tills
  argmax skiljer sig; frö 4243). Vana och textväg är då identiska i båda armarna; bara kartan skiljer.
  [1(R = S) − 1(R = A)] − [1(R_donor = S) − 1(R_donor = A)] i D, > 2 SE.
- **K4b:** oförändrad.
- **K4c:** 1(R = A | sann S) − 1(R = A | donor-S) > 2 SE: kartan gör rapporten sannare. Mot vanan redovisas.
- **Grind 4:** K4a och K4b och K4c, var för sig.
- **Låst extern kombinerare** (`combiner_fit.py`, `sprint4_combiner.json`, multinomial logistisk regression på train4):
  utan S (T + position, träff 64,7 % på train4) och med S (+ log S, 70,8 %; R = S i D 20 %).
- **Inre väg** (villkor för tolkning): kontrollmodellen slår kombineraren utan S, > 2 SE, både totalt och där A ≠ T.

**Tolkning (ersätter tidigare):** Grinden visar att rapporten använder kartan kausalt som en ledtråd. Det skiljer inte AST
från rationell vägning av ledtrådar: en kombinerare med S, T och position, tränad på facit A utan någon introspektion,
klarar alla tre kriterierna på train4. Det AST-specifika kräver dessutom:
1. att en inre väg finns (kontroll > kombinerare utan S), och
2. att schemamodellen följer kartan **mer** än den rationella kombineraren med S gör (`K4a_vs_combiner` > 2 SE), dvs.
   prioriterar schemat framför sin egen tillgång, särskilt där kontrollen har rätt (andel R = S redovisas för båda).
Saknas 1 är resultatet "rapporten följer kartan i brist på annat". Finns 1 men inte 2: rationell vägning. Båda: AST-likt.
Detta är tolkningsregler, inte grind; grind 4 bedöms på K4a–c.

Omtokeniseringen av det kapade svaret kontrolleras och loggas i tidsmätningen (`report_train.py time`).
Träningsloggen `results/report_train.log` checkas in före låsning.
