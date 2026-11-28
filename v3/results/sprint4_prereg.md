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
