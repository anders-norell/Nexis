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
