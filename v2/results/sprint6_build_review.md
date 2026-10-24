# Sprint 6 (1c), steg 1: oberoende granskning av bygget

Granskare: separat agent, 25 okt 2026. Underlag: commit cacd252, specvillkoren 24 okt 23:45, CLAUDE.md.
Bara otränade agenter och syntetiska kanaler har körts. Ingen tränad checkpoint har laddats, och inget mått på var probens fel pekar har beräknats på riktiga data.

## Utlåtande: **godkänd med anmärkningar**

Fas A i piloten (`sh run_sprint6.sh A`) kan startas. Piloten är blind och beror inte på "om". Anmärkning 3 ska vara åtgärdad före fas B, eftersom fas B kör kapacitetsbaslinjen. Anmärkning 1 och 2 ska vara lösta och beslutade av produktägaren före låsningen vid grind 6a.

## Kontrollerat utan anmärkning

- **Låsta 5a-filer orörda.** Repot är en grund klon, så `git diff 988b8ae` går inte att köra (commiten finns inte). Kontrollen gjordes i stället på två sätt: (1) alla nio filer stämmer mot sha256 i `results/h5_lock.json`, (2) `git diff 123fc06 HEAD` är tom för dem.
- **Testerna går igenom.** `test_1c.py` och `test_probe_1c.py` ger båda ALL OK (2 s respektive 16 s).
- **Handskrivna gradienter.** Kanalkolumnens indatagradient `dA·Wz[k] + dpre·W1[k]` är rätt (schema_1c.py:82). Reparametriseringen, kostnadstermen (dμ = μ/ln2, d log σ = (σ²−1)/ln2), klippningen av log σ och maskningen är också rätta. Alla stämmer mot finita differenser, för alla tre varianterna.
- **Ingen annan inlärningssignal.** Modulen tränas bara via dz (policyns förlust inklusive kritikern) och straffet β·bitar. Dess egna huvuden w2/wv/ws/W1 får nollgradient: de är döda parametrar, men lika många i alla villkor. Kritikerns andel av modulgradienten loggas (0,87 på otränad agent).
- **Tidsordning.** z_t beräknas av observationen vid t och läses av policyn vid samma t. Vid t = 0 är kanalen stängd. Proben läser kanalen vid t+1 och t.
- **"Utan" och "med".** "Utan" är exakt policyns tensor, med schemakolumnerna satta till 0. "Med" är exakt 1b-E:s indata. Båda är testade.
- **Proben.** Den ger varje kandidat samma poängfunktion, ordningen spelar ingen roll, den saknar konstant per kandidat och den tränas balanserat. Index 0 (avsett fönster) används ingenstans i träningen annat än som etikett när ingen fångst skett.
- **Nollmodellen.** Den har likformig prior och ersätter glimten efter fångstens förhandssannolikheter. q matchas på massan på faktiskt fönster och löses om i varje bootstrapomgång. Utan matchbart q räknas körningen som ej bedömbar.
- **Bootstrap** görs över episoder, som kvot av summor.
- **Blindhet.** `pilot_1c.py` anropar aldrig `measure()`, och `measure()` vägrar köra utan lås. Se dock anmärkning 4.

## Anmärkningar

**1. Allvarlig (måste lösas före låsning vid 6a): "om" är inte kalibrerat för en kanal utan rollinformation.**
Gäller probe_1c.py:292–307 (`null_curves`) och probe_1c.py:411–418, samt test_probe_1c.py:165–168.
Jag körde syntetiska kanaler där felen inte bär någon information om avsikten:
- **Rumslig markör.** Faktiskt fönster markeras vid t+1 med sannolikhet *r*, annars bara brus. "om" blev −1,7 SE vid r = 0,4, +2,0 till +2,6 SE vid r = 0,6, +4,1 till +4,7 SE vid r = 0,7, och ej bedömbart från r = 0,8.
- **Fel jämnt fördelade på fel kandidater.** "om" blev −10,7 SE vid träffsäkerhet 0,5 och −2,2 SE vid 0,7.

Byggets egen "AST-lika" kanal ger +4,0 SE. Testsviten skiljer alltså inte AST från en kanal utan rollinformation.

Mekanismen: nollmodellen gör redan vid q = 0 fel på kandidater med samma innehåll i begränsad form (i snitt 1,47 per händelse), och dessa fel hamnar aldrig på avsett fönster. Probens rollblinda fel hamnar däremot ibland där. Strax ovanför q = 0 blir "om" därför sant per konstruktion. Vid lägre träffsäkerhet blir det falskt per konstruktion, eftersom förhandsprioren lägger glimterna på avsett fönster.

Åtgärd:
- (a) Lägg in ett test med kanaler utan rollinformation vid träffsäkerhet 0,5–0,8 och kräv |z| < 2.
- (b) Om testet inte klaras: förregistrera en rollbyteskontroll. Vid K ≥ 3 räknas "om" också med ett slumpvis valt fel kandidatfönster som "avsett". Kräv att "om" minus kontrollen är > 2 SE, eller kräv q i ett inre intervall.
- (c) Produktägaren beslutar före låsningen.

**2. Stor (beslut före 6a): kanalen vid t är styrsignalen.**
Gäller probe_1c.py:116 och :170. z_t är det policyn använde för att välja avsett fönster vid t. En ren "titta hit"-kanal, alltså ingen modell av uppmärksamheten, ger därför massa på avsett fönster via särdragen från t. Syntetiskt höjde en ren styrmarkering vid t "om" med 1,2–2,4 SE. Villkoret säger "t och t−1", och bygget har tolkat det som avläsning vid t+1 och t. Tolkningen är rimlig men ger avsikten delvis gratis.
Åtgärd: förregistrera en känslighetsanalys med prob bara på t+1, och anteckna i låstexten att särdragen från t bär kommandot.

**3. Medel (före fas B): kapacitetsbaslinjen använder givarens mask för kanal och bitar.**
Gäller train_1c.py:46–51, schema_1c.py:118–120 och probe_1c.py:71–74. z och bitarna räknas på givarens giltiga rutor, och sedan nollas z utanför de egna rutorna. Med rutnät 6–9 hamnar i snitt ca 13 % av bitarna på rutor som policyn inte ser, och ca 13 % av de egna rutorna får z = 0. Det är samma nivå i bitar men lägre användbar kapacitet, alltså ett doskontrollfel.
Åtgärd: låt nätet använda givarens mask men kanalen och bitarna policyns egen (`mm[idx]`).

**4. Medel (redovisas i låstexten): (c) avslöjar delvis "om".**
Gäller probe_1c.py:345–351. Vid H5-händelser med K = 2 är probens massa på avsett exakt 1 − massan på faktiskt. Massan på faktiskt vid H5-händelser, som (c) redovisar blint, blandar alltså in svaret för K = 2.
Åtgärd: redovisa (c) uppdelat på K = 2 och K ≥ 3 först efter låsningen, eller ange andelen K = 2 och läckan ärligt i låstexten.

**5. Medel: förutsättning (b) och doskontrollen saknas i piloten.**
Gäller pilot_1c.py:113–141. Ingenstans jämförs "med" mot kapacitetsbaslinjen, och det kontrolleras inte att kapacitetsbaslinjens uppnådda bitar ligger nära huvudvillkorets. En kanal som policyn inte kan använda kan hamna långt under målet, eftersom β då drivs mot golvet 1e-7 utan att något håller bitarna uppe. Det kontrolleras inte heller att någon körning når sin nivå.
Åtgärd: lägg till `benefit(…, base="cap")` och kräv skillnad > 2 SE i båda fröna (förregistrera). Logga uppnådda bitar mot målet och kräv t.ex. ±20 %, både för nivåregeln och för kapacitetsbaslinjen.

**6. Liten: nivåregeln kan dra slutsatsen för tidigt.**
Gäller pilot_1c.py:134–138. Saknas en "med"-körning, eller har den inte blivit klar, föreslår regeln fas U ("ingen nytta på någon nivå").
Åtgärd: kräv att alla sex "med"-nyttor finns innan regeln drar slutsatser. Faserna C och D (namnen `e1e-4`, `dc-*`) hanteras inte av `level_rule`. Lägg till dem eller dokumentera att de bedöms manuellt.

**7. Liten: kapacitetsbaslinjen körs alltid med larm.**
Gäller run_sprint6.sh:38. Avgör "utan" nivån (via fas U), ska kapacitetsbaslinjen matcha "utan", så att den har samma parameterantal (4 644 mot 4 804).
Åtgärd: välj `--alarm` efter det villkor som avgjorde nivån.

**8. Liten: givarlogiken.**
- `rng.permutation(B)` ger i snitt en fixpunkt per batch, och den episoden får då sin egen indata (train_1c.py:32). Åtgärd: använd en derangering, t.ex. en permutation följd av cyklisk förskjutning.
- När givaren är klar ser modulen en fryst observation, och givaren kör bara 82 % av stegen på otränad agent (train_1c.py:42–44). Åtgärd: redovisa andelen per körning, vilket redan görs. Överväg att välja givare bland episoder som fortfarande kör.
