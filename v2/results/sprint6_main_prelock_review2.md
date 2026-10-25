# Sprint 6 (1c): andra granskningen före låsning av huvudkörningen

Granskaren är en separat agent (25 okt 2026). Jag har gått igenom `probe_1c.py`, `eval_main_1c.py`, `run_sprint6.sh` och `test_probe_1c.py` mot förra granskningen och CLAUDE.md. Jag har bara kört otränade agenter och syntetiska kanaler och inte laddat någon tränad checkpoint. Mina extra kanaler finns i scratch-skript och ligger inte i repot. Kalibreringen `test_probe_1c.py` går igenom (ALL OK).

## Utlåtande: **ej låsbar**

Fynd 2–6 är åtgärdade. Fynd 1 har bytt riktning men är inte löst (se nytt fynd A). Det beror på att låtsasfönstret alltid är iögonfallande och det avsedda aldrig är det. Därför kan en kanal som korrekt följer var uppmärksamheten faktiskt varit, utan någon rollinformation, klara kontrollen falskt.

## Förra granskningens fynd
- **1 (iögonfallande):** delvis åtgärdat. En ren iögonfallande-karta ger nu −58 SE. Det är konservativt, men felet finns kvar i omvänd form (A).
- **2 (skuggmodul):** åtgärdat. Jag kontrollerade 200 episoder för både 1c och 1b-E. Med skuggmodul blir found, händelser, F, faktiskt fönster och låtsasfönster bitidentiska med körningen utan, och F_sh skiljer sig från F. Mätt med otränade moduler (frö 30–32, 2 000 episoder, fångstrisk 0,5) är kontrollen −0,003 till −0,015 (z −2,8 till −8,7) för 1c med larm, −0,002 (−3,3) utan larm, −0,003 för 1b-E och −0,001 (−0,8) för kapacitetsbaslinjen. Tecknet är numera stabilt negativt.
- **3:** åtgärdat. `run_sprint6.sh` ingår i hashlistan, och `checkpoint_ok` kontrollerar fångstrisk, längd, kostnad, mål och flaggor. Rest: fröet och, för Baslinjen, `schema == "off"` kontrolleras inte (`eval_main_1c.py:62–82`), trots att docstringen säger "seed".
- **4:** åtgärdat. Cachen går nu på sha256 (`eval_main_1c.py:108–111`).
- **5:** saknar betydelse nu, eftersom AST-specificiteten jämför tränad 1c direkt med tränad 1b-E.
- **6:** står i docstringen. Det ska också stå i låstexten.

## Nya fynd

**A. Allvarlig: en korrekt kanal över "ännu inte besökta iögonfallande föremål" klarar kontrollen falskt (`probe_1c.py:229`).**
Proben tränas bara på fångststeg. Där är det faktiska fönstret ett iögonfallande föremål som just har besökts. En kanal där iögonfallande dämpas där uppmärksamheten faktiskt varit gör att det faktiska och det avsedda fönstret ser likadana ut, medan låtsasfönstret förblir markerat. Syntetiska kanaler som bara bär korrekt information och ingen roll gav:

| Kanal | Kontroll |
| --- | --- |
| iögonfallande med dämpning där fångsten skedde (r = 0,6–1,0) | +14 till +102 SE |
| iögonfallande × ej besökt | +4 respektive +12 SE |
| prioritetskarta: målfärg och iögonfallande på ej besökta rutor | +7 respektive +12 SE |

Kalibreringen saknar helt kanaler som byggs av miljöns kartor (målfärg/prioritet, besökt, dämpning av redan besökta platser). Den prövar bara mönster definierade på kandidaterna. Vid H5-händelser med låtsasfönster skiljer sig låtsasfönstret från det avsedda också på andra sätt: det har målfärg i 65 % av fallen mot 32 % och är besökt i 58 % mot 26 %.

Det här är troligt för en tränad 1c. Modulen får periferins iögonfallande-kanal och larmet, och nyttan består i att veta vad man faktiskt täckt (sprint 3: agenten använder kartan "täckt hittills").

Åtgärd:
1. Lägg in sådana kanaler i `test_probe_1c.py` med krav z < 2.
2. Inför ett andra kontrollfönster som inte är kandidat och som matchar det avsedda: ej iögonfallande, samma besökt-status och samma målfärgsinnehåll. Det läggs till som extra kandidat vid test. Kräv att båda kontrollerna klarar > 2 SE: låtsasfönstret fångar passiv vidareföring, det matchade fönstret fångar iögonfallande och dämpning. Det här har jag inte prövat, så kalibrera före låsning.
3. Produktägaren beslutar.

**B. Medel: en täckningskarta som slår ihop kommandon och faktisk täckning klarar kontrollen (+17 SE).**
Det är tolkningsfrågan om en minnesbild av efferenskopian ("jag tror att jag täckte dit jag siktade") räknas som AST-stöd. Åtgärd: låstexten ska säga det uttryckligen. Jämförelsen med 1b-E (som har samma indata) avgör om det är specifikt.

**C. Medel: kriteriet är snålt mot svaga eller blandade AST-signaler.**
- Svag AST (avsett markerat vid 10/20/30 % av missarna) ger +0,022/+0,052/+0,066 (+3,9/+7,0/+8,9 SE), att jämföra med den passiva biasen hos otränade moduler på −0,003 till −0,015.
- Om kanalen dessutom kodar iögonfallande sjunker även en stark AST-kanal från +0,14 till +0,026, och en svag till −0,038 (−4,7 SE).
- Ett nej på "om" är alltså inte informativt om kanalen kodar iögonfallande. Åtgärd: säg det i låstexten och redovisa probens vikt på iögonfallande (blint), eller lös det genom kontrollfönstret i A.

**D. Liten: inaktuella texter.**
- `probe_1c.py:18` och `eval_main_1c.py:8` säger "balanced".
- `measure_pair` (`probe_1c.py:565`) kallar sig "The judged 'om'", men måttet bara redovisas.
- Kalibreringssiffrorna i `measure` stämmer ungefär (AST +16,9 till +22,2 SE).

## Blindhet (iv)
Inga brott. `measure`, `c_all` och `measure_pair` anropas bara från `eval_main_1c.py` efter `_check_lock()`. `pilot_1c.py` använder bara `blind_summary` (K ≥ 3), och inga resultatfiler innehåller kontroll eller massa på avsett fönster. Låsfilen finns inte.
