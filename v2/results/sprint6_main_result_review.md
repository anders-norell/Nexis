# Sprint 6 (1c): granskning av huvudkörningens resultat inför grind 6

Granskaren är en separat agent (25 okt 2026). Underlag: `h5_1c_lock.json`, `sprint6_main_prereg.json`, `sprint6_main.json`,
`sprint6_main_eval.log`, `eval_main_1c.py`, `probe_1c.py`, `test_probe_1c.py`, de tre granskningarna före låsning, CLAUDE.md
och loggarna `results/logs/c6_mL-*`. Jag har inte ändrat någon kod och inte räknat om något utfall. Egna kontroller gjordes
bara med syntetiska kanaler på en otränad 1c-agent och ett tillfälligt lås (skript i scratch, inte i repot).

## Utlåtande: **bedömningen är korrekt enligt låsningen**

Alla 14 låsta filer har oförändrad sha256. Låsfilen skrevs 20:37, `sprint6_main.json` 23:30. Utvärderingsloggen visar en
enda körning i ordningen baslinje → 1c → kapacitet → 1b-E. Varje checkpoint är kontrollerad (fångstrisk 0,5, kostnad 0,0001,
≥ 1 M episoder, flaggor, frö). `judge()` kräver (a), (b) och (c) i alla tre fröna innan "om" kan klaras, och
AST-specificiteten kräver att "om" har klarats. Båda 1c-villkoren får **förutsättningar ej klarade**. Därmed bedöms inte
H5 och inte heller AST-specificiteten (`judged: false`). Det är rätt räknat:

| | (a) mot baslinje, z per frö 30/31/32 | (b) mot kapacitet | (c) prob > likformig |
| --- | --- | --- | --- |
| 1c med larm | +2,1 / +3,0 / **+1,75** | +2,6 / +3,3 / **+1,56** | +18 / +41 / +29 |
| 1c utan larm | −0,9 / +0,4 / −1,1 | −0,4 / +0,7 / −1,3 | **−0,8** / +5,0 / +5,2 |

## Fynd

**1. Viktig (tolkning): proben läser starkt negativa värden vid varje kanal som skiljer ut det avsedda fönstret, också när
kanalen inte vet något om fångst.** H5-värdena räknades fast förutsättningarna inte klarades. Det står i koden
(`evaluate_run` beräknar allt), inte i låsningen. De får redovisas, men bara som beskrivande. Jag körde några syntetiska
kanaler (2 000 episoder, prob tränad på fångststeg som i låsningen), z för lockbete/matchat:

| Kanal | Lockbete | Matchat | Korr. avsett–faktiskt |
| --- | --- | --- | --- |
| markerar alltid avsett (+3), ingen fångstinformation (testsvitens "intended") | −56 | −62 | – |
| avsett markerat negativt (−3), ingen fångstinformation | −55 | −68 | −0,19 |
| avsett negativt + faktiskt med sannolikhet 0,6 | −25 | −24 | −0,40 |
| bara iögonfallande | −54 | **+3** | −0,25 |
| passiv svag markering av avsett | −6 | −6 | −0,09 |
| AST-lik (r = 0,6) | +21 | +22 | −0,90 |
| **1c med larm, tränad (frö 30/31/32)** | −25/−77/−39 | −19/−91/−40 | −0,20 till −0,24 |
| **1b-E, tränad** | −94/−96/−108 | +0,6/−9/−20 | −0,06 till +0,05 |

Proben tränas bara på fångststeg, och där är det avsedda fönstret aldrig det faktiska. Proben lär sig därför att varje drag
som utmärker avsikten betyder "inte här", oavsett tecken. Det var granskning 3:s fynd 3, och det är därför kriteriet är
konservativt. 1c med larm har samma mönster som en ren avsiktsmarkör: båda kontrollerna är starkt negativa och korrelationen
är svag. Det säger att kanalen bär *var agenten siktade* i en form som skiljer sig från faktiskt fönster. Det säger däremot
inte att kanalen vet att fångst skett. Avsiktsinformation finns passivt i indata via efferenskopian och matchningssignalen.
1b-E ser annorlunda ut: lockbetet är starkt negativt men det matchade fönstret ligger nära noll, ungefär som en kanal som
kodar iögonfallande/faktiskt. **Storleken i SE (−17 till −95) mäter alltså hur lätt avsikten går att avläsa. Den mäter inte
ett "anti-AST-fel".** Rätt läsning är riktningen. Inget pekar mot AST-signaturen, men de negativa värdena är inget belägg för
en kanal som rättar sitt fel. Mätfel finns det däremot inga tecken på. Kontrollerna ligger på noll för de tomma kanalerna
(kapacitet: |z| ≤ 1,1, också i skuggmodulen), bootstrap sker över episoder (cirka 3 händelser per episod med händelser,
`boot_clipped_frac` 0), och tecknet är förenligt med kalibreringen.

**2. Viktig: det nollmodellsjusterade "om" går inte att använda.** Det blir −35 till −36 SE också för kapacitetsbaslinjen,
vars kanal är tom ((c) z ≈ 0, probförlust ≈ slump). För 1c utan larm blir det −35 till −40. Vid högt q (0,92) lutar
nollmodellen mot avsett, vilket CLAUDE.md redan noterade för tom kanal. Värdena −46 till −85 för 1c med larm ska därför
inte redovisas som H5-mått. De bör bara nämnas med den här förklaringen.

**3. Medel: (b) är en kontroll för parametrar, inte för information (doskontrollregeln).** Kapacitetsbaslinjens kanal sjunker
från cirka 45 till < 12 bitar inom 15 000 episoder och slutar på 0,5–2,4 bitar. (c) blir då z −1,6 till −0,1 och träffandelen
78,4–79,1 % (baslinjen 78,6–79,0). Det beror på konstruktionen: indata från en annan episod är värdelös, så kostnaden
nollställer kanalen. Kontrollen ändrar alltså nästan ingen information mot 20–28 bitar i huvudvillkoret, och därmed är (b) i
praktiken en upprepning av (a) mot en baslinje med fler parametrar. Den utesluter att extra parametrar, modul eller brus ger
vinsten, men den utesluter inte att vilken informativ extrakanal som helst skulle göra det. En verklig doskontroll med
blandad men informativ indata går inte att bygga på det här sättet, och det borde ha stått i förregistreringen. Utfallet
påverkas inte, eftersom (a) redan faller i frö 32.

**4. Medel: 1c utan larm prövar inte självmodellering, den kollapsar.** Kanalen har 2–4 bitar, probförlusten ligger nära
slump och (c) faller i frö 30. Det är väntat: modulen får exakt policyns egna tensorer och har inget nytt att förmedla, så
kostnaden drar ner den. Tillsammans med punkt 3 betyder det att 1c:s lilla vinst kommer av **ny indata** (matchningssignal
och larm) som leds via kanalen, inte av att modulen modellerar policyns uppmärksamhet. Det som saknas är en kontroll där
policyn får larmet direkt, utan modul. Den kontrollen avgör om "schemat" tillför något utöver jämföraren.

**5. Liten: frö 32 och SE.** SE är rätt beräknat enligt låsningen: oparat, binomialt, 4 000 episoder,
√(0,00627² + 0,00645²) = 0,0090, och z = 1,75. Samma bankepisoder gör att ett parat SE vore mindre, men låsningen säger
oparat och det ska inte ändras. Däremot har kriteriet låg styrka. Vid en verklig vinst på cirka 2 p är chansen per frö för
> 2 SE ungefär 0,6, och för alla tre fröna ungefär 0,25. Sammanvägt är 1c med larm +2,0 p (z ≈ 3,9 över tre frön), och fas E
gav +2,3/+2,8. **En liten verklig nytta är sannolik, men den räcker inte till det låsta kriteriet.** Det får redovisas så,
men det ändrar inte bedömningen.

**6. Liten: 1b-E-jämförelsen.** 1b-E når 81,2–81,5 % med 139–149 bitar, och 1c med larm når 80,5–81,3 % med 20–28 bitar.
Utan facit tar 1c alltså nästan hela den nytta som går att hämta, med ungefär en sjättedel av bitarna. Ingen av dem visar
AST-felet. Per frö skiljer de sig åt i olika riktningar (lockbete +4,8 till +43 SE, matchat −16 till −92), men det är inte
bedömt och inte tolkbart enligt punkt 1. Skillnaden visar mest att 1c kodar avsikten mer avläsbart än 1b-E.

**7. Formalia.** Kapacitetskörningen i frö 30 visar 9 min eftersom minutvärdet nollställdes vid återstarten 20:10.
Checkpointen har ≥ 1 M episoder (kontrolleras). "var" och känslighetsproben pekar åt samma håll som den primära proben.

## Tolkning

Enligt låsningen är svaret nej: förutsättningarna klaras inte, så H5 för 1c bedöms inte. Med larm ger 1c en liten nytta,
cirka 2 p, som sannolikt är verklig men inte når kriteriet i alla tre fröna. Nyttan beror på jämförarens signal och inte på
en självmodell som växer fram. Utan den signalen kollapsar kanalen. Den beskrivande H5-bilden går mot AST, men inte så
starkt som SE-talen antyder. Kanalen skiljer det avsedda fönstret från det faktiska i stället för att förväxla dem. Det är
vad en avsikts- eller larmmarkör ger med den här proben, och det är inget positivt belägg för felkorrigering. Den här
uppställningen har alltså inte frambringat något styrschema med AST:s felsignatur. Ett nytt försök bör börja med att visa
att något utöver jämförarsignalen bärs, till exempel med kontrollen där policyn får larmet direkt.
