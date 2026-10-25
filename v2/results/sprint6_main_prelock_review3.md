# Sprint 6 (1c): tredje granskningen före låsning av huvudkörningen

Granskaren är en separat agent (25 okt 2026). Jag har gått igenom `probe_1c.py` (pick_matched, probs_ext, measure), `eval_main_1c.py` och `test_probe_1c.py` mot de två tidigare utlåtandena och CLAUDE.md. Jag har bara kört otränade agenter (1c med larm, frö 30, fångstrisk 0,5, 1 500 episoder, 4 104 H5-händelser, 1 895 av dem med båda kontrollfönstren) och syntetiska kanaler. Ingen tränad checkpoint är laddad. Testsviten kördes en gång och gav ALL OK. Skripten ligger i scratch och inte i repot.

## Utlåtande: **låsbar med anmärkningar**

Fynd A och C från andra granskningen är lösta. Ingen kanal utan information om avsedd plats klarar det sammansatta kriteriet. Jag prövade testsvitens 14 kanaler och 12 egna, bland annat målfärg × ej täckt, graderad täckning, täckning + iögonfallande/prioritet, "fel bland kandidater" + prioritet och suddig markör kring faktiskt fönster. Alla gav "nej". Det som återstår (fynd 1) handlar om tolkning och kräver ingen kodändring före låsningen, men det måste stå i låstexten.

## Fynd

**1. Viktig: kriteriet skiljer inte "schemat har fel mot avsett" från "kanalen bär avsikten bredvid den faktiska platsen" (`probe_1c.py:631–633`).**
En kanal som markerar både avsett fönster (efferenskopian) och faktiskt fönster klarar kriteriet, trots att den inte gör något fel:

| Kanal | Lockbete | Matchat |
| --- | --- | --- |
| avsett + faktiskt alltid | +88 SE | +85 SE |
| avsett svagare (1,5) + faktiskt (3) | +14 SE | +14 SE |
| avsett bara vid fångst + faktiskt (en lokal relä av jämförarens −1) | +88 SE | +85 SE |

Den informationen finns passivt i modulens indata vid t+1. Efferenskopian remappar foveán till avsedda rutor (`agent_rl.py:96–109`), och med larm ligger matchningssignalen −1 just där (`agent_rl.py:41–49`). Om kanalen klarar kriteriet avgörs då av tecknet. Med avsett markerat negativt blir resultatet −6 SE, och med samma tecken som faktiskt fönster blir det godkänt. Det har inget med en förenklad modell att göra.

Jämförelsen med 1b-E skyddar inte säkert. 1b-E tränas med facit mot faktiskt fönster och har därför skäl att trycka bort avsikten, så "AST-specifikt" kan klaras av en 1c som bara reläar jämföraren. CLAUDE.md säger redan detta om effcov, men låstexten behöver säga det mer generellt.

En enkel skiljelinje finns. Korrelationen per händelse mellan probens poäng på avsett och på faktiskt fönster är −0,84 för AST-lik kanal (fel = avsett i stället för faktiskt). För kanalerna som bär båda är den −0,00 till +0,35.

Åtgärd:
- (a) Låstexten ska säga att ett ja på "om" betyder att kanalen bär avsedd plats vid fångst. Det betyder inte att schemat har fel.
- (b) Lägg till den korrelationen, eller probens massa på faktiskt fönster vid H5 mot fångststeg utan H5-villkor, som redovisat mått. Den läses som "förenklad modell" bara om den är klart negativ.
- Produktägaren avgör om (b) ska vara kriterium. Jag rekommenderar redovisat.

**2. Liten: det matchade fönstret skiljer sig något från det avsedda (otränad agent, H5 med båda fönstren, n = 1 895).**
Avsett minus matchat:

| Egenskap | Avsett | Matchat | Skillnad (z) |
| --- | --- | --- | --- |
| Överlapp med faktiskt fönster | 1,8 % | 0 % | +9 |
| Iögonfallande föremål i fönstret | 0,063 | 0 | +11 |
| Avstånd till närmaste iögonfallande | 2,44 | 2,62 | −4,7 |
| Täckt efter steget | 0,116 | 0,099 | +4,4 |
| Täckt före steget | 0,109 | 0,099 | +2,6 |
| Avstånd till faktiskt fönster | 3,49 | 3,60 | −2,2 |

Kantavstånd och fönsterstorlek är lika.

Det beror på två saker. `pick_matched` (`probe_1c.py:140–143`) utesluter överlapp med alla kandidater och varje iögonfallande ruta, men det avsedda fönstret får överlappa det faktiska och innehålla iögonfallande föremål som inte var kandidater. Dessutom matchas besökt-status som binär "någon ruta täckt före steget" (`:123`), medan kanalen ser täckningen efter steget.

Följden syns i kalibreringen. Ren sann täckning ger matchat +1,9 SE (testloggen) och ren suddig markör +1,2 SE. Båda stoppas av lockbetet (−16,7 respektive +0,7), men marginalen är liten och går åt det håll som ger falskt ja.

Åtgärd: redovisa tabellen för tränade agenter (blind, den kräver inga probvärden). Gör också en känslighetsanalys som bara tar med händelser där avsett fönster inte överlappar faktiskt fönster och saknar iögonfallande föremål. Det kan göras i efterhand utan att ändra kriteriet.

**3. Liten: probs_ext är rättvis, möjligen konservativ (`probe_1c.py:331–342`).**
Samma poängfunktion används och jämförelsen görs inom samma softmax. Det avsedda fönstret förekommer i varje träningshändelse som negativt exempel (proben tränas bara på fångststeg), medan det matchade fönstret aldrig förekommer. Proben lär sig alltså snarare att trycka ner avsiktsliknande drag. Det går emot ett ja. AST-lik kanal klarar ändå kriteriet med god marginal (+21/+22 SE vid r = 0,6). Ingen åtgärd behövs.

**4. Liten: redovisningen i `eval_main_1c.py` använder fortfarande bara lockbetet.**
- `norepeat` (`:167`), `sens_control` (`:168`) och `measure_pair` (`probe_1c.py:643–666`) bygger enbart på lockbetskontrollen.
- Tolkningsvillkoret ("ett ja läses som AST bara om gruppen utan upprepat sikte pekar åt samma håll") bör gälla båda kontrollerna. Åtgärd: lägg till `split.norepeat.matched` och `h5_sens["criterion"]` i `judge`. Det är en ändring som ska göras före låsningen.

**5. Liten: låsning och kontroller stämmer med CLAUDE.md, med en rest.**
- `judge` kräver förutsättningarna (a)–(c) och `criterion.passed` i alla tre fröna. AST-specificiteten jämför båda kontrollerna mot 1b-E med SE = roten ur summan av varianserna. Kontrollen av checkpoints och sha256-cachen är som beskrivet.
- Träningskoden är oförändrad sedan förregistreringen 3c338d3. Diffen berör bara `probe_1c.py`, `eval_main_1c.py` och `test_probe_1c.py`.
- Rest: `test_probe_1c.py` (kalibreringen) saknas i `lock_hashes` (`probe_1c.py:525`). Den bör läggas till.

## Blindhet (iv)
Inga brott. Sex körningar har final-checkpoints, men `sprint6_main.json` och låsfilen finns inte. `measure`, `c_all`, `measure_pair` och `probs_ext` anropas bara från `eval_main_1c.py` efter `_check_lock()` och från testerna med tillfälligt lås på otränade agenter. Låsningen måste skrivas innan `eval_main_1c.py run` eller något annat skript körs på c6_mL-checkpoints.
