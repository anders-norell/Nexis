# Sprint 6 (1c): granskning av huvudkörningens utvärdering före låsning

Granskare: separat agent, 25 okt 2026. Underlag: `eval_main_1c.py`, `probe_1c.py`, `run_sprint6.sh` (fas M), `test_probe_1c.py`, CLAUDE.md, `sprint6_build_review.md`.
Jag har bara kört otränade agenter (frö 30–32, fångstrisk 0,5, full testbank 4 000 och probbank 4 000) och syntetiska kanaler. Ingen tränad checkpoint är laddad. Från piloten har jag bara läst händelseantal i `sprint6_pilot.json`.

## Utlåtande: **ej låsbar** (fynd 1 måste lösas; resten är små)

## Stämmer med beslutet
Följande är kontrollerat och stämmer:
- **Parametrar i fas M:** fångstrisk 0,5, kostnad 0,0001 (fast), `--max-episodes 1000000` (det sista argumentet vinner), frö 30–32 och fem villkor. Entropin sänks över 300k som i fas E.
- **1b-E:** körs som i 5a (`eref1e-4` = `--schema channel_e --rate-cost 0.0001`, mål 0, facitförlust via `channel_grads`). Enda skillnaden är fångstrisken, och den är beslutad.
- **Bankerna:** testbank 2 000 000 + i och probbank 7 000 000 + i, båda med 4 000 episoder, och slutcheckpoints.
- **Kriterierna (a)–(c):** oparat SE. (c) bootstrappas över episoder som kvot av summor.
- **Låtsasfönstret:** fel kandidat, inte avsett fönster, `L[act,j]=0`.
- **`next_overlap`:** beräknas på kommando t+1, och överlappsvillkoret är rätt.
- **AST-specificitetens SE:** roten ur summan av fyra varianser. Det är konservativt, eftersom tränad och otränad agent delar episoder och har positiv kovarians.
- **Den otränade referensen:** `load_untrained` ger exakt startvikterna från träningen (samma `make_agent*`, samma frö, cfg ur `m_args`, fångstrisk 0,5).
- **Blindhet:** inga brott. `measure`/`c_all` vägrar köra utan lås.

## Fynd

**1. Allvarlig: rollbyteskontrollen förväxlar roll med iögonfallande (`probe_1c.py:226–229`, `test_probe_1c.py:117`).**
Låtsasfönstret är alltid ett iögonfallande föremåls fönster. Det avsedda fönstret är det nästan aldrig, eftersom agenten inte siktar på iögonfallande föremål. En kanal som bara kodar iögonfallande, utan någon information om roll, klarar därför kontrollen. Modulen får dessutom periferins iögonfallande-kanal som indata.

Syntetiskt prov (markör på faktiskt fönster med sannolikhet r, plus +amp på alla iögonfallande kandidater):

| | amp 1 | amp 2 | amp 3 |
| --- | --- | --- | --- |
| r = 0,6 | +0,9 SE | +3,8 SE | |
| r = 0,5 | | +6,5 SE | +14,9 SE |
| r = 0,7 | | | +7,4 SE |

Tecknet beror på hur proben tränas: tränas den bara på fångststeg blir det −29 SE. Att dra ifrån den otränade agenten tar inte bort felet, eftersom träningen kan förstärka kodningen av iögonfallande av skäl som inte har med avsikt att göra (iögonfallande förutsäger fångst). "om" kan alltså bli sant per konstruktion.

Åtgärd:
- (a) Lägg in fallet "iögonfallande-karta" i kalibreringen, med krav |z| < 2.
- (b) Inför ett kontrollfönster som liknar det avsedda i iögonfallande: ett lottat fönster som inte är en kandidat, saknar iögonfallande föremål och har samma begränsade innehållsklass som det avsedda. Det läggs till som extra kandidat vid test (proben är rollblind och kan värdera vilket fönster som helst). Kriteriet blir p(avsett) − p(kontrollfönster).
- (c) Alternativt: residualisera kanalsärdragen mot fönstrets observerade iögonfallande innan proben tränas. Det ska anpassas på steg utan fångst.
- Produktägaren beslutar före låsningen.

**2. Medel: den otränade referensen är brusig och har en annan händelsefördelning (`eval_main_1c.py:105–109`).**
Mätt kontroll för otränade agenter (SE 0,0002–0,0013):

| Villkor | Frö 30 | Frö 31 | Frö 32 |
| --- | --- | --- | --- |
| Med larm | +0,0095 (+7,4 SE) | −0,0030 (−9,5) | −0,0058 (−12,9) |
| 1b-E | nästan identisk med "med larm" i alla frön | | |
| Utan larm | +0,0032 | +0,0005 | +0,0005 |
| Kapacitetsbaslinje | −0,0003 | 0,0000 | −0,0001 (alla \|z\| < 2) |

Den passiva komponenten byter alltså tecken mellan frön. Den är en egenskap hos slumpvikterna, inte en stabil "vidareföring". Variationen mellan probfrön är försumbar (±0,0003).

Händelsefördelningen skiljer sig också. Otränade agenter: träff 24 %, 6,0 händelser per episod, andel K ≥ 3 = 0,769, låtsasfönster i 55 % av fallen, upprepat sikte 11 %. Tränade (pilotens dcL): ca 3,5 händelser per episod och K ≥ 3 ca 0,80.

Åtgärd: mät referensen som en **skuggmodul**. Den tränade agenten kör episoderna, och en otränad modul med samma frö får samma observationer parallellt utan att policyn ser den. Då blir händelserna identiska och jämförelsen parad per händelse. Redovisa också fördelningsmåtten ovan för både tränad och otränad agent.

**3. Liten: låsningen täcker inte allt (`probe_1c.py:445–448`).**
`run_sprint6.sh` saknas i `lock_hashes`, men den styr träningsargumenten. Åtgärd: lägg till filen. Låt dessutom `evaluate_run` kontrollera mot `PARAMS` att varje checkpoints `m_args` och `m_episodes` stämmer (fångstrisk 0,5, `rate_cost` 0,0001, `rate_target` 0, ≥ 998 400 episoder). Annars skulle en körning som stoppats av `--minutes 240` passera tyst.

**4. Liten: cachning per körningsnamn (`eval_main_1c.py:85`).**
En omtränad körning utvärderas inte om, samma problem som i `eval_mvp`. Åtgärd: spara checkpointens sha256 i resultatet och jämför vid omkörning.

**5. Liten, att notera: AST-specificiteten blir i praktiken tränad 1c minus tränad 1b-E.**
De otränade referenserna för 1c med larm och 1b-E är nästan identiska, så de tar ut varandra. Det är rättvist, men låstexten bör säga det.

**6. Liten, att notera: "utan larm" jämförs i (b) med kapacitetsbaslinjen med larm (`eval_main_1c.py:118`).**
Det är dokumenterat och konservativt. Det ska stå i låstexten.
