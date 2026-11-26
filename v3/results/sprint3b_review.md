# Sprint 3b – fristående granskning (26 nov 2026)

Granskare: separat Claude-instans som inte skrev koden. Underlag: `results/sprint3b_prereg.md` (inkl. tillägget före
låsning), `extract_tokens.py`, `nexis3/compress.py`, `train_compress.py`, testerna, `sprint3b_train/lock/test/posthoc.json`
och token-datan (`tok_train_*.pt`, `tok_test_*.pt`). Testsviten: `python3 -m pytest -q tests` → 85 klarade, 1 överhoppad.

## Sammanfattning

**Godkänd med anmärkningar – formellt. I sak visar grind 3 mindre än "alla tre klarade" antyder.**

- Proceduren är ren: testämnena är nya, låsningen föregick testet, låsfilens sha256 stämmer, testet kördes en gång,
  inget läckage av facit runt attentionen hittades.
- **K1 är i praktiken sant per konstruktion.** En slumpmässig, otränad P (bara G tränad) klarar K1 (z 12). På den låsta
  modellen slår en enda konstant driftvektor (S_{t+1} = S_t + c, skattad på träning) G lika bra (G ej bättre, z 1,4), och
  en drift per tokenposition slår G klart (z −7,4 till driftens fördel). G har alltså inte lärt sig någon dynamik utöver
  "S flyttar sig på ett typiskt sätt från token till token". S kodar position starkt (linjärt R² 0,88).
- **K3 är svag som nollmodell.** PCA till 4 dimensioner bevarar avstånden bättre än S (ρ 0,66 mot 0,60), och
  tokenpositionen ensam (ett tal) ger ρ 0,56. Slumpprojektionen (0,39) är en låg ribba som nästan vilken tränad
  projektion som helst klarar.
- **K2 är det enda kriteriet med innehåll, men informationen är retrospektiv.** S vid första token (innan svaret) är inte
  bättre än konstant (0,516 mot 0,500). Redan S vid andra raden (attention från den första skrivna svarstoken) ger 0,287 mot
  konstant 0,446. S läser alltså var det skrivna ordet hämtades ifrån (målmening eller lockmening), inte en
  förutsägelse av utfallet. Felsvaren kommer till 87 % ur lockmeningen. Det är verklig information i attentionen, men den
  beskriver vad som redan hänt.

## Fynd

**1. (Allvarlig, tolkning) K1 klaras av triviala modeller.**
Konstruktionen G = S + MLP(S, q) gör identiteten till specialfallet 0, och P tränas själv på ε. Då räcker det att S har en
förutsägbar drift. Kontroller (test, k = 4):

| Förutsägelse av S_{t+1} | ε (samlat) |
| --- | --- |
| identitet | 0,190 |
| låsta G | 0,142 |
| konstant drift c (träning) | 0,132 |
| drift per position (0/1/2/3+, träning) | 0,106 |

Parat per text (som K1): identitet − konstant drift +0,071 (z 26), konstant − G +0,003 (z 1,4), positionsdrift − G −0,022
(z −7,4). Otränad P med slumpvikter, bara G och Q tränade: ε_G 1,02 mot identitet 1,25, **K1 z 12** (både λ = 10 och 0).
Med λ = 0 (bara ε) kollapsar P till en nästan deterministisk kod: ε_G 0,005 mot 1,03 (syns redan i valideringens λ = 0,1:
0,011 mot 0,91). Övergången 0 → 1 (före/efter första svarsordet) står för en stor del: 27 % av variansen i S ligger mellan
t = 0 och senare token. Även för övergångar efter första token är G bättre än identitet (0,149 mot 0,181), men inte bättre
än en positionsdrift. Slutsats: K1 visar inte att G fångar någon dynamik i attentionen.

**2. (Allvarlig, tolkning) K2 mäter avläsning i efterhand, inte förutsägelse.**
Y-huvudet får medel över alla svarstoken av S. Attention från rader där svarsordet redan skrivits pekar på den mening ordet
kopierades ur. Data: på test är 230/240 rätta svar hämtade ur målmeningen, felen ur lockmeningen 43/60 (övriga 17 utan
ordöverlapp); i träningen felen 327/375 ur lockmeningen. Kontroll med låsta P, Y-huvud på S vid en enda position:

| S vid | log-förlust | konstant (samma texter) | n |
| --- | --- | --- | --- |
| position 0 (innan svaret) | 0,516 | 0,500 | 300 |
| position 1 (efter första svarstoken) | 0,287 | 0,446 | 275 |
| position 2 | 0,289 | 0,561 | 173 |

Hela vinsten uppstår när första svarstoken finns. Att S skiljer mål från lock trots att P inte vet vilken mening som är
målet (målens och lockens platser är jämnt fördelade, 75/75/75/75 på test) är intressant: attentionmönstret skiljer sig när
modellen kopierar ur rätt respektive fel mening. Men det är en "vad gjorde jag"-signal, inte en "kommer jag att lyckas".

**3. (Måttlig) Läckage via svaret hittills (del 6) och svarslängd – inte huvudförklaringen.**
Felsvaren är längre (test 7,9 mot 3,5 token, träning 6,9 mot 4,1), och S kodar position, så medel över token kan bära
längd. Kontroller (tränade om med samma `fit`, λ = 10, frö 0, test):

| P ser bara | Y S | K2 z (mot karikatyren) |
| --- | --- | --- |
| allt (replik av låst) | 0,320 | 6,2 |
| del 6, svaret hittills | 0,512 | 2,6 |
| meningar 1–4 | 0,472 | 1,7 |
| allt utom del 6 | 0,401 | 3,9 |
| frågan + övrigt | 0,420 | 1,8 |
| slumpvikter, otränad P | 0,531 | 1,6 |

Del 6 ensam klarar K2 formellt (z 2,6) men är sämre än konstant (0,500) – det visar att K2-jämförelsen är svag, inte att del 6
bär signalen. Utan del 6 klaras K2 ändå. Längd ensam (förregistrerad kontroll) 0,558, sämre än konstant. Inget direkt
facit når P: P ser bara `att`, Y-huvudet bara poolad S (test `test_P_reads_only_attention` stämmer med koden).

**4. (Måttlig) K2:s jämförelse är svag.** Karikatyren är sämre än konstant prediktion på test (0,559 mot 0,500) och
på validering (0,622 mot 0,626). Att slå den säger lite; även en P som ser bara del 6 gör det. Efterhandsjämförelsen mot
konstant (z 5,6) är det relevanta måttet och bör stå i grindrapporten, märkt som efterhand.

**5. (Måttlig) K3:s nollmodell är för svag.** Bootstrap (100 dragningar, 400 token, test), Spearman mot avstånd i A:

| Representation | ρ |
| --- | --- |
| låsta S (k = 4) | 0,60 ± 0,03 |
| PCA, 4 komponenter (träning) | 0,66 ± 0,03 |
| PCA, 2 komponenter | 0,60 ± 0,03 |
| bara tokenposition (1 tal) | 0,56 ± 0,03 |
| karikatyren | 0,39 ± 0,03 |
| otränad P | 0,42 ± 0,03 |
| slumpprojektion (frö 12345) | 0,39 ± 0,03 |

Avstånden i den standardiserade A domineras av positionen (t = 0 skiljer sig från senare token, bl.a. eftersom del 6 är
noll vid t = 0). Att S slår slumpprojektionen följer av att S kodar position. k = 8 klarar inte K3 (0,04 ± 0,03), vilket
grindrapporten bör nämna. K3 visar inte trohet utöver det PCA ger gratis.

**6. (Liten) Låsning och historik – i ordning, en kosmetisk avvikelse.**
Låsfilens sha256 stämmer för alla sju filer (tre modeller, tre kodfiler, förregistreringen). Ordning enligt filtider:
träningsdata 06:51:49, modell k4 06:55:35, `sprint3b_train.json` 06:55:51, lås 06:57:22, testdata till arbetsytan
06:57:32, `sprint3b_test.json` 06:58:18. Commits: förregistrering (dd46fc5, 06:55:04) före modellen, lås (29bc351) före test
(07b9ce8); `test` vägrar köra om resultatfilen finns. Testämnena: 10, ingen överlappning med träningsämnena; frö 8000 mot 5000.
Avvikelser: (a) klockslagen i förregistreringen ("ca 07:10", "ca 07:30") är senare än arbetsytans klocka för samma steg
(06:55–06:57) – troligen uppskattade tider, men de bör rättas eller förklaras; (b) låsfilen hashar inte datafilerna; (c)
`train_compress.py` ändrades (18 rader, `ylength_controls`) efter träningen men före låsningen – tillåtet och redovisat.
Granskningens egna kontroller använde testdatan igen, beskrivande; de påverkar inte grindbeslutet.

**7. (Liten) Kod.** Inga fel som påverkar resultaten hittades.
- `Batch.cur`: `ok[ok.clone()] &= …` ger rätt mask; övergångar stannar inom text (prövat i test och på riktiga data, 1 021
  övergångar på test = Σ(T−1)).
- `pool`: `S[b.pos == 0]` ger en rad per text i textordning eftersom varje text har minst en token; rätt.
- `eps_terms`: nämnaren är variansen över alla token (även mellan texter), samma för G och identitet; rätt enligt prereg.
  Notera att normeringen inte hindrar att P väljer en lättförutsägbar kod (fynd 1).
- `split`: 768 träning, 292 validering, 440 texter med ett valideringsämne som icke-mål kastas; ingen överlappning.
- `caricature`: lager 9–17, medel över huvuden, normerad över fyra meningar; enligt prereg.
- `ylength_controls`: längd klipps vid 12 och "= 2"-flagga; tränas på träning med tidigt stopp på validering. Att S + längd
  blir sämre än S (0,396 mot 0,320) tyder på att Y-huvudets tidiga stopp är brusigt med 292 valideringstexter.
- `extract_tokens`: svaret avkodas och tokeniseras om (`m.answer` → `ans_ids`), så de analyserade svarstoken kan skilja sig
  något från de genererade. Påverkar inte slutsatserna.
- `k3`: `diff_boot_sd` blandar textomdragning och tokenurval; rimligt, men nollmodellen är problemet, inte spridningen.

## Kontroller som kördes (skript i /tmp, inte incheckade)

1. Ämnesöverlapp, textantal, svarslängd och svarskälla (mål/lock) per rätt/fel, mål- och lockplatser.
2. K1 på låst k = 4: identitet, G, konstant drift, positionsdrift; parat per text; positionsavkodning ur S (R² 0,88);
   varians mellan t = 0 och senare (27 %); ε uppdelat på övergång 0 → 1 och senare.
3. Nyträning med samma `fit` (λ = 10, frö 0): replik, otränad P (fryst), P på del 6 / meningar / utan del 6 / fråga + övrigt,
   samt λ = 0. Repliken återger låsta siffror (Y S 0,320, K1 z 24,8, K2 z 6,2, K3 0,20).
4. Y ur S vid enskild position (0, 1, 2).
5. K3 mot PCA, karikatyr, otränad P, slumpprojektion och bara position.

## Förslag till tolkning i grindrapporten

- Grind 3 är formellt klarad enligt de låsta kriterierna, utan procedurfel.
- **K1** visar inte att G modellerar attentionens dynamik: en konstant drift gör lika bra och en drift per position bättre;
  även en otränad P klarar kriteriet. "Kommutering" är alltså inte visad i någon meningsfull mening.
- **K3** visar inte trohet utöver vad PCA (och tokenposition) ger; slumpprojektionen var en för låg ribba. k = 8 klarar inte K3.
- **K2** är verkligt och överraskande starkt: en liten komprimering av attentionen skiljer rätt från fel bättre än konstant
  (efterhand z 5,6) och mycket bättre än den handvalda karikatyren. Men signalen finns först när första svarsordet skrivits
  och avslöjar ur vilken mening ordet hämtades. Det är en självavläsning i efterhand ("jag kopierade ur lockmeningen"),
  inte en självmodell som förutsäger utfallet. Värt att följa upp: vad i attentionen skiljer kopiering ur mål från
  kopiering ur lock när P inte vet vilken mening som är målet.
- Före nästa sprint: K1 bör jämföras mot drift per position (eller kräva G bättre än den), K2 mot konstant och bara på
  information före svaret (eller per position), K3 mot PCA med samma k.
