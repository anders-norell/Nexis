# Nexis — överlämning mellan sessioner

Läs den här filen först i varje ny session. Uppdatera avsnittet **Status** och pusha vid varje sprintgrind.

## Projektet i korthet

Nexis prövar Michael Grazianos Attention Schema Theory (AST) i små artificiella agenter.
Produktägare: Anders Norell. Byggs av Claude. All kommunikation med produktägaren sker på svenska.

**Avgörande fråga (v2):** kommer ett systems kunskap om sin egen uppmärksamhet från en förenklad
modell (schemat) och inte från uppmärksamheten själv? Prövas på två sätt, som är MVP:ns kärna:

1. **Falskt schema (H2):** flyttar agenten sin verkliga spotlight när schemat ljuger?
2. **Avsedd kontra faktisk attention (H5):** när spotlighten fångas av en distraktor, anger schemat
   den *avsedda* platsen hellre än någon annan felaktig plats?

Att ett schema hjälper styrningen är redan visat (Wilterson & Graziano 2021, Piefke m.fl. 2024).
Det nya är de två punkterna ovan.

## Var saker finns

| Vad | Var |
| --- | --- |
| Gällande spec (levande, redigerbar) | https://claude.ai/artifact/GsEmfNtb1vs2bhXxwc6vsW — "Nexis v2 – spec" |
| Ordlista (begrepp i vardagsspråk) | https://claude.ai/artifact/2syuVpzf2GcAfaP9DgSgjC — "Nexis – ordlista" |
| Ögonblicksbild av specen | `docs/spec_v2.md` (uppdateras vid varje grind; det levande dokumentet gäller vid skillnad) |
| v1: spec, kod och resultat | `docs/spec.md`, rotkatalogens `*.py`, `results/` — rör inte, det är ett avslutat försök |
| v2: kod | `v2/` (skapas i sprint 1) |
| v2: resultat | `v2/results/` |

Specen läses med docs-verktygen (inte web-fetch). Produktägaren svarar ibland på öppna frågor direkt i
specen; kontrollera ändringar sedan förra sessionen innan arbetet börjar.

## Arbetsregler

- **Sprintar med grindar.** Varje sprint avslutas med att produktägaren godkänner resultatet mot specen
  innan nästa påbörjas. Påbörja aldrig nästa sprint utan godkännande.
- **Trösklar låses före körning** (i sprint 3, efter piloten) och ändras aldrig efter att resultat setts.
  Signifikanskriterium: skillnad > 2 standardfel (SE för skillnaden, över testepisoderna inom fröet) och
  samma riktning i alla frön. (Rättat 22 okt från "2 SD"; beräkningarna har alltid gjorts med SE.)
- **Alla resultat rapporteras**, även misslyckanden. Ett godkänt resultat visar att mekanismen beter sig
  som AST förutsäger, inget om upplevelse.
- **Oberoende granskning:** vid varje grind granskar en separat agent kod och resultat mot specen.
- **Doskontroll (produktägarens lärdom vid grind 4, 23 okt):** innan ett ingrepp eller en kontroll låses ska
  ett test visa att ingreppet ändrar ungefär lika mycket indata i kontrollen som i huvudvillkoret. Sprint 4:s
  Blandat-kontroll för H2 (a) var nästan tom (0,76 mot 6,1 ändrade rutor) utan att tester eller granskning såg det.
- **Pusha ofta:** arbetsytan töms mellan sessioner. Allt som inte är pushat går förlorat.
- **Långa körningar:** kör i bakgrunden (upp till ca 2 h per kommando). Logga till fil, inte bara stdout.

## Teknisk miljö

- Arbetsytan har 2 CPU-kärnor, Python 3, NumPy, Matplotlib.
- **PyTorch går inte att installera** (paketkällan blockeras, kontrollerat 21 okt 2026). v2-agenten har
  handskriven backprop i NumPy (`v2/agent_rl.py`, testad mot finita differenser). Träning: ca 18 000
  episoder/min per kärna; kör frön parallellt med `OMP_NUM_THREADS=1`.
- Stopp-kroken kräver att allt är pushat när ett svar avslutas. Under träning ändras loggarna hela tiden;
  committa dem och fortsätt.
- v1 tränades med övervakad förlust och mjuk attention. v2 kräver förstärkningsinlärning med hård
  spotlight (t.ex. REINFORCE med baslinje eller A2C). Det är långsammare och nyckfullare; räkna med
  att sprint 2 kan kräva flera försök.
- Fas 2 (GPT-2 via TransformerLens) kräver PyTorch och är blockerad tills det lösts. Den körs bara om
  fas 1 visar effekt.

## Lärdomar från v1 som styr v2

- Uppgiften måste *kräva* schemat. I v1 kunde kontrollagenten sluta sig till var attention hamnat
  utifrån vad den såg, så schemat gav liten eller ingen effekt. Kontrollera detta redan i piloten.
- Trösklar ska bygga på pilotdata, inte på önskad effektstorlek.
- v1:s mest AST-specifika fynd (schemat drar mot avsedd plats, 31 % mot ca 7 % väntat) var en
  bifångst. I v2 är det förregistrerad hypotes H5.

## Sprintplan (MVP)

| Sprint | Innehåll | Grind |
| --- | --- | --- |
| 1. Miljö | Sökmiljö: rutnät, periferi (färg, ej form), spotlight med fast fönster, distraktorer, ledtrådsnivå "svag", fångstmekanism (spotlighten dras ibland till iögonfallande distraktor), slumpvisa episodparametrar. Enhetstester och slumpagent. | Miljön beter sig enligt spec |
| 2. Kompetens | Baslinjeagent utan schema lär sig söka klart bättre än slumpagenten. | Kompetens visad |
| 3. Pilot | Tid per körning, spridning mellan frön, kontroll att schemat behövs. Trösklar och körplan föreslås. | Trösklar godkända och låsta |
| 4. MVP | Baslinje, Fast schema (1a), Blandat schema; frön 10–12 (9 körningar à 1M episoder). Låsta kriterier F, H1, H2, H3a/H3b (alternativ A, grind 3). | Resultat bedömda mot låsta kriterier |
| 5. H5 | 5a: diagnostik av H2-asymmetrin, schemamodul (1b) med matchningssignal och flaskhals (dosserie), pilot. 5b: huvudkörning. | 5a: H5:s trösklar låsta. 5b: bedömd mot dem |
| 6. Utbyggnad | Zoom, skalning, övriga ledtrådsnivåer, 1c, fem frön, sedan social fas. | Enligt resultat |

## Status

- **Senast uppdaterad:** 26 okt 2026 (steg 0 klart)
- **Sprint 0 (spec):** klar. Spec v2 godkänd i huvudsak av produktägaren.
- **Sprint 1 (miljö):** **klar, grind 1 godkänd 21 okt 2026** (produktägaren delegerade bedömningen till Claude).
  Kod: `v2/env.py`, `v2/agents.py` (slump-, avsöknings- och styrd referensagent; den styrda kan spåra
  avsedd, härledd eller faktisk plats), `v2/test_env.py` (46 tester), `v2/check_env.py` → `v2/results/sprint1_env_check.*`.
- **Sprint 2 (kompetens):** **klar, grind 2 godkänd av produktägaren 22 okt 2026.**
  - Grind (låst 21 okt före träning): K1 träffandel > slump med > 2 SE, K2 ≥ 73 % (bästa minneslösa
    referens 71,4 %), alla tre frön, 4 000 testepisoder per frö, stokastisk policy.
  - Resultat: 89,0 / 89,9 / 88,5 % (slump ≈ 28 %, z ≈ 70). Söklutning utan fångst 0,28 steg/distraktor.
    En körning per frö (1 M episoder, ca 49 min, tre parallellt), inga kasserade försök.
  - Oberoende granskning: godkänd med anmärkningar, inget läckage (`v2/results/sprint2_review.md`).
  - Kod: `v2/agent_rl.py` (retinotop prioritetskarta: GRU per ruta med delade vikter, efferenskopia
    remappar foveán till avsedda rutor; schemakanal 2 kanaler/ruta och 1b-huvud inbyggda men avstängda;
    handskriven backprop, float32), `v2/train_rl.py` (A2C, GAE, återupptagbar), `v2/test_agent_rl.py`
    (15 tester inkl. finita differenser), `v2/metrics.py`, `v2/references_sprint2.py`, `v2/eval_gate2.py`,
    `v2/run_sprint2.sh`. Resultat: `v2/results/s2_base_s{0,1,2}.*`, `sprint2_gate.json`, `sprint2.png`.
  - Schemavillkor är konfiguration: `train_rl.py --schema fixed|learned|shuffled --sl <vikt för 1b>`.
    Öppen designpunkt för 1b: huvudet förutsäger förra stegets faktiska fönster och matas in med ett
    stegs fördröjning; avgörs i sprint 3.
- **Produktägarens linje:** experimentet ska vara så biologiskt och evolutionärt realistiskt som möjligt, så
  länge det är förenligt med experimentets komplexitet. Väg in det i designval.
- **Beslut 21 okt:** (1) efferenskopia av föregående kommando till alla villkor (`obs["efference"]`);
  (2) graderad fångst nerifrån: 0–3 iögonfallande distraktorer, brusig salience i periferin, fångstchans
  ∝ salience; (3) inlärt schema 1b ingår i MVP, H5 mäts på 1b. 1c (utan facit) efter MVP. Zoom väntar
  om den ökar komplexiteten för mycket.
- **Beslut 22 okt:** ordningen efter MVP blir 1c för en ensam agent, sedan en social fas med flera agenter
  som måste förutsäga varandras uppmärksamhet (produktägarens tes: självmodellen är nödvändig av
  spelteoretiska skäl). Social fas startar först när 1c visat något. Skiss i specen. Fas 2 oförändrad.
- **Varning inför sprint 3 (viktigast):** baslinjen utan schema når ca 89 %, perfekt schema-referens
  93,5 %. Schemat har bara ca 4 procentenheter att tillföra — v1:s fälla. Piloten ska först träna 1a och
  mäta vinst samt om kanalen läses (lesion, blandat schema). Under ca 3 punkter: flytta arbetspunkten
  innan trösklar låses (bara fönster 1: gap ca 5; högre fångstrisk: gap ca 10 vid p_capture 0,6; eller en
  fovea som avslöjar mindre). Det är en specändring som produktägaren beslutar. Agenten siktar aldrig på
  iögonfallande objekt; H5:s nollmodell måste ta hänsyn till det.
- **H5 mätvillkor (spec, låses i sprint 3):** bara fångster där foveán utesluter avsedd plats; schemat avläses
  steget efter fångsten; fångster som hittar målet utesluts; nollhypotes = idealisk betraktares gissning med
  samma indata samt övriga iögonfallande platser, inte ren slump. Annars blir H5 sann per konstruktion.
- **Inriktning 22 okt (produktägaren):** begreppsbildning ska ingå i det som driver fram ett schema (förslag:
  diskreta symboler i schemakanalen i 1c och den sociala fasen, matade tillbaka som inre tal). Grundmotiv:
  en fullständig funktionalism som inte alienerar oss från oss själva och världen. Se specen.
- **Styrning (överenskommet 22 okt):** produktägaren tycker att Claude styr mycket. Därför: före varje sprint
  skriver Claude planen i klartext i specen (vad, varför, vad som händer om svaret blir nej) och väntar på
  produktägarens godkännande innan något körs. Skilj tydligt på produktägarens beslut och tekniska val.
- **Sprint 3 (pilot): plan skriven i specen 22 okt 2026, väntar på produktägarens godkännande. Inget kört.**
  Avsnittet "Sprint 3: pilot – plan" i specen: fem frågor (vinst ≥ 3 p för 1a vid någon tillåten
  arbetspunkt, kanalen läses, spridning, H5 mätbar, tid), steg 1–5, förregistrerad regel för val av
  arbetspunkt (närmast standard: mest avslöjande fovea, lägst fångstrisk, båda fönstren), H6-alternativ
  A–D (Claude rekommenderar A: variera foveán, H6 beskrivande, helt blind fovea utesluten eftersom H5 då
  blir omätbar), fem beslutspunkter för produktägaren och Claudes tekniska val.
  Kärninsikt: en träff avslutar episoden, så foveáns innehåll duger bara till att lokalisera spotlighten
  mot periferins karta. Foveán läcker alltså schemats information.
- **Beslut 22 okt (produktägaren):** planen godkänd i alla fem punkter, H6 = alternativ A. Claude får välja
  väg mot "lyckat utfall" = giltigt och informativt experiment (inte bekräftade hypoteser). Regeln för
  arbetspunkt oförändrad.
- **Steg 1 klart 22 okt:** `EnvConfig.reveal` (maskerad blick: fönsterflaggor kvar, occupied 0,5; reveal=1
  bitidentiskt med sprint 2), `train_rl.py --reveal --p-capture --windows` (args sparas i checkpoints),
  `h5.py` + `test_h5.py` (idealisk betraktare, H5-händelser; slumpagent: 97 % av fångsterna utesluter
  avsedd plats), `pilot_eval.py` (intakt/lesion per körning → `results/sprint3_pilot.json`).
- **Steg 2–5 klara 22 okt (natten, produktägaren sov).** Allt står i specens sprint 3-avsnitt.
  - Arbetspunkt = standardinställningarna (regeln). 1a +7,2–8,2 p över baslinjen i frö 0–3, lesion 48–58 %.
  - H6 utan stöd: vinsten oberoende av foveán (blind fovea lika bra för baslinjen). Min foveá-förklaring var fel.
  - Agenten använder bara kartan "täckt hittills" i schemat (nollad → ~80 %), inte "fönstret förra steget".
  - H1: lutning över alla episoder speglar bara träffandelen (granskaren). Rättat mått = steg bara i
    hittade episoder, med fångst; frö 1: 1a 0,35, baslinje 0,38, Blandat 0,36 → H1 får troligen inte stöd.
  - 1b (delat huvud, vikt 1/0,3/0,1, samt separat schemamodul `--schema module`) ger ingen vinst. Trolig
    orsak: foveán remappas till *avsedda* rutor, fångst svår att upptäcka → H5 riskerar att följa av
    arkitekturen. H5-utfallet ej inspekterat (bara antal händelser).
  - Förslag (väntar på produktägaren vid grind 3): alternativ A = MVP med Baslinje/1a/Blandat, F+H1+H2+H3,
    1M episoder, frön 10–12; H5 egen sprint med matchningsinformation till schemamodulen. B och C i specen.
  - H2 riktas mot kartan "täckt hittills" (falskt obesökt / falskt besökt vid steg 3). H2-kod ej skriven än.
  - Live-loggar ignoreras av git; `v2/sync_logs.sh` kopierar färdiga körningar till `v2/results/logs/`.
- **Granskning klar:** godkänd med anmärkningar (`v2/results/sprint3_review.md`). Att göra i sprint 4 innan
  körning: bank med ett frö per testepisod (parning gäller nu bara första omgången om 500), precisera H2
  (lögnens varaktighet, 2×2-regel, för få målfärgsrutor, Blandat som kontroll). 1b-siffror i `sprint3_1b.json`.
- **Grind 3, produktägarens villkor (22 okt kväll), införda i specen:** (1) H2 och fröbanken preciserade
  före låsning (H2: lögn i kartan "täckt hittills" inför fjärde kommandot, varar resten av episoden,
  (a) falskt obesökt / (b) falskt besökt ≤ 3 målfärgsrutor, ej målet; "siktar dit" = avsett fönster
  överlappar; mått = antal av 3 följande kommandon, med minus utan ingrepp, parat; godkänt > 2 SE åt
  förutsagt håll i alla tre frön för (a) och (b), och 1a:s effekt > Blandats med > 2 SE. Fröbank: 4 000
  episoder, frö 2 000 000 + i, separata strömmar för miljö och agentens val). (2) H3 omformulerad:
  H3a = lesion under hela träningen = Baslinjen vs 1a; H3b = akut lesion vid test; båda krävs.
  (3) "2 SD" rättat till standardfel. Ordlistan (se tabellen överst) uppdaterad på tre punkter.
- **Grind 3 godkänd 22 okt 22:30** ("Jag följer dina rekommendationer"): alternativ A, H1 med rättat mått,
  H3 = H3a + H3b (ingen slumpvis avstängning under träning). **Trösklarna är låsta** (specens tröskeltabell,
  signifikanskriterium, H2 i detalj, fröbank, H3). Ändringar före körning kräver produktägarens godkännande.
- **Sprint 4-plan godkänd av produktägaren 22 okt 22:44** (bockad + "svar: Y" i specen). **Inget kört ännu.** Steg: (1) fröbank + H2-ingrepp
  + testskript, H2-kod prövas aldrig på pilotens agenter; (2) 9 körningar, frön 10–12, 1M ep; (3) bedömning;
  (4) granskning, grind 4. Förregistrerat: körning < 73 % träff görs om en gång med frö + 1000.
- **Sprint 4 pågår (startad 22 okt 22:45).** Steg 1 klart och pushat 22:54 (commit 4b550d4): `v2/bank.py`
  (fröbank 2 000 000 + i, strömmar env/agent + h2 för H2:s val; H2 (a)/(b)/(c); Blandat-kontroll med
  givare = nästa episod i blocket om 500), `v2/test_bank.py` (13 tester, bara otränade agenter),
  `v2/eval_mvp.py` (låsta kriterier, oparat SE utom H2), `v2/run_sprint4.sh` (återupptagbar; kör om
  med `sh run_sprint4.sh` om bakgrundskommandot dött). `env.py` fick två slumpkrokar, bitidentiskt.
  Tekniska val står i specen under steg 1.
- **Sprint 4 klar 23 okt ca 01:30, grind 4 väntar på produktägaren. Inget nytt får köras före beslut.**
  - Nio körningar (frö 10–12, 1 M ep, 22–24 min styck), ingen < 73 %, ingen omkörning.
    Resultat `v2/results/sprint4_mvp.json`: **F klarat** (1a +6–7,5 p mot Baslinje och Blandat, ≥ 9,6 SE),
    **H1 ej klarat**, **H2 klarat** ((a) +0,07–0,22, (b) −0,81–0,84 kommandon av 3; 1a ≫ Blandat),
    **H3a och H3b klarade**. (c) falskt fönster (redovisas): 1a −0,07.
  - Granskning: godkänd med anmärkningar (`v2/results/sprint4_review.md`). Stor anmärkning: Blandat-kontrollen
    i H2 (a) var nästan tom (rutorna = egen täckning, sällan markerade i givarens karta). Tilläggsanalys
    (`v2/supp_h2a.py`, `bank.run_bank(a_cells="shown")`, `sprint4_supp_h2a.json`): med verklig lögn reagerar
    Blandat ±0,02–0,05, 1a fortfarande > 10 SE större. Andra: (a) större på slutcheckpoints; (b) delvis
    mekaniskt (−0,29 per kommando); `eval_mvp.py` cachar per körningsnamn.
  - Grind 4-frågor i specen: godkänna resultatet; sprint 5 (H5) som nästa steg, plan först.
- **Grind 4 godkänd 23 okt 10:48** (produktägaren, i chatten). Hans bedömning, införd i specen: H2 (b) är det
  starkaste fyndet, men i 1a har schemat aldrig haft fel i träningen, så det visar att schemat behandlas som
  auktoritet, inte att det är en förenklad modell (H5:s och 1c:s uppgift). H3 är svagare än det ser ut
  (H3a:s träffhalva = F; återbesök delvis per konstruktion; H3b utanför träningsfördelningen), H2 bär
  resultatet. Asymmetrin i H2 är oförklarad och ska diagnostiseras före H5-piloten.
- **Villkor för sprint 5 (produktägaren):** planen ska hantera risken att H5 blir falsk per konstruktion
  (modul med matchningssignal tränad mot faktisk plats pekar på faktisk plats). Jämför med en variant med
  flaskhals som måste förenkla, och ta in H2-asymmetrin som underlag.
- **Plan 5a reviderad 23 okt ca 12:30 efter nio kommentarer från diskussionstråden; väntar på produktägarens
  godkännande. Inget kört.** Avsnitt "Sprint 5: H5" i specen (svar postade i alla nio trådar):
  (1) diagnostik på sprint 4:s agenter, bara rutor i målets färg: E1 eget minne av sikte och synintryck (E2 sammanslagen),
  E3 rimlighet (bara steg 3 ljuger, delat efter fångst, jämfört med samma rutor i hela (a)), E4 asymmetrin i måttet
  (konkurrerande obesökta målfärgsrutor, mått per ruta normerat), plus slutcheckpoints och (b) per kommando;
  (2) bygge: matchningssignal (upptaget + färg, bara till modulen); **flaskhals = kostnad per informationsmängd på
  modulens inre representation, full indata**, avkodaren ser bara kanalen (ett brusigt tal per ruta, bitar summerade
  per steg); nivåer **50/25/12,5 % av referensens informationsmängd** (referens = variant "utan kostnad" med mycket
  liten fast kostnad, medel över två frön; vikten regleras under träning mot nivån); ändrat H5-mätvillkor
  (utesluta med upptaget och färg); doskontroll inkl. gruppstorlek för beteendemåttet; nollmodell, beteendemått och
  H2 (b) i modulens karta byggs och testas på otränade agenter;
  (3) pilot: Baslinje, 1b utan kostnad (först), 1b 50/25/12,5 %, frö 20–21, 400k, pilotbank 4 000 000 + i, blind för
  var felen pekar; (4) trösklar, nollmodell och beteendemått låses vid grind 5a.
  **Nollmodell (Claudes justering av trådens förslag):** försvagad ideal betraktare med samma träffsäkerhet vid fångst
  som schemat (sannolikhet för glimten upphöjd till faktor 0–1, vald bara på träffsäkerhet); trådens form ger noll på
  avsedd plats vid H5-händelser och vore automatiskt godkänd. AST-signatur = schemats felmassa på avsedd plats >
  nollmodellens. **Beteendemått:** inom agent, fångster där schemat har fel (mest sannolika ruta i avsett fönster) mot rätt,
  matchat på ideala betraktarens sannolikhet, bara när avsedd ruta har osett målfärgsföremål; godkänt färre återvändanden
  > 2 SE alla frön. **H2 (b) på 1b i 5b** (svar på orakelinvändningen), meningsfullt bara om 1b-agenten läser schemat.
  Fem beslutspunkter: planen, flaskhalsens form och plats, beteendemått som kriterium, H2 (b) på 1b, nollmodellen.
  Ordlistan: matchningssignal, dosserie, doskontroll, flaskhals (uppdaterad), nollmodell.
- **Diskussionstrådens förslag 22–23 okt införda i specen 23 okt ca 11:45** (produktägaren skickade texten):
  skärpt tolkning av sprint 4 (1a = orakel; MVP visar schemat som auktoritet, inte förenklad modell; AST:s
  kärna prövas i H5 och 1c; H3 tillför lite eget; asymmetrin = vad agenten behandlar som auktoritet), och nytt
  avsnitt under "Efter MVP" (sammanhållande fråga, Rogers/Minsky/Dennett–Harari, tre prövbara frågor: inre
  kapning, schemat som samordning, från modell till berättelse), märkt ej beslutat. Kommentaren vid grind 3
  är överspelad (produktägarens egen markering); svar postade i de två kommentarstrådarna.
- **Plan 5a godkänd med villkor 23 okt 12:04** (produktägaren, i chatten; alla fem beslutspunkter enligt
  rekommendation). Villkor införda i specen före bygget: (a) nollmodellen försvagas genom blandning (varje signal
  byts med sannolikhet q mot slumpvärde, q matchas mot schemats träffsäkerhet), inte exponent (0 förblir 0);
  syntetiskt test före låsning; (b) referensens fasta kostnad låst 0,001 per bit och steg, även 0,0001; skiljer
  sig informationsmängden > 1,5× används minsta informationsmängd med träffsäkerhet över alla steg inom 1 p av
  0,0001-varianten (= förutsättningskontrollen); (c) riskstycket: ren komprimering drar också mot avsedd plats,
  ett ja på H5 är en följd av AST, inte unikt; det AST-specifika prövas i 1c.
- **Steg 1 (diagnostik) klart 23 okt ca 12:40:** `v2/diag_h2.py` → `results/sprint5_diag_h2.json`. Återskapar låsta
  H2-siffror exakt. Iögonfallande föremål uteslutna (teknisk rättelse efter 500-episodkontroll). E3 och E4 förklarar
  inte asymmetrin, E1 oavgjord (golv): normerat stänger (b) 93–95 % av gapet, (a) 10–33 %. E1-provet svagt (fångade rutor
  ligger intill iögonfallande föremål). Steg 3-rutor 2–3× större effekt (närhet eller färskhet, oavgjort).
  (b) per kommando −0,29, alla tre kommandon −0,95. Beteendemåttet mäts som det står (ingen delning/normering).
- **Steg 2 (bygge) klart och pushat 23 okt 12:22 (commit 003308e), inget tränat före push.** `v2/schema_channel.py`
  (ChannelSchema: modul med full indata + matchning, brusig kanal ett tal/ruta, bitar = KL mot N(0,1), avkodare ser
  bara kanalen; channel_grads BPTT), `agent_rl.py` (matching_signal, läge "channel"; gamla agenter bitidentiska),
  `train_rl.py` (--schema channel --rate-cost --rate-target --rate-eta), `h5.py` (observer_restricted,
  excludes_intended_restricted, observer_mixed = nollmodell villkor a), `h5_bank.py` (run_bank_module: H5-händelser,
  träffsäkerhet, bitar, beteendegrupper bara som ja/nej ≥ 200, H2 (b) i modulens karta), `pilot_5a.py`
  (ref/all → results/sprint5a_pilot.json), `run_sprint5a.sh A` / `B T50 T25 T12`. Tester: `test_schema_channel.py`,
  `test_h5_5a.py`.
- **Steg 3 (pilot) fas A klar 23 okt 13:17; FÖRUTSÄTTNINGEN KLARAS INTE; piloten stoppad, fas B ej startad.
  Väntar på produktägarens beslut (specen, "Steg 3: pilot – resultat": A/B/C/D).** 1b med full info (kostnad 0,0001)
  rätt vid 20–21 % av H5-händelserna (krav ≥ 50 %; begränsade betraktaren 0,94 på faktiskt fönster). Rätt utan fångst
  99,9–100 %, vid fångst 11–12 % (fångst ≈ 1/3 av stegen). Referensen ej väldefinierad: 78 bitar (0,001) mot 183
  (0,0001), faktor 2,4 > 1,5. Ingen nytta: 1b 86,3–87,2 % träff, Baslinje 86,6–86,8 %. Förlusten planar ut ~200k.
  Tolkning: informationen finns men modulen hittar inte fångsten (trolig orsak: rutorna kommunicerar bara via globalt medel; vart felen pekar är inte mätt).
  Förslag: A (rek.) lokal "var passar glimten"-signal per ruta; B global max-samordning; C längre träning (avråds);
  D avsluta H5 i denna form. Blindhet bevarad: var felen pekar ej tittat på.
- **Beslut 23 okt 14:03 (produktägaren): alternativ E** (Claudes förslag efter frågan om mest evolutionärt sannolik
  silvertejpslösning): global larmsignal = matchningssignalen ihopräknad till en siffra ("något stämde inte där jag
  siktade"), sänd till alla rutor i modulen (`agent_rl.alarm_signal`, läge `channel_e`). Allt annat oförändrat.
  Byggt, testat, pushat; fas A med E (`sh run_sprint5a.sh EA`, körningar p5_eref1e-3/1e-4, frö 20–21) startad ca 14:15,
  därefter `python3 pilot_5a.py eref` (förutsättning + referens). Baslinjens p5_off återanvänds. Fas B med E:
  `sh run_sprint5a.sh EB <T50> <T25> <T12>` om förutsättningen klaras; annars stopp och beslut.
- **Pilot med E klar 23 okt 16:17.** Förutsättningen klarad (0,0001-varianten rätt vid 57,2/57,4 % av H5-händelserna).
  Referensen ej väldefinierad (75 mot 177 bitar) → halveringsregeln: körningar p5_ehalf{44.2,22.1,11.05,5.53}_s{20,21}.
  Rätt vid H5: 57→54→44→26→14 % vid 177→44→22→11→5,5 bitar; rätt över alla steg 76→76→73→64→48 %. Träff 86,5–87,9 %
  (Baslinje 86,7): liten nytta. Tvetydighet i villkor b (medel eller per frö): medel → referens 44, nivåer 22/11/5,5
  (rekommenderas); per frö → 88, nivåer 44/22/11. Alla körda. Doskontroll beteendemått: felgruppen < 200 i alla
  körningar (ja/nej-svaret läcker att felen inte mest ligger på avsedd plats; redovisat). Steg 4-förslag i specen:
  H5-kriterium (andel felmassa på avsett fönster, schema vs blandningsnollmodell med q matchat på massa på faktiskt
  fönster, ratio-of-sums, bootstrap, > 2 SE alla tre frön, per nivå), beteendemått (i) 20 000-episodbank + N ≥ 200
  eller (ii) beskrivande, H2 (b) på 1b redovisas, 5b = Baslinje + 0,0001 + tre nivåer, frö 30–32, 400k, MVP-banken.
  Oberoende granskning startad 16:25 (ska skriva `v2/results/sprint5a_review.md`, blind). Sedan grind 5a.
- **Granskning 5a klar 23 okt 16:40: godkänd med anmärkningar** (`v2/results/sprint5a_review.md`). Kod, pilot och
  villkor b godkända. **H5-kriteriet i steg 4 kan inte låsas**: (1) q omatchbar (nollmodellens massa på faktiskt ≥ 0,33
  vid q = 1, schemat 0,09–0,29); (2) olika stöd (schemat diffust över rutnätet, nollmodellen bara kandidater; vid ett
  enda iögonfallande föremål är nollmodellens andel ≡ 1); (3) glimtersättning jämn vs efter förhandssannolikhet ändrar
  ribban 0,54–0,58 → 0,72–0,77; (4) H5-beräkningskoden saknas; (5) beteendemått → beskrivande; (6) referensläsning inom
  bruset, 5,5 bitar ej matchbar; (7–8) två formuleringar rättade; (9) 5b: lås mål i bitar, pröva förutsättning igen,
  slutcheckpoint. Reviderat förslag + fem beslutspunkter i specen ("Grind 5a: dina beslut"). Grind 5a i två steg:
  beslut → H5-kod skrivs, provas på otränade agenter, pushas, genomförbarhet per nivå (bara träffsäkerhet) → låsning.
- **Grind 5a, produktägarens beslut 24 okt 01:24 (införda i specen under "Grind 5a: dina beslut"):** referens = medel
  över fröna (22 och 11 bitar bedöms, 5,5 redovisas); nollmodellens glimt ersätts efter kapningens förhandssannolikheter;
  matchning på omnormerad massa på faktiska fönstret, reserv mest sannolik ruta, annars ej bedömbar; beteendemått och
  H2 (b) på 1b beskrivande; resten godkänt, H5 per nivå utan samlat ja. **Ändrad inriktning:** H5 i 1b blir ett
  beskrivande prov som körs en gång på pilotens E-körningar (frö 20–21) när trösklarna är låsta. **5b körs inte.**
  Nästa sprint blir 1c (återanvänder H5-maskineriet, H5 som kriterium på ett schema som växer fram under belöning).
- **H5-koden skriven, provad, pushad 24 okt (commit ad3e092):** `v2/h5_eval.py` (regioner I/A/O, nollmodell med
  glimt efter förhandssannolikhet, cellkarta, matchning med reserv, bootstrap över episoder), `v2/h5_5a.py`
  (`feasibility` blind; `measure` vägrar utan `results/h5_lock.json`, som skrivs först efter produktägarens låsning),
  `v2/test_h5_eval.py` (otränade agenter och syntetiska scheman). Testbank för H5: MVP-banken 2 000 000 + i.
- **Kodgranskning 24 okt** (`v2/results/sprint5a_h5_code_review.md`, tre omgångar, blind): första versionen lutade mot
  underkänt per konstruktion (schemats diffusa massa på kandidatrutor). Rättat, men **väntar på produktägarens ja**
  (specen "Grind 5a, steg 2", fem ändringar): (1) nollmodellen får schemats diffusa andel per händelse, fördelad efter
  schemats täthet per rutklass färg × besökt (mätt utanför kandidaterna; reserv: körningens visited/unvisited-kvot,
  sedan samlad form); (2) delade rutor ut; (3) reserven över kandidatrutor + platåregel; (4) känslighet med argmax-
  matchning, olika utslag → "osäker"; (5) känslighet utan reservhändelser (55 % av händelserna). Alla syntetiska
  prov ≈ 0. Granskaren: låsbart.
- **Genomförbarhet (blind, slutgiltig, `results/sprint5a_h5_feasibility.json`; tidigare versioner _v1/_v2):** massmatchning
  går på alla nivåer i båda fröna, q 0,62–0,71; ribban (nollmodellens andel på avsett) 0,59 (utan kostnad), 0,59–0,61
  (44), 0,56–0,57 (22), 0,52–0,53 (11), 0,48 (5,5). **Inget H5-utfall är beräknat.** Efter produktägarens ja:
  `python3 h5_5a.py lock "<svar>"` och sedan `python3 h5_5a.py measure` (ca 30 min) → `results/sprint5a_h5.json`.
  Ändras någon kod efter låsning vägrar measure.
- **Grind 5a steg 2, produktägarens beslut 24 okt 15:56:** ändringarna 1–5 ja; låsning ja med tre villkor i låstexten
  (reservsnedvridning anges ärligt; ändring 5 noteras bara vid teckenbyte eller > 2 SE; klarar varianten utan kostnad
  tillskrivs arkitekturen). Motiv för körningen: 1b blir referens för 1c med samma låsta instrument. Därefter direkt
  till 1c: prob vars prior inte gynnar avsikten, diskret avläsning, minst tre frön.
- **H5 låst 24 okt (commit 988b8ae, `v2/results/h5_lock.json`) och körd en gång (`results/sprint5a_h5.json`): ej klarad
  på någon nivå.** Schemats andel av felmassan på avsett 0,13–0,34, nollmodellens 0,48–0,61, skillnad −0,13 till −0,46,
  17–71 SE. Argmax-känslighet samma utslag (ingen "osäker"); utan reservhändelser inget teckenbyte, ingen skillnad
  > 2 SE; varianten utan kostnad klarar inte heller. Tolkning: felen går till andra iögonfallande föremål; andelen
  stiger med minskad kapacitet men massan utanför kandidaterna stiger lika (0,39 → 0,76), troligen diffusitet, inte
  avsikt. Resultatgranskning: `v2/results/sprint5a_h5_result_review.md`.
- `docs/spec_v2.md` uppdaterad 23 okt vid grind 4-beslutet (se rubrikraden i filen för rev). Exportera specen dit vid varje grind.
- **Grind 5a stängd 24 okt kväll** (produktägaren, från resultattråden): H5-resultatet godkänt som referens för 1c.
  Hans tolkning: följer av konstruktionen (facit + larm → gissar bland iögonfallande; flaskhals → diffus karta), säger
  inget om AST (1b inget styrschema), stärker 1c. Krav på 1c i specen: bara belöning, nytta mätt; larm som faktor
  med/utan; avsikten inte gratis (nolläge, probens prior); diskret avläsning, ≥ 3 frön; fel i "om" och "var", låst före.
- **1c-plan skriven i specen 24 okt ("Sprint 6: 1c ... – plan"), väntar på godkännande. Inget byggt eller kört.**
  Modul som 1b men utan facit, tränad bara via policyns gradient genom kanalen; informationskostnad som energikostnad
  i belöningen; larmfaktor (utan = bara policyns indata, med = + matchningssignal + larm); prob på frusen agent väljer
  bland kandidatfönster, balanserad bank, ingen konstant för avsett; mått nytta (förutsättning > 2 SE alla frön),
  H5 "om" och "var" mot 5a:s nollmodell; referens: samma prob på 1b:s pilotkörningar. Pilot frö 40–41, 400k;
  huvudkörning frö 30–32, 9 körningar. Fem beslutspunkter.
- `docs/spec_v2.md` ej exporterad vid grind 5a: docs-exporten kommer bara inline som base64 och kan inte skrivas till
  repot från sessionen. Det levande dokumentet gäller.
- **1c-plan godkänd med villkor 24 okt 23:45** (specen, "Sprint 6 (1c), produktägarens svar"; villkoren gäller före
  planen där de skiljer sig): "utan" = exakt policyns tensorer, "med" = exakt 1b-E:s indata; rollblind prob (samma
  poängfunktion per kandidatfönster ur kanalens värden, ingen uppgift om avsett), ser kanalen vid t och t−1;
  kostnad = direkt straff (kapacitetsgräns), reglerad mot nivå ur 44/22/11, regel: lägsta nivå där "med" ger nytta
  > 2 SE i båda pilotfröna (40, 41), annars "utan" avgör; förutsättningar (a) nytta > 2 SE alla frön, (b) matchad
  kapacitetsbaslinje (samma parametrar och kanalbredd, kanalen tränad på blandad indata från start) ger inte samma
  nytta, (c) proben > likformig > 2 SE alla frön; nollmodellen likformig prior över kandidaterna; 1b-E vid vald nivå
  frö 30–32 som jämförelse (avgör AST-specificitet); "var" beskrivande; bitar med/utan fångst i piloten; kritikerns
  gradient redovisas; vid utebliven nytta: kollapskontroll → 0,0001 → dubbel fångstrisk → stopp.
- **Sprint 6 (1c) steg 1 (bygge) klart 25 okt, inget tränat.** Bara nya filer (låsta 5a-filer orörda):
  `v2/schema_1c.py` (Policy1c: baslinjens policy + gradient in i kanalkolumnen; Module1c: GRU + brusig kanal, ingen
  avkodare/förlust, kostnad = direkt straff beta·bitar reglerat mot mål; utan larm = exakt policyns tensor, med = exakt
  1b-E:s; shuffle = matchad kapacitetsbaslinje, givare = annan episod i batchen), `v2/train_1c.py` (A2C, modulen tränas
  bara via policyns förlust genom z; loggar bitar efter fångst/ej fångst, kritikerns andel av modulgradienten),
  `v2/probe_1c.py` (rollblind prob 8 särdrag ur kanalen vid t+1 och t per kandidatfönster, MLP, balanserad probbank
  7 000 000 + i; nollmodell likformig prior + glimt efter fångstprior, q matchat på massa på faktiskt; (c) blint;
  "om"/"var" bara via `measure()` som kräver `results/h5_1c_lock.json`; ej matchbart q = ej bedömbar),
  `v2/pilot_1c.py` (eval / level / ref1b → `results/sprint6_pilot.json`, final-checkpoints, pilotbank 4 000 000 + i),
  `v2/run_sprint6.sh` (faser A, B L, U, C, D). Tester: `test_1c.py` (FD för policy och modul genom kanalen),
  `test_probe_1c.py` (rollblindhet, nollmodell, syntetiska kanaler). Ca 15 000 ep/min → ca 27 min per pilotkörning.
  Syntetiskt: tom kanal ger "om" −9,5 SE (nollmodellen lutar mot avsett vid högt q); AST-lik kanal (40 % fel, alla mot
  avsett) +4 SE.
- **Granskning av bygget 25 okt: godkänd med anmärkningar** (`v2/results/sprint6_build_review.md`). Rättat direkt:
  (3) kapacitetsbaslinjens kanal och bitar på egna rutor (`forward(..., cm=)`), (8) givare = derangering, (4) blind
  sammanfattning bara över händelser med K ≥ 3 (vid K = 2 är massa på avsett = 1 − massa på faktiskt); full (c) via
  `c_all()` efter låsning, (5) bitar mot nivån (±20 %) och förutsättning (b) i piloten (`pilot_1c.py b L e|u`),
  (6) nivåregeln väntar tills fas A är utvärderad, (7) kapacitetsbaslinjen följer det villkor som avgjorde nivån
  (`run_sprint6.sh B L e|u`, cap/capu). **Öppet till produktägaren före grind 6a:** (1) "om" ej kalibrerat för kanal
  utan rollinformation (+2 till +4,7 SE vid träffsäkerhet 0,6–0,7, −10,7 vid 0,5) → kalibreringstest eller
  rollbyteskontroll; (2) kanalen vid t bär styrkommandot → prob bara på t+1 som känslighetsanalys eller primärt.
  Repot är en grund klon: låskontroll via sha256 i `h5_lock.json`, inte git-commits. Fas A startad 25 okt (se nedan).
- **Beslut 25 okt 02:40 (produktägaren, Claudes rekommendationer):** (1) rollbyteskontroll som kriterium för "om",
  (2) proben läser primärt bara t+1 (4 särdrag, `COLS_PRIMARY`), t+1 och t som känslighet, "om" uppdelat på om
  kommando t+1 siktar på avsett igen (tolkningsvillkor). **Kalibreringen (test_probe_1c.py, 2 000 episoder) visade att
  kontrollen i formen om(äkta) − om(låtsas) med nollmodellen på båda sidor är skev** (−8 till −44 SE för kanaler utan
  rollinformation). Byggt som probens massa på avsett minus på låtsasfönstret (ej nollmodell); då |z| < 2 för alla
  kanaler utan rollinformation, AST-lik +19–26 SE, "fel åt annat håll" −17, styrsignal vid t når bara känslighetsproben.
  Avvikelsen från beslutets ordalydelse ska godkännas av produktägaren före 6a.
- **Pilot klar 25 okt ca 05:20: STOPP enligt regeln, förutsättning (a) klaras inte i något villkor.** Med larm: 44 bitar
  +1,2/+1,4 p (z 1,7/1,8), 22 och 11 bitar ≤ +0,6; kostnad 0,0001 +1,2/+1,9 (z 1,6/2,6); dubbel fångstrisk +1,7/+0,6 (z
  1,8/0,7). Utan larm aldrig nytta (−1,6 till +0,9 p). Policyn med larm är beroende av kanalen (blandad vid test −5 till
  −15 p) men vinner lite. Bitar efter fångst = utan fångst i alla körningar (ingen avvikelsekod). Kritikerns andel av
  modulgradienten 1–16 %. Ingen kollaps. H5 bedöms inte för 1c. Inget om var probens fel pekar har beräknats.
  Sprint 3:s rutnät visar att 1a vinner +11 p vid fångstrisk 0,5, där 1c med larm bara tog +1,7/+0,6 → flaskhalsen är
  troligen inlärningssignalen, inte uppgiften (Claudes förslag att flytta arbetspunkten tillbakadraget).
- **Fas E (efterhandsanalys, godkänd 25 okt 08:14):** `sh run_sprint6.sh E` = baslinje + 1c med larm, kostnad 0,0001,
  fångstrisk 0,5, 1 M episoder, frö 42–43 (körningar c6_dcL-*). Nytta > 2 SE båda fröna → föreslå huvudkörning;
  annars produktägarens val: prediktiv signal (förutsäga nästa glimt) eller avsluta + resultatrapport (skrivs oavsett).
  Fortfarande sprint 6.
- **Fas E klar 25 okt 09:33 (efterhand, blind):** 1c med larm vs baslinje vid 1 M episoder, fångstrisk 0,5: frö 42 +2,3 p
  (80,9 mot 78,6 %, z 2,6), frö 43 +2,8 p (80,9 mot 78,1 %, z 3,1) → **nytta > 2 SE i båda fröna**. Bitar ~21/steg.
  Blandad kanal vid test −9 till −13 p. Prob vid fångst (K ≥ 3) långt över likformig (z 27–28). Kritikerns andel 2–6 %.
  Inlärningskurvor (träningseval, 2 000 ep) brusiga. Efterhand: låser inget. Kapacitetskontroll (b) ej prövad.
  Huvudkörning godkänd 25 okt 12:53 ("kör på", inkl. fast kostnad 0,0001 och kontroll utan nollmodell).
- **Huvudkörningen förberedd men INTE låst eller startad:** `v2/eval_main_1c.py` (lock/run, kriterier (a)–(c), "om" via
  rollbyteskontroll, AST-specificitet mot 1b-E), `run_sprint6.sh M` (15 körningar c6_mL-*, frö 30–32, fångstrisk 0,5,
  kostnad 0,0001, 1 M ep). Kontrollens godkännande kräver inte längre matchbart q. **Röktest 25 okt: otränade agenter
  klarar rollbyteskontrollen** (1500 ep: 1c med larm +0,011 / +5,2 SE, utan larm +0,004 / +3,3, 1b-E +0,010 / +5,1,
  kapacitetsbaslinjen 0,000 / +0,5): kanalen för passivt vidare den remappade foveán och matchningssignalen, som ligger på
  avsedda rutor. Förslag till produktägaren: kriteriet = tränad minus otränad agent (samma arkitektur och frö) > 2 SE.
  Produktägaren sa ja 14:21 (tränad minus otränad).
- **Granskning före låsning 25 okt: EJ LÅSBAR** (`v2/results/sprint6_main_prelock_review.md`). (1) Låtsasfönstret är alltid
  iögonfallande, det avsedda aldrig: med balanserad probträning lär sig proben "det icke-iögonfallande är det faktiska" av
  stegen utan fångst, så en kanal som bara kodar iögonfallande klarar kontrollen (+54 SE). (2) Otränad referens byter tecken
  mellan frön, annan händelsefördelning. (3–4) låsning/kontroll av checkpoint, cache. **Åtgärdat (väntar på produktägaren):**
  proben tränas bara på fångststeg (`probe_1c.training_set`, avviker från villkoret "balanserad bank"), låtsasfönstret =
  annan felaktig iögonfallande kandidat; kalibrering: ingen kanal utan rollinformation klarar (noll, markör, fel kandidater,
  iögonfallande, styrsignal, passiv avsiktsindata: z −59 till +0,8), AST-lik +17–22 SE. Därmed åter kriterium = tränad
  agents kontroll > 2 SE; otränad skuggmodul (samma arkitektur/frö, samma episoder, kanalen ej till policyn) redovisas parat.
  `eval_main_1c.py` kontrollerar checkpoint (fångstrisk, kostnad, ≥ 1 M ep, flaggor) och cachar per sha256;
  `run_sprint6.sh` i låsningen.
- **Andra granskningen 25 okt: EJ LÅSBAR** (`v2/results/sprint6_main_prelock_review2.md`): kanaler byggda av miljöns kartor
  (dämpad iögonfallande, iögonfallande på obesökt, prioritetskarta) klarar låtsaskontrollen falskt (+4 till +102 SE),
  eftersom avsett fönster är systematiskt annorlunda än låtsasfönstret.
- **Huvudkörningen startad 25 okt 16:22** (förregistrering `v2/results/sprint6_main_prereg.json`: körplan + (a)–(c));
  stoppad vid 2 h-gränsen med 6/15 klara (baslinje, 1c med larm), återstartad 20:10 (`sh run_sprint6.sh M`).
- **Beslut 25 okt 20:10 ("kör på" på Claudes förslag): matchat kontrollfönster** (`probe_1c.pick_matched`: ej kandidat,
  ej iögonfallande, samma målfärg och besökt-status som avsett, innehåll ≠ glimten), poängsatt i utökad softmax. H5-kriteriet =
  avsett > låtsasfönster OCH avsett > matchat fönster, båda > 2 SE alla frön (`measure()["criterion"]`). Kalibrering
  (2 000 ep): inga kanaler utan rollinformation klarar (inkl. dämpad, obesökt iögonfallande, prioritet, täckning, fel
  kandidater); AST-lik klarar (+16–22 SE), även blandad med iögonfallande; täckningsminne via efferenskopian klarar
  (AST-förenligt, ej AST-specifikt — det avgör 1b-E-jämförelsen, nu för båda kontrollerna).
- **Tredje granskningen: låsbar med anmärkningar** (`v2/results/sprint6_main_prelock_review3.md`); anmärkningar åtgärdade.
  Viktigast för tolkningen: en kanal som markerar både avsett och faktiskt fönster klarar kriteriet; korrelationen mellan
  probens massa på avsett och faktiskt redovisas (AST-lik ca −0,8).
- **GRIND 6a: H5 för 1c LÅST 25 okt ca 21:00** (`v2/results/h5_1c_lock.json`, commit 857e103), före all utvärdering av
  huvudkörningen. Efter träningen: `python3 eval_main_1c.py run` → `results/sprint6_main.json` (vägrar om koden ändrats).
- **Huvudkörningen klar och utvärderad 25 okt 23:30** (`v2/results/sprint6_main.json`): **förutsättningarna klaras inte,
  H5 bedöms inte för 1c.** Träff: baslinje 79,0/78,6/78,9 %, 1c med larm 80,9/81,3/80,5 (frö 32 z 1,75 → (a) faller),
  1c utan larm 78,2/79,0/77,9 (kanal kollapsad 2–4 bitar), kapacitetsbaslinje 78,5/78,3/79,0 (0,5–2,4 bitar), 1b-E
  81,3/81,2/81,5 (139–149 bitar). H5 beskrivande: avsett får mindre massa än låtsasfönster och matchat fönster (−17 till
  −95 SE), korrelation avsett–faktiskt −0,2.
- **Resultatgranskning: bedömningen korrekt** (`v2/results/sprint6_main_result_review.md`). Negativa H5-värden = avsikten är
  avläsbar och proben (tränad på fångststeg) lär sig "avsikt = inte här"; inget belägg för AST eller emot; nollmodells-"om"
  oanvändbart (−35 SE även för tom kanal); (b) kontrollerar parametrar, inte information (kapacitetsbaslinjen kollapsar);
  vinsten kommer från jämförarsignalen (larmet), inte från en självmodell; sammanvägt +2,0 p (z ≈ 3,9), kriteriet har låg
  styrka. Föreslagen nästa kontroll: larmet direkt till policyn utan modul.
- **GRIND 6 GODKÄND 26 okt 00:59** (produktägaren: "kör vidare med din rekommendation"): resultatet godkänt, 1c avslutas i
  denna form. Resultatrapport sprint 1–6 skriven som dokument: https://claude.ai/artifact/EYa5LBThWfTTp76SjS3jRv
  ("Nexis v2 – resultatrapport sprint 1–6"). Inget nästa steg beslutat (alternativ i rapporten: larmet direkt till policyn,
  prediktiv signal till modulen, social fas, fas 2).
- **Inriktning 26 okt 01:20 (produktägaren):** lämna AST som måttstock, cybernetisk linje (Conant–Ashby, nödvändig varietet,
  reafferensprincipen). **Sprint 7-plan skriven i specen 26 okt ("Sprint 7: cybernetisk linje, fördröjd återkoppling"),
  väntar på godkännande. Inget byggt eller kört.** Foveán visar glimten D steg senare (D = 0/1/2), jämföraren tidsinställd;
  baslinje, 1a, 1c med/utan larm; huvudmått = självreferens (kanalen vid t bär faktiskt fönster vid t innan glimten kommit,
  mot otränad skuggmodul), plus använt, förenklat, nytta, form. Steg: bygge → utrymmeskontroll (baslinje + 1a, frö 50–51,
  400k, grind 7a: minsta D med 1a ≥ +8 p) → pilot 1c → låsning 7b → huvudkörning frö 60–62. Fyra beslutspunkter.
- **Diskussionstråden 26 okt: sprint 7-planen "nej i nuvarande form"; Claude håller med. Reviderad plan i specen ("Sprint 7,
  reviderad plan efter diskussionstråden"), väntar på godkännande.** Skäl: baslinjen använder inte glimten (H6, blind fovea
  lika bra) → fördröjning ändrar inget; 1a-facit utan fördröjning = tautologi; grinden väljer D = 0 vid fångstrisk 0,5;
  självreferens sant per konstruktion via efferenskopian. Teori: internmodellprincipen (Francis & Wonham) + framåtmodeller,
  PCT som nollteori. Steg 0: larmet direkt till policyn utan modul (frö 30–32, fångstrisk 0,5, 1 M, jämförs med sprint 6:s
  körningar). Huvudlinje: inre störning (inre kapning). Mått: kalibrera självreferens mot efferens/salience-kanaler, "använt"
  huvudmått, beteendesignatur, prob på baslinjens dolda tillstånd, ett låst kriterium. Förslag: social fas villkoras av
  positiv kontroll i social miljö i stället för att 1c visat något. **Rapporten rättad:** 1b-E med facit ≈ 1c i vinst → policyns
  gradient troligen inte flaskhalsen; gapet = lokalisering vid fångst (57 % mot ideal 94 %).
- **Förslag 26 okt (på produktägarens begäran): självmodell under socialt tryck, nollhypoteser från en ensam agent** (specen,
  väntar på godkännande, inget byggt). Två agenter, brusig syn på den andras spotlight, egna ofullständiga ledtrådar;
  ensam kontroll = uppspelad, icke-reagerande partner (samma indata). 2 × 2 ensam/social × kontinuerlig/symboler, socialt
  som samarbete och tävling. H0-0 positiv kontroll (facit om den andras uppmärksamhet hjälper, grind 8a), H0-1 självmodell
  lika ensam/socialt (låst huvudkriterium), H0-2 oberoende kod för egen/andras, H0-3 symboler × socialt, H0-4 sanning vid
  samarbete = tävling. Kostnad 18–24 körningar, 15–25 h. Fyra beslutspunkter.
- **Precisering 26 okt 01:42 (produktägaren):** frågan är om *nyttan* av en självmodell kräver begreppsbildning + teori om
  andras sinnen. Uppställning 2 × 2 × 2 (självmodell med/utan × levande/uppspelad partner × symboler/kontinuerlig);
  huvudmått = självmodellens nytta per ruta; nollhypotes = lika nytta i alla fyra; tesen stöds om rutan symboler + teori om
  andras sinnen > var och en av de övriga > 2 SE alla frön; förutsättning H0-0. H0-1–H0-4 beskrivande. I specen.
- **Diskussionstråden 26 okt om förslaget; reviderat förslag i specen ("Reviderat förslag efter diskussionstråden"), väntar på
  godkännande.** "Kräver nytta" motsägs av egna data (1a +7–11 p ensam, lesion −44 p) → tesen i två led: (1) storlek, givet
  schema gör större nytta socialt (interaktion), (2) uppkomst, självbyggt schema växer fram socialt men inte ensamt.
  Hanabi-miljö: partnern ser min faktiska spotlight inkl. fångster, jag vet bara vart jag siktade; samarbete = inte söka
  samma rutor. Krav: samma målledtråd för båda, kanaler matchade i bitar, kopplad (yoked) ensam kontroll, H0-0 = facit om
  partnerns dolda tillstånd > bara position, styrkeberäkning före låsning. Teori om andras sinnen = falsk tro; begrepp =
  gemensam kod (H0-2 i kärnan), generalisering, tala om det icke närvarande. Led 1: 2 × 2 givet schema med/utan × levande/
  kopplad, symboler, samarbete. Steg 0 separat, inre störning uppskjuten. Fyra beslutspunkter.
- **BESLUT 26 okt 01:57 (produktägaren, efter diskussionstrådens rekommendation): den större utökningen väntar.** Ordning:
  (1) steg 0 nu: larmet direkt till policyn (sluter sprint 6); (2) resultatrapporten slutförs med steg 0 och metodlärdomar,
  på engelska om den ska ut, data skilda från tolkning; (3) reproducerbarhetsskript i repot som återskapar huvudtabellerna;
  (4) extern granskning av en människa med ML-erfarenhet, sedan kontakt med Grazianos grupp och Piefke m.fl.;
  (5) paus för beslut: social tes, inre störning eller ingenting; ny studie får egen spec. **Inget nytt bygge mellan 1 och 5.**
- **Steg 0 startat 26 okt ca 02:10** (commit b37fef9, förregistrering `v2/results/sprint6_step0_prereg.json`):
  `v2/train_alarm_direct.py` (baslinjens policy + matchningssignal och larm som indata, train_rl.py oförändrad),
  körningar `c6_mL-alarmdirect_s{30,31,32}`, fångstrisk 0,5, 1 M ep. Efter träning: `python3 eval_step0.py` →
  `results/sprint6_step0.json`. Regel: inom 2 SE av 1c med larm eller över i alla frön → modulen tillförde inget.
  Låsta filer orörda (båda låsen kontrollerade).
- **Steg 0 klart 26 okt (`v2/results/sprint6_step0.json`): modulen tillförde inget utöver larmet** (förregistrerade regeln
  uppfylld). Larm direkt till policyn: träff 80,9/81,3/81,1 % (frö 30–32); minus 1c med larm 0,0/−0,05/+0,6 p (z 0,0/−0,06/
  0,7); minus Baslinje +1,9/+2,6/+2,2 p (z 2,1/2,9/2,5, > 2 SE i alla frön, vilket 1c med larm inte klarade i frö 32).
  Sprint 6 sluten. Nästa: steg 2 (rapport), 3 (reproducerbarhetsskript).
- **Steg 2 (rapport på engelska) utkast klart 26 okt:** https://claude.ai/artifact/WgWqgRBXEqaChu3NMB76JF ("A comparator, not a
  self-model: testing attention-schema predictions in small RL agents"). Data (avsnitt 4) skild från tolkning (6), metodlärdomar,
  begränsningar, avvikelser (bilaga). Siffror kontrollerade mot results/*.json. Referenser utan länk citerade ur minnet.
  Avsnitt 8 hänvisar till reproducerbarhetsskriptet (steg 3, ej skrivet). Produktägaren har inte beslutat om extern granskning.
- **Rapporten granskad 26 okt (annan Claude-instans, 19 kommentarer, "kan skickas ut efter ändringar").** Alla elva sakfel
  prövade mot kod/JSON/källor och bekräftade: steg 0 matar matchningskarta + larm; larmet = ELLER; 1c:s bitkostnad på
  modulens förlust; 1c utan larm saknar matchningssignal (två saker ändras); entropi 700k i sprint 4; H5 5a två frön; −0,29 =
  kvarstående effekt; pilot utan larm max +0,7; bitar efter fångst −2,6 till +9,2; (b) föll också i frö 32, sammanvägt post hoc;
  Liu/Piefke rättade. Införda, plus mellan-frö-medel, KI för steg 0, lokalisering 26–45 % på bedömda nivåer, skuggmodul,
  begränsningar, nya referenser (ur minnet: Agarwal, Hewitt & Liang, Mnih 2014, Sperry, Wolpert), bilaga 12–14.
  Kvar: steg 3 (reproducerbarhetsskript), mänsklig ML-granskare.
- **Steg 3 klart 26 okt: `v2/reproduce.py`** (`tables` sekunder / `evaluate` 33 min / `train` ca 10–15 h). `evaluate` på de
  incheckade checkpointerna återskapade alla sex resultatfiler exakt (4 809 värden, 0 skillnader;
  `results/reproduce_evaluate_report.md`). Omträning från grunden av c6_mL-alarmdirect_s30 gav bitidentiska vikter
  (`results/reproduce_determinism.md`). Full omträning ej körd. Rapportens avsnitt 8 uppdaterat. Kvar: steg 4 (mänsklig granskare).
- **26 okt 17:03: produktägaren mejlade Piefke och Thorat (Osnabrück)** med rapport-PDF, ordlista-PDF och länk till repot,
  som gjordes publikt 26 okt. Pitch och mejltext: claude.ai-projektet, `claude/nexis_pitch.md`. Repo-dokument uppdaterade:
  `docs/PROJECT_OVERVIEW.md`, `docs/report_v2.md` (ögonblicksbild), `v2/README.md`; `docs/spec_v2.md` märkt inaktuell.
  Ordlistan har engelsk flik. Väntar på svar och/eller mänsklig ML-granskare. Inget nytt bygge.
- **27 okt: svar från Sushrut Thorat** (delade en GPT-konversation; frågade vilket påstående om AST och medvetande vi gör).
  Kritiken bedömd som i huvudsak riktig: titeln överdrev (policyn kan själv representera uppmärksamhet); absolut
  handlingsrymd gör att agenten aldrig behöver veta var spotlighten står (bara bokföring); H5-förutsägelsen följer inte av
  AST utan extra antaganden; mejlet överdrev Piefke-jämförelsen (konvergens inom budget, minne/efferenskopia). Rapporten
  reviderad: ny titel "A learned attention-schema module adds no benefit over hand-coded mismatch features in small RL search
  agents", snävare sammanfattning, nytt diskussionsstycke om uppgiftens design, H5 och Piefke rättade, tre nya begränsningar.
  `docs/report_v2.md` och `docs/PROJECT_OVERVIEW.md` uppdaterade. Utkast till svarsmejl givet. Hans designförslag (samma
  felsignal kräver olika handlingar beroende på var uppmärksamheten hamnade + kausala prov på policyns inre) = utgångspunkt
  om ny studie.
- **27 okt: animationen "Nexis genom agentens ögon"** (https://claude.ai/artifact/YaJypo4vjvo7cFUKTaswsz, delad med länk)
  rättad: märkt som konstruerad illustration, 57 %-siffran flyttad till 1b-E, steg 0 (larm direkt, 81,1 %) i berättelse och
  diagram, ny slutsatskort med svagheter (absolut handlingsrymd, policyns representation ej prövad), 1a-punkten markerad som
  uppskattning; svensk/engelsk växling (`#sv`/`#en`). Länkad från rapporten (avsnitt 2) och `v2/README.md`.
