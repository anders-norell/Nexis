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
| Ögonblicksbild av specen | `docs/spec_v2.md` (22 okt 2026; det levande dokumentet gäller vid skillnad) |
| v1: spec, kod och resultat | `docs/spec.md`, rotkatalogens `*.py`, `results/` — rör inte, det är ett avslutat försök |
| v2: kod | `v2/` (skapas i sprint 1) |
| v2: resultat | `v2/results/` |

Specen läses med docs-verktygen (inte web-fetch). Produktägaren svarar ibland på öppna frågor direkt i
specen; kontrollera ändringar sedan förra sessionen innan arbetet börjar.

## Arbetsregler

- **Sprintar med grindar.** Varje sprint avslutas med att produktägaren godkänner resultatet mot specen
  innan nästa påbörjas. Påbörja aldrig nästa sprint utan godkännande.
- **Trösklar låses före körning** (i sprint 3, efter piloten) och ändras aldrig efter att resultat setts.
  Signifikanskriterium: skillnad > 2 SD och samma riktning i alla frön.
- **Alla resultat rapporteras**, även misslyckanden. Ett godkänt resultat visar att mekanismen beter sig
  som AST förutsäger, inget om upplevelse.
- **Oberoende granskning:** vid varje grind granskar en separat agent kod och resultat mot specen.
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
| 4. MVP | Baslinje, Fast schema (1a), Inlärt schema (1b), Blandat schema; 3 frön vardera (12 körningar). H5 mäts på 1b. Ingrepp vid test: falskt schema (H2), lesion (H3), fångst (H5). | Resultat bedömda mot låsta kriterier |
| 5. Utbyggnad | Zoom, skalning, övriga ledtrådsnivåer, 1c, fem frön. | Enligt resultat |

## Status

- **Senast uppdaterad:** 22 okt 2026
- **Sprint 0 (spec):** klar. Spec v2 godkänd i huvudsak av produktägaren.
- **Sprint 1 (miljö):** **klar, grind 1 godkänd 21 okt 2026** (produktägaren delegerade bedömningen till Claude).
  Kod: `v2/env.py`, `v2/agents.py` (slump-, avsöknings- och styrd referensagent; den styrda kan spåra
  avsedd, härledd eller faktisk plats), `v2/test_env.py` (46 tester), `v2/check_env.py` → `v2/results/sprint1_env_check.*`.
- **Sprint 2 (kompetens):** **klar, grind 2 godkänd av produktägaren 22 okt 2026.**
  - Grind (låst 21 okt före träning): K1 träffandel > slump med > 2 SD, K2 ≥ 73 % (bästa minneslösa
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
- **Nästa steg: sprint 3 (pilot).** Börja med att skriva sprint 3-planen i klartext i specen och få den
  godkänd. Öppet beslut i specen: tillägget H6 (schemats vinst växer när foveán avslöjar mindre; piloten
  varierar foveáns tydlighet och fångstrisk, tränar Baslinje och Fast schema 1a i varje läge, väljer
  arbetspunkt där 1a ger tydlig vinst innan trösklar låses; ingen vinst ens med tvetydig fovea = omtag).
