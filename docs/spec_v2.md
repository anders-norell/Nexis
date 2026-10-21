<!-- Ögonblicksbild av livedokumentet "Nexis v2 – spec" (https://claude.ai/artifact/GsEmfNtb1vs2bhXxwc6vsW), 22 okt 2026. Det levande dokumentet gäller vid skillnad. -->

# Nexis v2 – spec

2026-10-20 · @Anders Norell

## Syfte och teoretisk grund

Nexis v2 frågar om ett systems kunskap om sin egen uppmärksamhet kommer från en förenklad modell och inte från uppmärksamheten själv. I så fall ska modellen styra beteendet även när den är fel, och den ska ha fel på det sätt som Grazianos Attention Schema Theory (AST) förutsäger. Det är AST:s förklaring av medvetenhet, och såvitt vi kunnat se är den inte prövad i artificiella agenter.

AST skiljer på två saker. Uppmärksamheten är processen: signaler som tävlar om vidare bearbetning. Schemat är en förenklad beskrivning av processen, som används för att styra den. Upplevd medvetenhet är enligt AST schemats innehåll. Arbetshypotesen här är att schemat fungerar som uppmärksamhetens proprioception, ett inre sinne riktat mot den egna uppmärksamheten.

Ett schema i projektets mening måste uppfylla tre villkor:

1. Det handlar om processen (var fokus ligger, hur brett, om det skiftar), inte om innehållet som hämtas.
2. Det är förenklat jämfört med den faktiska attention.
3. Det används kausalt för att styra attention.

Fas 1 prövar alla tre villkoren i en agent byggd för ändamålet. Fas 2 körs bara om fas 1 visar effekt och undersöker då om en verklig transformer bär ett schema av den sort som fas 1 visat vara användbart. Referenser: Wilterson & Graziano (PNAS 2021), Liu, Bolotta, Zhu, Bengio & Dumas (2023), Webb, Kean & Graziano (2016).

## Lärdomar från v1 och tidigare forskning

Att ett schema hjälper styrningen är redan visat. Det som är nytt i Nexis v2 är ingreppen i schemat och testet av schemats systematiska fel, och dit bör tyngdpunkten flyttas.

**Redan gjort av andra:**

| Studie | Vad den visade | Överlapp med Nexis v2 |
| --- | --- | --- |
| Wilterson & Graziano 2021 | Agent som fångar en boll med en spotlight. Ett givet schema förbättrar styrningen, och att stänga av det efter träning försämrar den kraftigt. | Fast schema (1a) och lesion är replikation. |
| Piefke, Doerig, Kietzmann & Thorat 2024 | Ett schema som inte är förprogrammerat växer fram ur belöningsinlärning och bär information om spotlightens läge. Det är en ledtråd, inte en kopia, och gör mest nytta vid måttlig störning. | Framväxande schema (1c) är delvis gjort. Störningsnivå motsvarar vår ledtrådsvrid. |
| Farrell, Ziman & Graziano 2024 | Med transformer-attention hjälper ett schema att bedöma andras attention och att samarbeta, utan att förklaras av extra kapacitet. Schemat tränas med uttryckligt facit. | Den sociala funktionen, som vi inte testar. |
| ASAC 2025 | Ett schema inbyggt som styrenhet i transformerns attention. | Nära fas 2, men som konstruktion, inte som undersökning av en färdig modell. |

**Det som inte tycks vara gjort** (bedömt utifrån studierna ovan, ingen fullständig litteraturgenomgång):

- Falskt schema som kausalt ingrepp: flyttas den verkliga attention när schemat ljuger?
- Förregistrerat test av schemats systematiska fel, att schemat drar mot *avsedd* snarare än *faktisk* attention.
- Schema över attention-*historik* i en sökuppgift med återbesök, mätt med söklutning.
- Flaskhals som gör schemat förenklat, så att karikatyren uppstår i stället för att konstrueras.

**Lärdomar från v1 (oktober 2026, ****repo****):**

- *Uppgiften måste kräva schemat.* I v1 skulle agenten följa ett mål. Schemat gav +2,6 procentenheter, säkert men långt under tröskeln, och i v2 ingen skillnad alls. Kontrollagenten kunde sluta sig till var attention hamnat utifrån vad den såg. Samma risk finns i sökuppgiften och ska kontrolleras redan i piloten.
- *Trösklar ska motsvara en meningsfull effekt, inte en önskad.* v1 krävde 10 och 20 procentenheter utan underlag. I v2 sätts trösklarna efter piloten.
- *Det mest AST-specifika fyndet var en bifångst.* När attention drev iväg angav schemat den avsedda platsen i 31 % av fallen, mot cirka 7 % väntat. Det blir en förregistrerad hypotes i v2.
- *Kod som kan återanvändas:* den egenbyggda autodiffen i NumPy, eftersom PyTorch fortfarande inte går att installera i arbetsytan, samt delar av miljön, modellen och den statistiska utvärderingen.

## Avgränsning

Nexis v2 är ett mätbart experiment om schemats funktion, inte ett försök att skapa eller påvisa medvetande. Huvudfrågan avgörs i fas 1, i en miljö där styrning av en begränsad attention är nödvändig.

**Ingår:**

- Fas 1: en liten agent med begränsad, styrbar attention i en sökmiljö, med och utan schema.
- Kausala ingrepp i schemat, kontroller och en skalningsserie över agentstorlek.
- Fas 2, villkorad av fas 1: undersökning av om GPT-2 small bär ett schema av den sort fas 1 visat vara användbart.

**Ingår inte:**

- Påståenden om subjektiv upplevelse hos modellen.
- Stora modeller och multiagentmiljöer.
- Rapportering i naturligt språk ("jag uppmärksammar X"). Det kan bli en senare fas.

## Begrepp och mappning

Varje AST-begrepp får en konkret motsvarighet i båda faserna, så att varje hypotes går att mäta.

| AST-begrepp | Fas 1 (agent) | Fas 2 (GPT-2 small) |
| --- | --- | --- |
| Försöksperson | Liten agent med minne över tidssteg | GPT-2 small, 12 lager, 12 huvuden |
| Inre tillstånd | Agentens dolda tillstånd per tidssteg | Residualströmmen per lager och token |
| Uppmärksamhet (processen) | En spotlight som bara visar en del av miljön per steg | Attention-vikterna, per huvud en fördelning över tidigare tokens |
| Schema | Sammanfattning av spotlightens läge och historik, matad till agenten | Komprimerad sammanfattning av attention, avläst med probe |
| Styrning | Agentens val av nästa spotlightläge | Mäts inte |

**Terminologi:** *attention* avser processen. *Fokus* avser var attention ligger, det vill säga de positioner som får störst vikt eller som spotlighten täcker. *Bredd* avser hur utspridd attention är, och *skifte* hur mycket fokus flyttar mellan steg.

**Schemats innehåll:** i fas 1 spotlightens nuvarande läge, senaste förflyttning och en grov karta över besökta områden. Zoom, och därmed bredd, läggs till som första utbyggnad efter MVP. I fas 2 position, bredd och skifte för attention-vikterna. Schemat är medvetet grovt, i linje med villkor 2.

**Förbehåll:** attention-vikter är ett omstritt mått på vad som faktiskt påverkar utfallet. Analogin med biologisk uppmärksamhet är en arbetshypotes, inte ett antagande.

## Fas 1: kan schemat vara avgörande?

Fas 1 skapar en miljö där styrning av en begränsad attention är nödvändig och testar om schemat då gör en stor och kausal skillnad. Huvudkriteriet är ett ingreppstest: ett falskt schema ska flytta agentens fokus förutsägbart.

**Miljö:** visuell sökning i ett rutnät. Agenten ser detaljer bara där spotlighten ligger (fast fönsterstorlek i MVP), och dessutom en grov, lågupplöst bild av hela rutnätet, motsvarande periferin. Periferin visar till exempel färg men inte form. Målet ska hittas bland distraktorer inom en stegbudget, och återbesök kostar steg. Rutnätets storlek, spotlightens fönster, stegbudget och antal distraktorer varierar slumpmässigt mellan episoder, så att agenten inte kan lära sig dem utantill. Upplägget bygger på Wilterson & Graziano (2021).

**Ledtrådar som gör smart sökning möjlig:** perifera särdrag som delvis matchar målet (styrd sökning) och statistisk struktur i var målet brukar ligga. Ledtrådarnas styrka varieras i tre nivåer. Ingen: periferin visar bara att det finns objekt, och systematisk avsökning är optimal. Svag: de flesta distraktorer delar målets färg. Stark: få distraktorer delar målets färg.

**Huvudmått:** söklutning, det vill säga hur många fler steg som krävs per tillagd distraktor. Linjär avsökning ger brant lutning, styrd sökning flack.

**Agent:** en liten policy med minne över tidssteg, tränad med förstärkningsinlärning: belöning när målet hittas och ett litet avdrag per steg. Insignaler är ledtråden om vad som söks, periferin, spotlightens innehåll och, i schemavillkoren, schemat. Agenten väljer vid varje steg nästa spotlightläge.

**Två schemavarianter:**

- **Fast schema (****1a****):** vi räknar ut sammanfattningen av spotlightens läge och historik med en fast formel och matar in den. Agenten lär sig läsa ett färdigt sinne.
- **Inlärt schema (****1b****):** agenten tränas också att förutsäga sin egen spotlightsammanfattning, med det faktiska läget som facit, och den egna förutsägelsen matas tillbaka. Schemat blir då förenklat på agentens egna villkor och kan vara systematiskt fel, vilket ligger närmare AST.

**Villkor att jämföra:**

| Villkor | Schemakanal | Syfte |
| --- | --- | --- |
| Baslinje | Ingen | Referens |
| Fast schema (1a) | Formelberäknad sammanfattning | Testar nyttan av ett givet schema |
| Inlärt schema (1b) | Agentens egen förutsägelse | Testar ett självbyggt schema |
| Blandat schema | Sammanfattning från en annan episod, samma storlek | Kontroll för extra kapacitet |
| Lesion | Avstängd vid test, slumpvis avstängd även under träning | Analog till uppmärksamhet utan medvetenhet, utan ovan indata |

**Ingrepp och dos–respons:**

- *Falskt schema:* schemat anger ett annat fokus än det faktiska. Mät om spotlighten flyttas i förutsägbar riktning.
- *Brusnivåer:* schemat görs gradvis brusigare. Styrningen ska försämras gradvis och monotont om schemat verkligen används.

**Skalning:** alla villkor körs i minst tre agentstorlekar.

**Hypoteser:**

- **H1:** schemavillkoren ger flackare söklutning än både Baslinje och Blandat schema, särskilt när distraktorerna liknar målet.
- **H2:** ett falskt schema flyttar fokus förutsägbart. Det är det starkaste beviset för att schemat styr.
- **H3:** vid Lesion rör sig spotlighten fortfarande, men styrningen försämras: fler återbesök och instabilare fokus. Det speglar Webb, Kean & Graziano (2016).
- **H4:** effekten består eller växer med agentstorleken. Försvinner den är schemat bara en krycka för små modeller.
- **H5 (AST-specifik, från v1):** när spotlighten fångas av en iögonfallande distraktor och hamnar någon annanstans än agenten valt, anger schemat oftare den *avsedda* platsen än någon annan felaktig plats. Det kräver en fångstmekanism i miljön, så att avsedd och faktisk attention kan skilja sig åt.

**Mätvillkor för H5 (tillagt 21 okt efter granskning, låses i sprint 3):** i ungefär hälften av fångsterna ser foveán likadan ut på den fångade platsen som på den avsedda. Då kan ingen agent veta att den fångats, och att schemat då anger den avsedda platsen är bara en korrekt gissning, inte stöd för AST. H5 mäts därför så här:

- Bara fångster där foveán utesluter den avsedda platsen.
- Schemats angivelse läses av steget efter fångsten, när agenten har sett den fångade platsen.
- Fångster som råkar hitta målet räknas inte, eftersom episoden då tar slut.
- Jämförelsen görs inte mot ren slump. Den görs mot vad en idealisk betraktare med samma indata skulle gissa, och mot de andra iögonfallande platserna.

H5 stöds bara om schemat drar mot den avsedda platsen *trots* att informationen säger emot.

**Verifieringskriterier (förslag, tröskelvärden fastställs före körning):**

- [ ] H1 och H3 godkända om skillnaden består över minst fem träningsfrön.
- [ ] H2 godkänd om fokusförskjutningen följer det falska schemat signifikant oftare än slumpen.
- [ ] Dos–respons-kurvan är monoton.
- [ ] H4 bedöms över minst tre storlekar.

**Senare steg, 1c: framväxande schema.** Körs bara om 1a eller 1b visar effekt. Agenten får en fri intern kanal som bara tränas mot belöningen, utan facit, och kanalen har en kostnad per informationsmängd (flaskhals). Därefter mäts om kanalens innehåll beskriver agentens egen spotlight och om beskrivningen är förenklad. Ett schema som växer fram under urvalstryck, och blir en karikatyr av sig själv, är det mest AST-trogna utfallet. Kanalen byggs in i arkitekturen från början men hålls avstängd, så att 1c blir en konfigurationsändring.

**Möjlig senare utbyggnad: dold uppmärksamhet.** Agenten väljer inom ett fast blickfält vad som bearbetas djupare, utan att flytta spotlighten. Enligt Graziano är det där schemat gör störst skillnad.

## Fas 2: finns schemat i en verklig transformer?

Fas 2 körs bara om fas 1 visar att ett schema gör skillnad. Den undersöker om GPT-2 small, som saknar inbyggt schema, ändå bär en självbeskrivning av sin attention av den sort som fas 1 visat vara användbar. Fas 1 anger alltså vilken signatur som ska letas efter.

**Varför inte bara nästa lager:** attention i lager L+1 beräknas direkt ur residualströmmen efter lager L. Att förutsäga den därifrån mäter bara att uträkningen har sina indata, inte att det finns en självbeskrivning.

**Hypotes ****2a****:** en tokens *egen* vektor i lager L innehåller en förenklad beskrivning (bredd, skifte, grov position) av tokenens attention flera lager senare (L+k, k ≥ 2), utöver vad token-identitet och position förklarar.

**Hypotes ****2b**** (AST-specifik):** avvikelserna mellan förutsagt och faktiskt fokus är systematiska, till exempel att skiften underskattas eller att breda fördelningar beskrivs som smala.

**Metod:**

1. Kör GPT-2 small på en fast engelsk textkorpus och spara residualström och attention per lager.
2. Beräkna schemasammanfattningen per token och lager.
3. Träna linjära prober med enbart tokenens egen vektor i lager L som indata och sammanfattningen i lager L+k som mål, för k = 2, 4 och 6.
4. Kör två kontroller: en probe på bara token-identitet och position, och samma probe på en slumpinitierad GPT-2 med identisk arkitektur.
5. Analysera hur förutsägbarheten avtar med k och om felen har struktur.

**Verifieringskriterier (förslag, fastställs före körning):**

- [ ] 2a stöds om proben slår båda kontrollerna på hållout-data för k ≥ 2 i flera lager.
- [ ] 2b stöds om felens struktur är stabil mellan dataset, inte brus.
- [ ] Resultaten är reproducerbara med fasta frön.

**Tolkning:** även ett positivt utfall visar bara att informationen finns, inte att modellen använder den. Jämförelsen med det fungerande schemat från fas 1 avgör om fyndet liknar ett schema eller är en biprodukt.

**Kända fällor:**

- *Triviala huvuden:* vissa huvuden tittar alltid på föregående eller första token. Rapportera per huvudtyp och exkludera dem.
- *Positionsläckage:* inlärda positionsinbäddningar kan ensamma förutsäga mycket av fokus. Kontrollproben fångar det.

## Miljö och arkitektur

Allt körs på CPU i Claudes molnarbetsyta, med Python och NumPy (egenbyggd autodiff från v1, eftersom PyTorch inte går att installera). Fas 1 sluter slingan nedan: agenten ser miljön genom spotlighten, får schemat som egen insignal och väljer nästa spotlightläge. Fas 2 läser bara av en färdig modell.

[Diagram: fas 1 · schemat som återkopplad sinneskanal, se livedokumentet]

Ingreppen sker i schemakanalen: falskt schema, brus eller avstängning. Allt annat i slingan lämnas orört, så att en effekt kan härledas till schemat.

- **Fas 1:** egen sökmiljö och små agenter, så att träning med flera frön och tre storlekar ryms på CPU.
- **Fas 2:** GPT-2 small via TransformerLens, som ger direkt åtkomst till residualström och attention per lager. Kräver PyTorch, som i dag inte går att installera i arbetsytan; frågan måste lösas innan fas 2.
- **Spårbarhet:** fasta frön, konfiguration och resultat versioneras tillsammans med koden.

## Genomförande

Arbetet börjar med en MVP som besvarar huvudfrågan med tolv träningskörningar. Kärnan är ingreppet med falskt schema och H5, avsedd kontra faktisk attention, i etapper där produktägaren godkänner varje grind innan nästa etapp påbörjas.

**MVP-omfattning:**

| Ingår i MVP | Senare |
| --- | --- |
| Baslinje, Fast schema (1a), Inlärt schema (1b), Blandat schema | Framväxande schema (1c) |
| Falskt schema och lesion som ingrepp vid test, fångst av spotlighten och H5 | Brusserie (dos–respons) |
| En agentstorlek | Skalning över tre storlekar |
| Ledtrådsnivå svag | Nivåerna ingen och stark |
| Fast fönsterstorlek | Zoom och bredd, som första utbyggnad |
| Tre frön per villkor | Fem frön för bekräftelse |

**Etapper och grindar:**

1. **Miljö:** sökmiljön med enhetstester och en slumpagent. Grind: miljön beter sig enligt spec.
2. **Kompetens:** baslinjeagenten lär sig hitta målet klart bättre än slumpagenten. Grind: utan detta är alla jämförelser meningslösa.
3. **Pilot:** tidsåtgång per körning och variation mellan frön mäts. Grind: tröskelvärden och körplan föreslås, godkänns och låses.
4. **MVP-körning:** de tre villkoren tränas och ingreppen testas. Grind: resultat bedöms mot de låsta kriterierna.
5. **Utbyggnad:** zoom, 1b, skalning och övriga nivåer, i den ordning resultaten motiverar.

**Signifikanskriterium:** en skillnad räknas när den överstiger två standardavvikelser, ungefär 95 % säkerhet, och har samma riktning i alla frön. Med tre frön är spridningen osäkert skattad, så variationen mäts även över tusentals utvärderingsepisoder per agent. Kriteriet låses i etapp 3 och ändras inte efter att resultaten setts.

**Lagring och granskning:** kod och resultat versioneras i projektets GitHub-repo, eftersom arbetsytan töms mellan sessioner. Vid varje grind granskar en separat agent kod och resultat mot specen.

## Sprint 1: miljön

Miljön är byggd och beter sig enligt spec; den oberoende granskningen fann inga blockerare. Grinden väntar på produktägarens godkännande. Kod och resultat: v2/ i repot.

**Utförande:** rutnät 6–9 rutor i kvadrat, 4–16 distraktorer, spotlight 1×1 eller 2×2 (fast inom episoden), stegbudget 0,6–1,2 gånger antalet objekt. Periferin visar färg och iögonfallande objekt, aldrig form. Foveán visar färg och form men saknar koordinater, så agenten vet inte direkt var spotlighten är. Fångsten är graderad och styrs nerifrån: 0–3 distraktorer per episod är iögonfallande med olika styrka. Periferin visar styrkan med brus, och varje steg kan de dra till sig spotlighten med en chans som växer med styrkan. Vilket objekt som fångar, och när, går inte att veta i förväg. Alla agenter får en kopia av sitt föregående kommando (efferenskopia), men inte var spotlighten faktiskt hamnade. Målet hamnar i övre vänstra kvadranten i 57 % av fallen (statistisk struktur). 46 enhetstester, alla godkända.

**Referensagenter, standardinställningar, svag ledtråd (4 000 episoder vardera):**

| Agent | Hittar målet | Söklutning (steg per distraktor) | Steg vid 0 distraktorer |
| --- | --- | --- | --- |
| Slump | 27 % | ≈ 0 | 61 |
| Avsökning av objekt | 73 % | 0,50 | 1,0 |
| Färgstyrd | 78 % | 0,26 | 1,0 |

Söklutningen är mätt på 8×8, fönster 1, utan fångst och med obegränsad budget. Med stark ledtråd har den styrda agenten 0,08. Lutningen redovisas alltid med skärning, eftersom slumpagenten har nästan ingen lutning men behöver cirka 60 steg.

**Fångst och värdet av att känna sin uppmärksamhet** (färgstyrd agent, standardinställningar, cirka en fångst per episod):

| Vad agenten vet | Hittar målet |
| --- | --- |
| Bara vart den ville titta | 79 % |
| Härleder fångst ur foveán, inget schema | 86 % |
| Var spotlighten faktiskt hamnade (perfekt schema) | 93 % |

Efter ändringen till graderad fångst har ett perfekt schema 7,7 procentenheter att tillföra utöver vad en agent utan schema kan härleda, mot 3,1 med den gamla fångsten. Det är referensagenter; piloten i sprint 3 måste visa samma sak för inlärda agenter. Fångsten gör också uppgiften svårare för alla, därför lägre träffandelar än i förra versionen.

**Beslut för produktägaren före sprint 2–3:**

- [ ] Får alla agenter sin föregående handling som insignal (efferenskopia)? Den ger nästan ett schema gratis när ingen fångst sker, och avgör hur mycket schemat kan tillföra.
- [ ] svar: jag låter claude avgöra detta
- [ ] Ska fångsten bli mindre förutsägbar, till exempel till en av flera iögonfallande platser eller med högre fångstrisk, så att en agent utan schema inte kan härleda den lika lätt?
- [ ] svar: jag vill att experimentet så öångt som möjligt ska överensstämma med en evolutionär approach.
- [ ] H5 kräver ett schema som kan ha fel på eget sätt. Ett fast formelberäknat schema (1a) är antingen aldrig fel eller fel per konstruktion. Förslag: lyft in inlärt schema (1b) i MVP, eller mät H5 med en probe av agentens dolda tillstånd.
- [ ] svar: så realistiskt från biologiskt evolutionärt perspektiv som möjligt ifall det är förenligt med komplexitet av experimentet.
- [ ] Godkänn sprint 1-miljön (grind 1).
- [ ] svar: jag litar på claudes bedömning

**Beslut 21 okt 2026** (efter produktägarens svar: biologiskt realistiskt där det är förenligt med experimentets komplexitet):

1. **Efferenskopia: ja, i alla villkor.** Djur har en kopia av sina egna rörelsekommandon (corollary discharge). Agenten vet alltså vart den siktade, men inte var uppmärksamheten hamnade. Det är just den luckan schemat ska fylla.
2. **Fångsten görs graderad och styrd nerifrån**, som hos djur: flera iögonfallande objekt med olika styrka, brusigt synliga i periferin, och fångstchans som växer med styrkan. Infört och testat.
3. **Inlärt schema (1b) lyfts in i MVP**, eftersom H5 kräver ett schema som kan ha fel på eget sätt. MVP blir fyra villkor med tre frön vardera, alltså tolv körningar. Det mest evolutionära, ett schema som växer fram utan facit (1c), väntar till efter MVP på grund av komplexiteten. Ett förbehåll: 1b tränas mot facit om var spotlighten faktiskt var, vilket hjärnan inte får. Det är en förenkling.
4. **Grind 1 godkänd.** Produktägaren delegerade bedömningen. Två oberoende granskningar fann inga blockerare, och ändringarna ovan är testade (46 tester).

### Sprint 2: kompetens

Grindkriteriet nedan låstes 21 okt 2026 innan träningen startade. Baslinjen måste både slå slumpen och visa att den använder färgledtråd och minne, annars har schemat inget att tillföra i sprint 3.

Referenser, standardinställningar, svag ledtråd (3 × 4 000 episoder):

| Referensagent | Hittar målet |
| --- | --- |
| Slump | 28,4 % |
| Färgstyrd utan minne | 66,4 % |
| Färgstyrd utan minne, med kvadrantprior (bästa minneslösa) | 71,4 % |
| Färgstyrd med minne av avsikt | 79,5 % |
| Färgstyrd, härleder fångst | 86,1 % |
| Färgstyrd, perfekt schema | 93,5 % |

**Förslag till grind 2.** Tre träningsfrön. Varje frö utvärderas på 4 000 nya episoder med standardinställningar, testfrön skilda från träningen, med agentens egen stokastiska policy.

1. K1, klart bättre än slump: träffandelen överstiger slumpagentens med mer än 2 SD i alla tre frön (projektets signifikanskriterium).
2. K2, kompetens som gör sprint 3 meningsfull: träffandel minst 73 % i alla tre frön. Det ligger mer än 2 SE över den bästa minneslösa referensen (71,4 %; tidigare avrundat till 71,5), så agenten måste använda både färg och minne. Taket utan schema är ungefär 79–86 %.

Diagnostik som redovisas men inte avgör: medelretur, andel återsiktningar (siktar där den redan siktat), andel första blickar på målfärg, söklutning, fångster per episod.

Klarar baslinjen K1 men inte K2 redovisas det som ett delresultat, och arkitekturen eller träningen justeras innan sprint 3.

**Agentens uppbyggnad (beslut 21 okt).** En retinotop prioritetskarta, ungefär som colliculus superior: varje ruta har en liten egen återkopplad enhet (GRU) med delade vikter, plus ett globalt medelvärde av kartan som kontext. Varje ruta får periferin på sin plats, vad som söks, och foveáns innehåll omräknat till rutan via efferenskopian. Det motsvarar remappning med corollary discharge hos primater. Agenten vet alltså vart den siktade, inte var uppmärksamheten hamnade. Varje ruta ger ett värde för nästa sikte; en kritiker skattar förväntad retur. Träning: A2C med GAE och entropibonus.

Inbyggt men avstängt: schemakanalen (per ruta: faktiskt fönster förra steget och karta över besökta rutor, nollad i baslinjen) och 1b-huvudet, som förutsäger var spotlighten faktiskt var (förlustvikt 0 i baslinjen). Villkoren 1a, 1b och Blandat blir därmed konfigurationsändringar.

**Beslut 22 okt 2026 (produktägaren):** nästa steg efter MVP blir en fas med flera agenter som måste förutsäga varandras uppmärksamhet. Den prövar schemats sociala, spelteoretiska funktion, som MVP inte testar, och om ett schema växer fram lättare under socialt tryck än under ren styrning. Produktägarens tes: självmodellen är en föreställning som alla andra, men nödvändig av spelteoretiska skäl.

Skiss för den sociala fasen (förslag 22 okt, ej beslutad), i Conways anda: minsta möjliga regler, inget facit, och se vad som växer fram.

1. Två agenter söker i samma rutnät. Var och en ser grovt var den andras spotlight låg. Tävling: den som hittar målet först vinner. Samarbete: de delar belöningen.
2. Varje agent har en fri intern kanal med kostnad per informationsmängd (1c), tränad bara mot belöningen.
3. Jämförelse 2 × 2: ensam eller social, med eller utan kanal. Tesen att schemat är spelteoretiskt nödvändigt förutsäger att kanalen beskriver den egna uppmärksamheten tidigare och tydligare i det sociala villkoret.
4. Mät om kanalen beskriver den egna uppmärksamheten, den andras, eller båda med samma kod. Graziano förutsäger samma kod.
5. Möjligt: urval och mutation (evolution) i stället för gradientinlärning, i linje med den evolutionära principen, om det ryms på CPU.

**Beslut 22 okt 2026 (produktägaren):** ordningen blir MVP, sedan 1c för en ensam agent, sedan den sociala fasen. Den sociala fasen startar först när 1c visat något i enkel form, så att ett uteblivet resultat kan skiljas från en inlärning som inte fungerar.

Koppling till fas 2 (22 okt): en språkmodell är formad av den sociala vägen, att förutsäga människor som modellerar varandras uppmärksamhet, men har inte belönats för en korrekt modell av sin egen. Om tesen att det sociala spelet räcker stämmer, kan ett schema ha vuxit fram där som bieffekt. Modellens egna utsagor bevisar inget, eftersom den lärt sig människors schemaspråk. Därför mäter fas 2 de inre tillstånden, inte vad modellen säger. Ordningen ändras inte: fas 2 körs om fas 1 visar effekt.

**Resultat grind 2 (22 okt 2026): klarad, K1 och K2 i alla tre frön.** Tre frön à 1 miljon episoder, en körning per frö, inga kasserade försök. Testat på 4 000 nya episoder per frö.

| Frö | Hittar målet | Slump, samma episoder | z mot slump | K1 | K2 |
| --- | --- | --- | --- | --- | --- |
| 0 | 89,0 % | 28,4 % | 70 | ja | ja |
| 1 | 89,9 % | 27,7 % | 73 | ja | ja |
| 2 | 88,5 % | 27,6 % | 70 | ja | ja |

Baslinjen hittar målet lika ofta som den handkodade agenten som härleder fångst (86,1 %) eller något oftare, men klart under ett perfekt schema (93,5 %). Söklutning utan fångst 0,28 steg per distraktor, nära den färgstyrda referensen. Vikter vid slutet av träningen ger samma resultat som de valda (88,8 / 89,8 / 88,9 %). Kurvor: v2/results/sprint2.png.

**Oberoende granskning: godkänd med anmärkningar.** Inget informationsläckage: agenten fungerar oförändrat mot en miljö som bara ger observationerna, och tvingad fångst varje steg ger 5,6 %, samma som agenten som bara minns sin avsikt. Utan efferenskopia faller den till 63 %, så minnet går via efferenskopian som avsett. Matematiken kontrollerad mot finita differenser. Två punkter åtgärdade: referensen 71,4 % kan nu återskapas ur repot, och CLAUDE.md är uppdaterad. Hela rapporten: v2/results/sprint2_review.md.

**Följd för sprint 3 (Claudes bedömning, delad av granskaren):** schemat har lite att tillföra vid standardinställningarna, cirka 4 procentenheter. Det är v1:s fälla igen: agenten härleder fångsten ur vad den ser. Piloten bör först träna en agent med fast schema (1a) och mäta vinsten och om agenten alls läser kanalen. Är vinsten under cirka 3 procentenheter bör arbetspunkten flyttas innan trösklarna låses: bara fönster 1 (gap cirka 5 punkter), högre fångstrisk (cirka 10 punkter vid dubbel risk), eller en fovea som avslöjar mindre om var den hamnat. Det är en specändring som produktägaren beslutar. Granskaren noterade också att agenten lärt sig att aldrig sikta på iögonfallande objekt; H5:s nollmodell måste ta hänsyn till det.

**Beslut för produktägaren:** godkänn grind 2.

## Öppna frågor och beslut för produktägaren

- [x] Fas 1 (agenten) körs först och avgör huvudfrågan. Fas 2 (GPT-2) körs bara om fas 1 visar effekt.
- [x] Fas 2 använder engelsk textkorpus.
- [x] Agenten i fas 1 tränas med förstärkningsinlärning: belöning vid träff, avdrag per steg.
- [x] Rutnätets storlek, fönster, stegbudget och antal distraktorer varierar slumpmässigt mellan episoder.
- [x] Tröskelvärden för verifieringskriterierna fastställs innan första körningen.
- [x] Spec och resultat förs i detta dokument.
- [x] Zoom och bredd tas med som första utbyggnad efter MVP. MVP har fast fönster, och schemat anger läge, förflyttning och besökta områden.
- [ ] svar:zoom och bredd verkar vara viktuga faktorer
- [ ] vi väntar med detta om detta leder till alltför äkad komplexitet
- [x] Ledtrådarnas styrka: tre nivåer, ingen, svag och stark. MVP kör nivån svag, övriga senare.
- [ ] Svar: förstår inte

*Förklaring av ledtrådsnivåerna:* nivåerna är olika svåra versioner av samma uppgift. Vid *ingen* visar periferin bara att det finns objekt, inte deras färg. Vid *svag* visar periferin färg, men de flesta distraktorer har målets färg, så färgen hjälper lite. Vid *stark* har få distraktorer målets färg, så färgen pekar nästan ut målet. MVP kör den svaga nivån, där agenten måste hålla reda på många likadana kandidater och schemat har mest att göra.
