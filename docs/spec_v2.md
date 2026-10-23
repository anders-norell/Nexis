<!-- Ögonblicksbild av det levande dokumentet "Nexis v2 – spec" (https://claude.ai/artifact/GsEmfNtb1vs2bhXxwc6vsW), exporterad 23 okt 2026 vid grind 5a (beslut väntar), rev 339. Vid skillnad gäller det levande dokumentet. -->

# Nexis v2 – spec

Oct 20, 2026 · @Anders Norell

## Läge just nu

Vi är i fas 1, agentexperimentet. Sprint 3, piloten, är klar och grind 3 godkänd, så trösklarna är låsta. Sprint 4, MVP, är klar och grind 4 godkänd 23 okt: F, H2 och H3 klaras, H1 klaras inte. Planen för sprint 5a, H5, väntar på ditt godkännande. Fas 2, GPT-2-proben, körs bara om fas 1 visar effekt.

Begreppen förklaras i [Nexis – ordlista](https://claude.ai/artifact/2syuVpzf2GcAfaP9DgSgjC).

| Sprint | Innehåll | Status |
| --- | --- | --- |
| 0 | Spec | Klar |
| 1 | Miljön | Klar, grind 1 godkänd 21 okt |
| 2 | Kompetens: baslinjeagent utan schema | Klar, grind 2 godkänd 22 okt |
| 3 | Pilot: arbetspunkt, spridning, trösklar | Klar, grind 3 godkänd 22 okt, trösklar låsta |
| 4 | MVP: Baslinje, 1a och Blandat; F, H1, H2, H3 | Klar, grind 4 godkänd 23 okt: F, H2, H3 klarade, H1 ej |
| 5 | H5: schemamodul med lokal matchning, egen pilot | Plan 5a väntar på ditt godkännande |
| 6 | Utbyggnad, sedan 1c och social fas | Ej påbörjad |

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
| [Wilterson & Graziano 2021](https://dataspace.princeton.edu/handle/88435/dsp01kp78gk430) | Agent som fångar en boll med en spotlight. Ett givet schema förbättrar styrningen, och att stänga av det efter träning försämrar den kraftigt. | Fast schema (1a) och lesion är replikation. |
| [Piefke, Doerig, Kietzmann & Thorat 2024](https://arxiv.org/abs/2402.01056) | Ett schema som inte är förprogrammerat växer fram ur belöningsinlärning och bär information om spotlightens läge. Det är en ledtråd, inte en kopia, och gör mest nytta vid måttlig störning. | Framväxande schema (1c) är delvis gjort. Störningsnivå motsvarar vår ledtrådsvrid. |
| [Farrell, Ziman & Graziano 2024](https://arxiv.org/abs/2411.00983) | Med transformer-attention hjälper ett schema att bedöma andras attention och att samarbeta, utan att förklaras av extra kapacitet. Schemat tränas med uttryckligt facit. | Den sociala funktionen, som vi inte testar. |
| [ASAC 2025](https://arxiv.org/abs/2509.16058) | Ett schema inbyggt som styrenhet i transformerns attention. | Nära fas 2, men som konstruktion, inte som undersökning av en färdig modell. |

**Det som inte tycks vara gjort** (bedömt utifrån studierna ovan, ingen fullständig litteraturgenomgång):

- Falskt schema som kausalt ingrepp: flyttas den verkliga attention när schemat ljuger?
- Förregistrerat test av schemats systematiska fel, att schemat drar mot *avsedd* snarare än *faktisk* attention.
- Schema över attention-*historik* i en sökuppgift med återbesök, mätt med söklutning.
- Flaskhals som gör schemat förenklat, så att karikatyren uppstår i stället för att konstrueras.

**Lärdomar från v1 (oktober 2026, [repo](https://github.com/anders-norell/nexis)):**

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

- **Fast schema (1a):** vi räknar ut sammanfattningen av spotlightens läge och historik med en fast formel och matar in den. Agenten lär sig läsa ett färdigt sinne.
- **Inlärt schema (1b):** agenten tränas också att förutsäga sin egen spotlightsammanfattning, med det faktiska läget som facit, och den egna förutsägelsen matas tillbaka. Schemat blir då förenklat på agentens egna villkor och kan vara systematiskt fel, vilket ligger närmare AST.

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

**Hypotes 2a:** en tokens *egen* vektor i lager L innehåller en förenklad beskrivning (bredd, skifte, grov position) av tokenens attention flera lager senare (L+k, k ≥ 2), utöver vad token-identitet och position förklarar.

**Hypotes 2b (AST-specifik):** avvikelserna mellan förutsagt och faktiskt fokus är systematiska, till exempel att skiften underskattas eller att breda fördelningar beskrivs som smala.

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

&#91;embedded content: fas 1 · schemat som återkopplad sinneskanal\]

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

**Signifikanskriterium:** en skillnad räknas när den överstiger två standardfel, ungefär 95 % säkerhet, och har samma riktning i alla frön. Standardfelet gäller skillnaden och räknas över de 4 000 testepisoderna inom varje frö. (Rättat 22 okt: tidigare stod det standardavvikelser, men beräkningarna har hela tiden gjorts med standardfel.) Kriteriet låses i etapp 3 och ändras inte efter att resultaten setts.

**Lagring och granskning:** kod och resultat versioneras i projektets GitHub-repo, eftersom arbetsytan töms mellan sessioner. Vid varje grind granskar en separat agent kod och resultat mot specen.

## Sprint 1: miljön

Miljön är byggd och beter sig enligt spec; den oberoende granskningen fann inga blockerare. Grinden väntar på produktägarens godkännande. Kod och resultat: `v2/` i [repot](https://github.com/anders-norell/nexis).

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

## Sprint 2: kompetens

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

1. K1, klart bättre än slump: träffandelen överstiger slumpagentens med mer än 2 standardfel i alla tre frön (projektets signifikanskriterium; stod tidigare felaktigt SD).
2. K2, kompetens som gör sprint 3 meningsfull: träffandel minst 73 % i alla tre frön. Det ligger mer än 2 SE över den bästa minneslösa referensen (71,4 %; tidigare avrundat till 71,5), så agenten måste använda både färg och minne. Taket utan schema är ungefär 79–86 %.

Diagnostik som redovisas men inte avgör: medelretur, andel återsiktningar (siktar där den redan siktat), andel första blickar på målfärg, söklutning, fångster per episod.

Klarar baslinjen K1 men inte K2 redovisas det som ett delresultat, och arkitekturen eller träningen justeras innan sprint 3.

**Agentens uppbyggnad (beslut 21 okt).** En retinotop prioritetskarta, ungefär som colliculus superior: varje ruta har en liten egen återkopplad enhet (GRU) med delade vikter, plus ett globalt medelvärde av kartan som kontext. Varje ruta får periferin på sin plats, vad som söks, och foveáns innehåll omräknat till rutan via efferenskopian. Det motsvarar remappning med corollary discharge hos primater. Agenten vet alltså vart den siktade, inte var uppmärksamheten hamnade. Varje ruta ger ett värde för nästa sikte; en kritiker skattar förväntad retur. Träning: A2C med GAE och entropibonus.

Inbyggt men avstängt: schemakanalen (per ruta: faktiskt fönster förra steget och karta över besökta rutor, nollad i baslinjen) och 1b-huvudet, som förutsäger var spotlighten faktiskt var (förlustvikt 0 i baslinjen). Villkoren 1a, 1b och Blandat blir därmed konfigurationsändringar.

**Resultat grind 2 (22 okt 2026): klarad, K1 och K2 i alla tre frön.** Tre frön à 1 miljon episoder, en körning per frö, inga kasserade försök. Testat på 4 000 nya episoder per frö.

| Frö | Hittar målet | Slump, samma episoder | z mot slump | K1 | K2 |
| --- | --- | --- | --- | --- | --- |
| 0 | 89,0 % | 28,4 % | 70 | ja | ja |
| 1 | 89,9 % | 27,7 % | 73 | ja | ja |
| 2 | 88,5 % | 27,6 % | 70 | ja | ja |

Baslinjen hittar målet lika ofta som den handkodade agenten som härleder fångst (86,1 %) eller något oftare, men klart under ett perfekt schema (93,5 %). Söklutning utan fångst 0,28 steg per distraktor, nära den färgstyrda referensen. Vikter vid slutet av träningen ger samma resultat som de valda (88,8 / 89,8 / 88,9 %). Kurvor: v2/results/sprint2.png.

**Oberoende granskning: godkänd med anmärkningar.** Inget informationsläckage: agenten fungerar oförändrat mot en miljö som bara ger observationerna, och tvingad fångst varje steg ger 5,6 %, samma som agenten som bara minns sin avsikt. Utan efferenskopia faller den till 63 %, så minnet går via efferenskopian som avsett. Matematiken kontrollerad mot finita differenser. Två punkter åtgärdade: referensen 71,4 % kan nu återskapas ur repot, och CLAUDE.md är uppdaterad. Hela rapporten: v2/results/sprint2\_review.md.

**Följd för sprint 3 (Claudes bedömning, delad av granskaren):** schemat har lite att tillföra vid standardinställningarna, cirka 4 procentenheter. Det är v1:s fälla igen: agenten härleder fångsten ur vad den ser. Piloten bör först träna en agent med fast schema (1a) och mäta vinsten och om agenten alls läser kanalen. Är vinsten under cirka 3 procentenheter bör arbetspunkten flyttas innan trösklarna låses: bara fönster 1 (gap cirka 5 punkter), högre fångstrisk (cirka 10 punkter vid dubbel risk), eller en fovea som avslöjar mindre om var den hamnat. Det är en specändring som produktägaren beslutar. Granskaren noterade också att agenten lärt sig att aldrig sikta på iögonfallande objekt; H5:s nollmodell måste ta hänsyn till det.

**Grind 2 godkänd av produktägaren 22 okt 2026 (i chatten: "då kör vi vidare med nästa sprint"). Kvar att besluta vid start av sprint 3:** tillägget för sprint 3 (förslag 22 okt): ny hypotes H6, schemats vinst växer när foveán avslöjar mindre om var uppmärksamheten hamnade. Piloten varierar hur avslöjande foveán är, och fångstrisken, tränar Baslinje och Fast schema (1a) i varje läge, och väljer arbetspunkt där 1a ger en tydlig vinst innan trösklarna låses. Ger 1a ingen vinst ens med tvetydig fovea krävs ett omtag av uppgiften.

## Sprint 3: pilot

### Plan (godkänd 22 okt)

Piloten ska svara på en fråga före alla andra: finns det en arbetspunkt där schemat verkligen behövs? Vid standardinställningarna har schemat högst cirka 4 procentenheter att tillföra (baslinje 89 %, perfekt schema 93,5 %). Piloten mäter också spridning mellan frön och tid per körning, så att trösklarna för MVP kan föreslås ur data. Inget körs förrän du har godkänt planen.

**Varför foveán står i centrum.** I den här uppgiften bär foveán nästan bara en sak: var spotlighten hamnade. En träff avslutar episoden av sig själv, så agenten behöver aldrig granska innehållet för att hitta målet. Det enda innehållet duger till är att jämföras med periferins karta, så att agenten kan räkna ut var den tittar. Foveán läcker alltså just den information schemat ska ge. Det förklarar varför baslinjen klarar sig nästan lika bra utan schema.

**Vad piloten ska ta reda på**

| Fråga | Varför | Om svaret blir nej |
| --- | --- | --- |
| 1. Ger Fast schema (1a) minst 3 procentenheter högre träffandel än baslinjen vid någon tillåten arbetspunkt? | Annars mäter MVP ett schema agenten klarar sig utan, v1:s fälla. | Omtag av uppgiften före MVP. Jag beskriver alternativen, du beslutar. |
| 2. Läser agenten kanalen? Lesion av 1a sänker träffandelen minst 3 punkter, och Blandat schema ger ingen vinst. | En vinst kan komma av extra indata eller kapacitet i stället för av schemats innehåll. | Jag söker ett tekniskt fel (t.ex. kanalens skalning) i ett försök. Hjälper det inte rapporterar jag, och du beslutar. |
| 3. Är spridningen mellan frön så liten att tre frön per villkor kan visa en skillnad på 2 standardfel? | MVP har tre frön per villkor. | Jag föreslår fem frön, ca 2/3 mer räknetid. Ditt beslut. |
| 4. Går H5 att mäta vid arbetspunkten: lär sig 1b var spotlighten är, och finns det tillräckligt många fångster där foveán utesluter den avsedda platsen? | H5 är MVP:s kärna och får varken vara sann eller omätbar per konstruktion. | En annan av de tillåtna arbetspunkterna. Finns ingen beslutar du om H5 skjuts upp. |
| 5. Ryms MVP:s tolv körningar i rimlig tid? | Två kärnor, ca 50 minuter per körning i sprint 2. | Kortare körningar om kurvorna tillåter, annars längre kalendertid. |

**Steg**

1. Förberedelse, ingen träning (ca 1 h arbete): reglage för hur mycket foveán avslöjar, med utökade tester; en idealisk betraktare som nollmodell för H5; mätskript för lesion och Blandat schema. Pushas innan något tränas.
2. Rutnät av arbetspunkter (ca 3,5 h körtid i bakgrunden): Baslinje och 1a, ett frö var, 400 000 episoder per körning. Foveán i fyra lägen (visar allt, hälften av stegen, en fjärdedel, inget) gånger två fångstrisker (standard 0,25 och dubbel 0,5), plus bara fönster 1 vid full fovea. 18 körningar. Standardpunkten körs först.
3. Val av arbetspunkt enligt regeln nedan. Valet skrivs in här och pushas innan steg 4.
4. Bekräftelse vid vald punkt (ca 3 h körtid): tre nya frön var för Baslinje, 1a och Blandat schema, plus ett frö 1b för att avgöra 1b:s utformning och kontrollera att H5 är mätbar.
5. Förslag på trösklar och körplan för MVP förs in här med underlag. En separat agent granskar kod och resultat. Grind 3: du godkänner, och trösklarna låses.

**Regel för arbetspunkten (bestäms nu, före körning):** den punkt som ligger närmast standardinställningarna där 1a vinner minst 3 punkter över baslinjen och lesion kostar minst 3 punkter. Närmast betyder i tur och ordning: mest avslöjande fovea, lägst fångstrisk, båda fönsterstorlekarna kvar. Regeln hindrar mig från att välja den punkt som råkar ge snyggast resultat.

**Vad piloten inte gör:** den kör aldrig falskt schema (H2) eller H5-testet, och inga pilotkörningar räknas i MVP, som tränas med nya frön. Lesion körs eftersom fråga 2 kräver den. Den ger en fingervisning om H3, och det redovisar jag öppet.

**Öppet beslut: tillägget H6.** Förslaget är att piloten varierar hur mycket foveán avslöjar, och att vi formulerar en hypotes: schemats vinst växer när foveán avslöjar mindre om var uppmärksamheten hamnade.

Min bedömning: ja till att variera foveán, men H6 bör vara beskrivande, inte ett godkännandekriterium.

- **För:** foveán är just den läcka som gör schemat överflödigt, så att dämpa den är det mest direkta sättet att göra uppgiften schemakrävande. Det har en biologisk motsvarighet: korta fixeringar och maskering ger en ofullständig bild av det man tittar på, och då måste hjärnan veta var den tittar på annat sätt. Kurvan över fyra lägen säger också något i sig: hur schemats värde beror på hur mycket världen själv avslöjar.
- **Emot:** H6 är nästan sann per definition. Mindre information i foveán gör all annan information om läget mer värd, så en bekräftad H6 säger lite om AST. Ju mer vi ändrar uppgiften tills schemat hjälper, desto större risk att vi bygger fram effekten. Därför regeln ovan.
- **Gräns:** en helt blind fovea får inte bli arbetspunkt. Då har inte ens en idealisk betraktare något att gå på och gissar alltid den avsedda platsen. H5 blir omätbar, eftersom det inte finns någon information som schemat kan gå emot. Läget körs i piloten bara som ytterpunkt på kurvan.

Alternativ:

- **A (min rekommendation):** variera foveán i piloten, redovisa H6 som beskrivande kurva, välj arbetspunkt enligt regeln, helt blind fovea utesluten.
- **B:** som A, men H6 blir en förregistrerad hypotes med tröskel som låses vid grind 3.
- **C:** variera inte foveán. Arbetspunkten flyttas bara med fångstrisk och fönsterstorlek. Sex körningar i steg 2 i stället för 18, men större risk att ingen punkt ger tydlig vinst.
- **D:** ingen flytt. MVP körs vid standardinställningarna med liten marginal. Jag avråder, det är v1:s fälla.

**Beslut för produktägaren** (skriv svar: under respektive punkt)

- [x] Godkänner du planen ovan?
- [x] H6: A, B, C eller D?
- [x] Får arbetspunkten flyttas med alla tre reglagen: foveá, fångstrisk och fönsterstorlek?
- [x] Gränsen för tydlig vinst och för att kanalen läses: 3 procentenheter?
- [x] Räknetid ca 7 timmar, troligen över två sessioner. Går det bra?

**Beslut 22 okt 2026 (produktägaren, i chatten):** alla fem punkterna godkända. H6 enligt alternativ A. Claude får välja den väg framåt som har störst sannolikhet att ge ett lyckat utfall. Claudes tolkning, för protokollet: lyckat betyder ett giltigt och informativt experiment, inte att hypoteserna bekräftas. Regeln för arbetspunkten gäller oförändrad, och alla resultat rapporteras.

**Tillägg till motiveringen för A (22 okt):** uppmärksamhet, i meningen urval av vad som bearbetas, är äldre än foveor. Hos fiskar och groddjur styr tectum vart ögon och huvud riktas, och läget av den egna uppmärksamheten kommer där från motorkommandon och förutsägelser snarare än från bilden. Grazianos evolutionära berättelse börjar där. Frågan är alltså inte om agenten har en fovea, utan hur mycket det utvalda innehållet avslöjar om var det kommer ifrån. Den nuvarande miljön är i det avseendet orealistiskt generös, så en fovea som avslöjar mindre är både mer mätbar och mer lik den situation där ett schema rimligen uppstod.

**Tekniska val (Claudes, redovisas men kräver inte beslut)**

- Foveáns reglage: varje steg visas innehållet med en viss sannolikhet. Annars visas bara vilka rutor fönstret täcker, som en maskerad blick. Koordinater ges aldrig.
- Pilotkörningar på 400 000 episoder, samma längd i alla villkor. Baslinjen nådde ca 87 % efter 200 000 episoder i sprint 2. MVP:s längd föreslås efter piloten.
- 1b: huvudet förutsäger förra stegets faktiska fönster, och förutsägelsen matas in nästa steg, som en efterhandsrapport. Bekräftas med enfrö-körningen i steg 4.
- Idealisk betraktare för H5: räknar sannolikheten för varje möjlig plats (den avsedda eller den som var och en av de iögonfallande objekten drar till) ur efferenskopian, den brusiga salience och foveáns innehåll. Den tar hänsyn till att agenten aldrig siktar på iögonfallande objekt.
- Lesion: schemakanalen nollas vid test. Blandat schema: tränat med en annan episods schema, som i specen.
- 4 000 testepisoder per frö, frön körs parallellt, allt loggas till fil och pushas löpande.

### Resultat, steg 1–4

**Steg 1 klart (22 okt 02:25):** reglaget för foveán, den idealiska betraktaren och mätskripten är testade och pushade. Med reglaget i läget "visar allt" är miljön bitidentisk med sprint 2.

**Steg 2: rutnätet klart (22 okt 03:30). H6 får inget stöd: vinsten beror på fångstrisken, inte på foveán.** Ett frö per punkt, 400 000 episoder, 4 000 testepisoder, standardfel cirka 0,6–0,9 punkter.

| Foveán visar | Fångstrisk | Baslinje | 1a | Vinst | Lesion kostar |
| --- | --- | --- | --- | --- | --- |
| allt | 0,25 (standard) | 87,5 % | 94,7 % | 7,2 | 44 |
| hälften | 0,25 | 88,0 % | 94,8 % | 6,8 | 35 |
| en fjärdedel | 0,25 | 87,5 % | 95,1 % | 7,7 | 39 |
| inget | 0,25 | 86,9 % | 94,5 % | 7,6 | 38 |
| allt | 0,5 | 76,2 % | 87,2 % | 11,0 | 46 |
| hälften | 0,5 | 76,2 % | 88,3 % | 12,1 | 35 |
| en fjärdedel | 0,5 | 77,8 % | 87,9 % | 10,1 | 38 |
| inget | 0,5 | 76,2 % | 87,3 % | 11,1 | 33 |
| allt, bara fönster 1 | 0,25 | 83,0 % | 93,2 % | 10,3 | 54 |

**Det här motsäger min förklaring om foveán.** Baslinjen klarar sig lika bra med helt blind fovea som med full. Den använder alltså inte foveán för att räkna ut var den hamnat, och läckan jag beskrev spelar ingen roll i praktiken. Det jag skrev om foveán som orsak till v1:s fälla stämmer inte för v2.

**Vad schemat faktiskt används till (diagnostik, frö 0, standardpunkt):** när bara kanalen "fönstret förra steget" nollas är träffandelen oförändrad (94,7 %). När kartan "täckt hittills" nollas faller den till 80,4 %. Utan fångst skiljer 1a och baslinjen nästan inget (98,9 mot 98,1 %). Tolkning: baslinjen minns var den *siktade* via efferenskopian. Vid fångst blir det minnet fel, och den lämnar rutor osökta. Schemat ger en karta över var uppmärksamheten *faktiskt varit*, och det är den agenten använder. Det är ett schema i AST:s mening, men om historiken, inte om läget just nu.

**Följd för MVP:** falskt schema (H2) måste riktas mot kartan över täckta rutor för att kunna flytta fokus. Ett falskt "fönster förra steget" skulle agenten troligen bara ignorera. Det förs in i tröskelförslaget i steg 5.

**Steg 3: arbetspunkt vald (22 okt 02:40): standardinställningarna.** Regeln tar den punkt närmast standard som klarar kraven, och redan standardpunkten klarar dem. Ett frö, 400 000 episoder, 4 000 testepisoder:

| Villkor | Hittar målet |
| --- | --- |
| Baslinje | 87,5 % |
| Fast schema (1a) | 94,7 % |
| 1a med schemat nollat vid test (lesion) | 50,5 % |

Vinsten är 7,2 ± 0,6 procentenheter (krav ≥ 3) och lesionen kostar 44 punkter (krav ≥ 3). Agenten lutar sig alltså tungt mot schemat när den har det, och klarar sig sämre än baslinjen utan det.

**Rättelse av min tidigare bedömning:** jag skrev att schemat bara hade cirka 4 punkter att tillföra, eftersom den handkodade agenten med perfekt schema nådde 93,5 %. Den referensen var inte optimal. En inlärd agent med schema når 94,7 %. Läckan via foveán finns, men baslinjen utnyttjar den inte fullt ut, och fallet blev inte v1:s fälla. Resten av rutnätet körs ändå klart som beskrivande kurva för H6, men det ändrar inte valet. Valet kan bara ändras om H5 visar sig omätbar vid standardpunkten (fråga 4), vilket prövas i steg 4.

**Steg 4: bekräftelse vid standardpunkten (22 okt 04:10). Effekten består i alla frön, och spridningen är liten.** 400 000 episoder per körning, 4 000 testepisoder per frö, standardfel 0,35–0,55 punkter. Frö 0 är pilotens, frö 1–3 är nya.

| Mått | Frö 0 | Frö 1 | Frö 2 | Frö 3 |
| --- | --- | --- | --- | --- |
| Baslinje, hittar målet | 87,5 % | 86,7 % | 87,1 % | 86,4 % |
| Fast schema (1a) | 94,7 % | 94,9 % | 95,0 % | 94,2 % |
| Blandat schema | – | 87,4 % | 88,3 % | 86,4 % |
| 1a med schemat nollat (lesion) | 50,5 % | 47,6 % | 51,2 % | 57,9 % |
| 1a, kartan "täckt hittills" nollad | 80,4 % | 82,4 % | 77,6 % | 83,0 % |
| 1a, "fönstret förra steget" nollat | 94,7 % | 94,5 % | 94,7 % | 94,2 % |
| Återbesök, baslinje / 1a / 1a lesion | 16 / 11 / 63 % | 16 / 11 / 65 % | 16 / 11 / 61 % | 17 / 11 / 55 % |
| Söklutning utan fångst, baslinje / 1a (steg per distraktor) | 0,31 / 0,27 | 0,31 / 0,28 | 0,29 / 0,26 | 0,33 / 0,28 |

Svar på pilotens frågor: (1) ja, vinsten är 7,2–8,2 punkter i varje frö; (2) ja, agenten läser kanalen, men bara kartan över täckta rutor; Blandat schema ger ingen vinst och nollas utan effekt; (3) ja, spridningen mellan frön är under en procentenhet, så tre frön räcker; (5) ja, en körning tar 7–8 minuter vid 400 000 episoder. Fråga 4, H5, besvaras i avsnittet om 1b nedan.

Två iakttagelser till. Lesion ger ett tydligt H3-likt mönster: agenten fortsätter röra sig men återbesöker i över hälften av stegen. Söklutningen utan fångst skiljer bara cirka 0,04 steg, eftersom schemat knappt behövs utan fångst (98,9 mot 98,1 % träff). H1 bör därför mätas med fångst påslagen (förslag i steg 5).

### 1b och H5

**1b: agenten har ingen nytta av sitt inlärda schema. H5 går att mäta, men resultatet riskerar att följa av arkitekturen. Det här är det viktigaste beslutet vid grind 3.**

| 1b-variant (frö 1, standardpunkt) | Hittar målet | Schemat nollat | Skillnad |
| --- | --- | --- | --- |
| Delat huvud, vikt 1,0 (som i specen) | 82,8 % | 82,8 % | 0,0 |
| Delat huvud, vikt 0,3 | 85,2 % | 84,5 % | 0,7 |
| Delat huvud, vikt 0,1 | 89,0 % | 86,8 % | 2,1 |
| Separat schemamodul (frö 1 / frö 2) | 88,4 / 88,8 % | 86,6 / 86,5 % | 1,8 / 2,3 |

Alla varianter ligger nära baslinjen (87 %), långt under 1a (95 %). Schemamodulen är ett eget litet nätverk som bara tränas att följa var uppmärksamheten faktiskt var, och dess karta ges till agenten med samma timing som 1a. Den har rätt i 99,9–100 % av stegen utan fångst. Ändå ger den ingen vinst. Det betyder att den inte fångar upp fångsterna, för det är just där 1a hjälper. H5-fall finns det gott om (4 100–4 500 per 4 000 episoder), men vart schemat pekar i dem har jag inte tittat på.

**Varför, enligt min bedömning.** Agentens arkitektur lägger det foveán visar på de rutor agenten *siktade* mot, via efferenskopian. För att upptäcka att den hamnat någon annanstans måste en ruta jämföra blicken med en annan rutas innehåll, och det kan rutorna bara göra via ett svagt globalt medelvärde. Det förklarar också att H6-kurvan blev platt: foveáns innehåll används knappt till att lokalisera. Följden för H5 är allvarlig. Om schemat pekar på den avsedda platsen kan det bero på att arkitekturen gör det svårt att göra något annat, inte på att ett inlärt schema är förenklat på det sätt AST förutsäger. Granskaren påpekar att jag överdriver: att blicken inte stämmer med den avsedda rutan kan rutan själv se, bara var uppmärksamheten hamnade kräver det globala medelvärdet. Två andra förklaringar är inte uteslutna: modulens karta är sannolikheter (ca 0,25 per ruta för ett 2×2-fönster mot 1 i 1a), och det delade huvudet ligger ett steg efter 1a. Arkitekturen är alltså en trolig delförklaring, inte bevisad. H5-sprinten måste pröva alla tre.

En ärlig anmärkning: att 1b inte ger någon vinst antyder redan åt vilket håll H5 skulle gå. Det har jag inte kunnat undvika att se.

**Alternativ (ditt beslut):**

- **A (min rekommendation):** MVP körs nu med kärnan som pilotdata stöder starkt: Baslinje, 1a och Blandat schema, med H1, H2 och H3. H5 flyttas till en egen sprint direkt efter. Där får schemamodulen lokal information om var blicken passar in (för varje ruta: stämmer det foveán visar med det periferin visar här?), och den varianten pilotas innan H5:s tröskel låses. Då prövar H5 om schemat följer avsikten trots att informationen finns lätt tillgänglig, vilket är den intressanta frågan.
- **B:** MVP enligt ursprunglig plan med schemamodulen som 1b. H5 mäts nu, men rapporteras med förbehållet ovan.
- **C:** bygg om först: matchningsinformationen till modulen, en ny pilot, sedan hela MVP. Ungefär en extra session.

A ger snabbast ett giltigt huvudresultat (H2, schemat styr uppmärksamheten) och skyddar H5 från en svag tolkning. Det ligger också närmast Grazianos bild: schemat som en egen modell med egna indata.

### Förslag: trösklar och körplan för MVP

**Steg 5: förslag på trösklar och körplan för MVP (låses vid grind 3, gäller alternativ A).**

Körplan: Baslinje, Fast schema (1a) och Blandat schema, tre nya frön vardera (10, 11, 12), nio körningar. 1 000 000 episoder per körning som i sprint 2, eftersom baslinjen fortfarande förbättras något efter 400 000; det gör jämförelsen försiktig. Ca 20 minuter per körning, under två timmar totalt. Test på 4 000 nya episoder per frö, samma episoder i alla villkor, agentens egen stokastiska policy.

Signifikans tolkas som i specen: skillnaden är större än 2 standardfel (från testepisoderna, inom fröet) och går åt samma håll i alla tre frön.

| Kriterium | Mått | Godkänt om | Pilotdata |
| --- | --- | --- | --- |
| F, förutsättning | Hittar målet | 1a högre än både Baslinje och Blandat | +7,2 till +8,2 punkter, ca 11 standardfel |
| H1 | Söklutning med fångst: 8×8, fönster 1, 4–16 distraktorer, steg bara i episoder där målet hittas | 1a flackare än både Baslinje och Blandat | Frö 1: 1a 0,35, Baslinje 0,38, Blandat 0,36. Troligen ingen skillnad mot Blandat |
| H2 | Falskt schema i kartan "täckt hittills" inför fjärde kommandot (se H2 i detalj) | (a) och (b) flyttar siktet i förutsagd riktning, och mer än hos Blandat | Ej mätt i piloten |
| H3a, lesion under hela träningen | Baslinjen, som är samma agent tränad med schemat nollat hela tiden, jämförd med 1a | Baslinjen hittar målet mer sällan och återbesöker oftare | 86–88 % mot 94–95 %; återbesök 16–17 % mot 11 % |
| H3b, akut lesion vid test | 1a med schemat nollat vid test, jämförd med intakt 1a | Träffandelen faller och andelen återbesök ökar | 48–58 % mot 94–95 %; återbesök 55–65 % mot 11 % |

**H3 omformulerad (22 okt, produktägarens villkor).** Lesion bara vid test är ett svagt prov: nästan vilken insignal som helst som ett nät vant sig vid gör skada när den tas bort. Därför prövas H3 på två sätt. H3a är lesion under hela träningen. Det är exakt Baslinjen, samma arkitektur med schemakanalen nollad från start, så inga nya körningar behövs. H3b är den akuta lesionen vid test. H3 är godkänd om både H3a och H3b klaras. Att den akut lesionerade agenten blir sämre än Baslinjen redovisas också. Det visar att agenten byggt sin strategi kring schemat, men det är inget kriterium. Möjligt tillägg som inte ingår: lesion efter halva träningen med fortsatt träning, för att se om agenten återhämtar sig. Det kostar tre körningar, ungefär en timme.

**Fröbank för testepisoderna (krav före körning).** Alla tester i MVP körs på samma fasta bank av 4 000 episoder. Episod nummer i har ett eget frö (2 000 000 + i) med två separata slumpströmmar, en för miljön och en för agentens val. Miljöns ström bestämmer rutnät, föremål, iögonfallande distraktorer, fångster, bruset i periferin och foveáns reglage. Samma episod börjar därför exakt likadant i alla villkor och frön, och vid H2 är förloppen med och utan ingrepp identiska fram till ingreppet. Kontroll innan något körs: ett test visar att två körningar av samma episod ger identiska förlopp, och att förloppen med och utan ingrepp är lika fram till steg 3.

**H2 i detalj (låses vid grind 3).** H2 prövas på 1a, med Blandat schema som kontroll. Agenten använder kartan över täckta rutor, inte kanalen för förra stegets fönster, så lögnen läggs i kartan.

- **Tidpunkt:** i schemat som agenten får inför sitt fjärde kommando, i episoder som fortfarande pågår efter tre steg.
- **Varaktighet:** lögnen består resten av episoden. Ny täckning efter ingreppet registreras som vanligt.
- **(a) Falskt obesökt:** markeringen tas bort för alla rutor agenten faktiskt täckt under de tre första stegen. Förutsägelse: fler av de tre följande kommandona siktar dit.
- **(b) Falskt besökt:** upp till tre otäckta rutor med ett föremål i målets färg markeras som täckta, slumpvis valda ur episodens frö. Själva målet väljs aldrig. Finns färre än tre används de som finns, och finns ingen räknas episoden inte i (b). Förutsägelse: färre kommandon siktar dit.
- **Siktar dit:** kommandots avsedda fönster överlappar minst en av de manipulerade rutorna. Det gäller både fönster 1 och 2×2.
- **Mått:** för varje episod antalet av de tre följande kommandona som siktar dit, med ingrepp minus utan ingrepp. Episoder som tar slut tidigare räknas med de kommandon som hann ges.
- **Godkänt:** medelskillnaden går åt förutsagt håll med mer än 2 standardfel, för både (a) och (b), i alla tre frön av 1a. Dessutom måste 1a:s effekt vara större än Blandats med mer än 2 standardfel. Annars kan effekten inte tillskrivas att schemat läses.
- **Redovisas utan att vara kriterium:** ett falskt "fönster förra steget", och hur många episoder som ingår i (a) och (b).

Om F inte klaras tolkas inget av H1–H3. Om H2 inte klaras men H3 gör det, styr schemat som en minneshjälp men inte som en modell som kan vilseledas. Det vore ett viktigt negativt resultat för AST-tolkningen.

Hur grinden läses: F måste klaras. Därefter bedöms H1, H2 och H3 var för sig. Inget samlat godkännande krävs, och alla utfall rapporteras.

### Oberoende granskning

**Oberoende granskning (22 okt 05:15): godkänd med anmärkningar.** Hela rapporten: v2/results/sprint3\_review.md.

- Bekräftat: reglaget i läget "visar allt" är bitidentiskt med sprint 2, ingen informationsläcka (inte heller i schemamodulen), lesionerna är implementerade som påstått, siffrorna stämmer med resultatfilerna, regeln för arbetspunkten tillämpades rätt och H5:s blindhet hölls.
- **H1-måttet var fel (åtgärdat ovan).** Mitt första förslag räknade steg i alla episoder, även missade. Missade episoder tar slut vid en stegbudget som växer med antalet distraktorer, så lutningen speglade mest träffandelen, det vill säga F en gång till. Räknat bara på hittade mål är 1a inte flackare än Blandat schema i frö 1. H1 ser alltså ut att inte få stöd. Mitt förslag: behåll H1 med det rättade måttet och rapportera utfallet, men räkna inte med det.
- **Testepisoderna är parade bara i första omgången om 500**, eftersom miljöns slumptal går vidare med antalet steg. Standardfelen är ändå giltiga. Men H2 kräver exakt samma episoder med och utan ingrepp, så MVP-koden behöver en bank med ett frö per episod. Åtgärdas i sprint 4 innan något körs.
- **Min förklaring till 1b var överdriven** (åtgärdat ovan).
- H2 måste preciseras före låsning: hur länge lögnen består, vad "siktar dit" betyder för ett 2×2-fönster, att det kan finnas färre än tre otäckta rutor med målets färg, och Blandat schema som kontroll. Jag skriver in det i sprint 4-planen.
- Mindre: ett test föll på toleransen i en gradientkontroll (inte ett fel, åtgärdat), och 1b-resultaten finns nu i v2/results/sprint3\_1b.json. H6-slutsatsen gäller den här arkitekturen, inte foveor i allmänhet.

### Grind 3: dina beslut

**Beslut för produktägaren vid grind 3** (skriv svar: under respektive punkt)

- [x] Väg för MVP och H5: A (rekommenderas), B eller C?
- [x] H1 med det rättade måttet, med vetskap om att det troligen inte får stöd?
- [x] Trösklarna enligt tabellen ovan, med H2 preciserat i sprint 4-planen innan något körs?

**Produktägarens villkor för grind 3 (22 okt kväll), och hur de är uppfyllda:**

- [x] H2 och fröbanken skrivs in i specen före låsning. Står under H2 i detalj och Fröbank ovan.
- [x] H3 får lesion även under träning eller formuleras om. Omformulerad till H3a, lesion under hela träningen (Baslinjen), och H3b, akut lesion vid test. Båda krävs.
- [x] "2 SD" rättas till standardfel. Rättat i Signifikanskriterium, grind 2 och pilotplanen. I CLAUDE.md rättas det också.

**Beslut 22 okt 2026 kl. 22:30 (produktägaren, i chatten: "Jag följer dina rekommendationer"): grind 3 godkänd.** Alternativ A gäller. H1 mäts med det rättade måttet. H3 prövas som H3a och H3b, utan slumpvis avstängning under träning.

**Trösklarna är låsta från och med detta beslut.** Det som är låst är tabellen "Förslag: trösklar och körplan för MVP", signifikanskriteriet, H2 i detalj, fröbanken och H3:s omformulering, allt i den lydelse de har nu. Det ändras inte efter att MVP:s resultat setts. Måste något ändras innan körning, till exempel för att H2 visar sig omöjligt att implementera som det står, skrivs ändringen och skälet här, och du godkänner den innan något körs.

## Sprint 4: MVP

### Plan (godkänd 22 okt 22:44)

MVP ska svara på projektets huvudfråga i fas 1: styr schemat agentens uppmärksamhet så att en lögn i schemat flyttar den (H2)? Det prövas mot de låsta kriterierna F, H1, H2 och H3 med nya frön, så att inget resultat från piloten räknas. Inget körs förrän du har godkänt planen.

**Steg**

1. Förberedelse utan träning, ca 2 timmars arbete. Två delar: fröbanken (ett eget frö per testepisod, med separata slumpströmmar för miljön och för agentens val) och ingreppen för H2. Till det kommer testskript för F, H1, H2 och H3 enligt de låsta definitionerna. Testerna visar att samma episod ger identiskt förlopp, att förloppen med och utan ingrepp är lika fram till steg 3, och att ingreppen ändrar exakt de rutor som specen anger och består resten av episoden. H2-koden prövas bara med enhetstester och otränade agenter, aldrig på pilotens agenter, så att inget H2-resultat syns före MVP. Allt pushas innan något tränas.
2. Träning, ca 2 timmar i bakgrunden. Baslinje, 1a och Blandat med frö 10, 11 och 12, nio körningar à 1 000 000 episoder, två åt gången.
3. Bedömning, ca 30 minuter. Alla kriterier räknas på fröbanken och förs in här med samtliga siffror, även de som inte klarar kriterierna.
4. Oberoende granskning av kod och resultat mot de låsta kriterierna. Grind 4: du bedömer resultatet.

**Om svaret blir nej**

| Utfall | Vad det betyder | Vad som händer |
| --- | --- | --- |
| En körning lär sig inte söka (under 73 % träffandel, sprint 2:s K2) | Träningen misslyckades, inte hypotesen | Körningen görs om en gång med frö + 1000. Det redovisas. Regeln gäller alla villkor lika och bestäms nu, före körning |
| F klaras inte | Schemat hjälper inte i nya frön, trots piloten | Inget annat tolkas. Jag rapporterar och föreslår orsaker. Du beslutar om fortsatt väg |
| H2 klaras inte, men H3 gör det | Schemat fungerar som minneshjälp men kan inte vilseledas | Ett viktigt negativt resultat för AST-tolkningen. Rapporteras som det är. Ingen ny körning för att rädda H2 |
| H1 eller H3 klaras inte | Delresultat | Rapporteras. Påverkar inte bedömningen av H2 |

**Tekniska val (Claudes, redovisas men kräver inte beslut)**

- Samma miljö och agent som i piloten: standardinställningarna, ingen ändring i arkitekturen.
- 1 000 000 episoder, eftersom baslinjen fortfarande förbättras efter 400 000. Det gör jämförelsen försiktig, till baslinjens fördel.
- Bästa checkpoint väljs på valideringsepisoder som i sprint 2. Fröbanken används bara för test.
- Fröbanken: episod i har frö 2 000 000 + i, i = 0 till 3 999. Samma bank används för alla villkor och frön.
- H1 mäts på en egen bank om 2 000 episoder per antal distraktorer (4, 8, 12, 16) med 8×8, fönster 1 och fångst påslagen.

**Beslut för produktägaren** (skriv svar: under punkten, eller svara i chatten)

- [x] Godkänner du sprint 4-planen?
- [ ] svar: Y

**Beslut 22 okt 2026 kl. 22:44 (produktägaren, i specen och i chatten):** sprint 4-planen är godkänd. Följ den steg för steg. Inga låsta kriterier ändras. Om H2 eller fröbanken visar sig omöjliga att genomföra som de står skrivs ändringen här och väntar på godkännande. H2-koden får inte prövas på pilotens agenter.

### Steg 1: fröbank, H2-ingrepp och testskript (klart 22 okt 22:55)

Fröbanken och H2 gick att genomföra som de står. Ingen ändring av låsta kriterier behövdes. Allt pushades innan träningen startade (commit 4b550d4).

- **Fröbank** (`v2/bank.py`): testepisod i har frö 2 000 000 + i, i = 0 till 3 999. Ur fröet skapas separata slumpströmmar för miljön och för agentens val. Varje episod i en batch drar ur sina egna strömmar, så en episod förlöper likadant oavsett vilka andra episoder som körs samtidigt.
- **H2-ingreppen** (`v2/bank.py`): (a) falskt obesökt, (b) falskt besökt och det redovisade (c) falskt "fönster förra steget", enligt H2 i detalj. Måttet räknas per episod, med ingrepp minus utan ingrepp, parat på samma bankepisod.
- **Bedömningsskript** (`v2/eval_mvp.py`): F, H1, H2, H3a och H3b enligt de låsta definitionerna, samt regeln om omkörning under 73 %.
- **Tester** (`v2/test_bank.py`, 13 tester, alla godkända): samma episod ger identiskt förlopp; förloppen med och utan ingrepp är lika till och med steg 3, med samma utvalda rutor; ingreppet ändrar exakt de rutor specen anger och inga andra; lögnen består resten av episoden och ny täckning registreras; episoder som tagit slut efter tre steg räknas inte; schemat byggs exakt som i träningen. Miljön är bitidentisk med tidigare version, och de gamla testerna (miljö, agent, H5) går igenom.
- H2-koden är bara prövad med enhetstester på otränade agenter. Den har inte körts på pilotens agenter.

**Tekniska val (Claudes, gjorda före träning, kräver inte beslut):**

- Valet av rutor i H2 (b) och läget för det falska fönstret i (c) dras ur en tredje ström från samma episodfrö. Annars skulle valet flytta miljöns eller agentens slumptal, och förloppen med och utan ingrepp skulle skilja sig efter steg 3 av andra skäl än lögnen.
- Blandat schema som kontroll i H2: samma lögn läggs i det schema agenten får, med rutorna valda ur agentens egen episod som för 1a. Givaren av schemat är nästa episod i samma block om 500, så ingen episod får sitt eget schema.
- Standardfel för skillnader mellan villkor räknas oparat, roten ur summan av de två kvadrerade standardfelen, som i piloten och vid grind 2. Det är det försiktigare valet. H2 räknas parat, som specen anger. Standardfel för andelen återbesök och för H1-lutningen tas fram med bootstrap över episoder (1 000 omdragningar).
- H1-banken: 2 000 episoder per antal distraktorer (4, 8, 12, 16), frö 3 000 000 + 10 000 × k + i. 8×8, fönster 1, fångst påslagen, stegbudget avrundat uppåt 1,2 × (k + 1) som i piloten.
- Träning: entropin trappas ned över 700 000 episoder, som i sprint 2:s körningar på 1 000 000 episoder.
- Under 73 % räknas på fröbanken med agentens stokastiska policy.

### Steg 2–4: träning, bedömning och granskning

**Resultat (23 okt 01:00): F, H2 och H3 klaras i alla tre frön med stor marginal. H1 klaras inte.** Alla nio körningar nådde 1 000 000 episoder. Ingen låg under 73 %, så ingen gjordes om. Siffrorna finns i `v2/results/sprint4_mvp.json`.

Träningen gick 22:54–00:51, 22–24 minuter per körning, två åt gången. Bästa checkpoint valdes på valideringsepisoder och hamnade mellan iteration 1 200 och 3 800 (av ca 3 900). Därefter testades varje körning på fröbankens 4 000 episoder.

**Bedömning mot de låsta kriterierna** (skillnad, i parentes antal standardfel)

| Kriterium | Frö 10 | Frö 11 | Frö 12 | Utfall |
| --- | --- | --- | --- | --- |
| F: träff, 1a − Baslinje (procentenheter) | +7,5 (12,0) | +6,2 (9,9) | +6,9 (11,3) | Klarat |
| F: träff, 1a − Blandat | +6,4 (10,6) | +6,0 (9,6) | +6,6 (10,9) | Klarat |
| H1: söklutning, 1a − Baslinje (steg per distraktor) | −0,021 (1,6) | −0,001 (0,1) | −0,003 (0,2) | Ej klarat |
| H1: söklutning, 1a − Blandat | −0,010 (0,8) | +0,006 (0,4) | +0,001 (0,0) | Ej klarat |
| H2 (a): 1a, kommandon mot falskt obesökta rutor, med − utan lögn | +0,22 (19,4) | +0,15 (16,7) | +0,07 (10,5) | Klarat |
| H2 (a): 1a:s effekt − Blandats | +0,22 (19,1) | +0,14 (16,0) | +0,07 (10,7) | Klarat |
| H2 (b): 1a, kommandon mot falskt besökta rutor, med − utan lögn | −0,81 (35,2) | −0,83 (38,3) | −0,84 (38,6) | Klarat |
| H2 (b): 1a:s effekt − Blandats | −0,74 (30,7) | −0,62 (25,9) | −0,91 (39,9) | Klarat |
| H3a: träff, Baslinje − 1a (procentenheter) | −7,5 (12,0) | −6,2 (9,9) | −6,9 (11,3) | Klarat |
| H3a: återbesök, Baslinje − 1a (procentenheter) | +5,7 (10,2) | +5,4 (9,6) | +5,5 (9,5) | Klarat |
| H3b: träff, akut lesion − intakt 1a | −50,2 (58,6) | −34,6 (40,4) | −46,7 (54,2) | Klarat |
| H3b: återbesök, akut lesion − intakt 1a | +56,4 (118) | +41,8 (80,0) | +52,6 (104) | Klarat |

Enligt grindregeln är F klarat, så H1, H2 och H3 bedöms var för sig. H3 kräver både H3a och H3b, och båda klaras.

**Träffandel och återbesök på fröbanken** (andel av alla steg där spotlighten bara täckte redan täckta rutor)

| Villkor | Träff frö 10 / 11 / 12 | Återbesök frö 10 / 11 / 12 |
| --- | --- | --- |
| Baslinje | 87,7 / 88,2 / 88,2 % | 16,2 / 16,2 / 16,0 % |
| Fast schema 1a | 95,1 / 94,4 / 95,1 % | 10,5 / 10,8 / 10,5 % |
| Blandat schema | 88,7 / 88,4 / 88,5 % | 16,6 / 15,9 / 15,8 % |
| 1a, schemat nollat vid test | 44,9 / 59,8 / 48,5 % | 66,9 / 52,6 / 63,2 % |
| Blandat, schemat nollat vid test | 88,7 / 88,5 / 88,4 % | 16,5 / 15,7 / 15,7 % |

Den akut lesionerade 1a-agenten är 28–43 procentenheter sämre än Baslinjen. Det redovisas men är inget kriterium: agenten har byggt sin sökstrategi kring schemat. Blandat-agenten påverkas inte alls av lesionen, den har lärt sig att inte använda kanalen.

**H2 i detalj:** antal av de tre följande kommandona som siktar mot de manipulerade rutorna, medel per episod

| Ingrepp | Villkor | Episoder frö 10 / 11 / 12 | Utan lögn | Med lögn |
| --- | --- | --- | --- | --- |
| (a) falskt obesökt | 1a | 1 775 / 1 887 / 1 834 | 0,17 / 0,19 / 0,20 | 0,39 / 0,34 / 0,27 |
| (a) falskt obesökt | Blandat | 1 933 / 1 935 / 1 934 | 0,33 / 0,32 / 0,32 | 0,33 / 0,32 / 0,31 |
| (b) falskt besökt | 1a | 1 735 / 1 848 / 1 797 | 0,95 / 0,97 / 1,00 | 0,14 / 0,15 / 0,17 |
| (b) falskt besökt | Blandat | 1 886 / 1 894 / 1 887 | 1,04 / 1,07 / 1,07 | 0,97 / 0,86 / 1,14 |
| (c) falskt fönster, redovisas | 1a | 1 775 / 1 887 / 1 834 | 0,20 / 0,20 / 0,19 | 0,12 / 0,13 / 0,11 |
| (c) falskt fönster, redovisas | Blandat | 1 933 / 1 935 / 1 934 | 0,20 / 0,20 / 0,20 | 0,20 / 0,22 / 0,19 |

Episoderna är de som pågick efter tre steg, ungefär hälften av banken. I (b) faller ca 40 per frö bort eftersom det inte fanns någon otäckt ruta i målets färg. Med lögnen (a) sjunker 1a:s träffandel till 92,5–93,3 %. Med lögnen (b) stiger den till 98,2–98,3 %, eftersom de falskt markerade rutorna enligt specen aldrig är målet.

**Min tolkning (Claudes, inte ett beslut):**

- **H2 får tydligt stöd.** En lögn i schemat flyttar agentens verkliga spotlight åt det håll AST förutsäger. Falskt besökta rutor undviks nästan helt: 0,15 kommandon mot ca 0,97 utan lögn. Agenten litar alltså på schemat framför sitt eget minne av var den siktade, trots att det minnet finns i efferenskopian och i dess tillstånd.
- **Effekten är asymmetrisk.** Falskt obesökta rutor drar bara måttligt (+0,07 till +0,22), och spridningen mellan frön är stor. En trolig förklaring är att agenten också minns var den siktat och därför sällan går tillbaka, oavsett vad schemat säger. Schemat avgör mest vad som ska hoppas över.
- **Blandat-kontrollen fungerar, men inte perfekt.** Blandat-agenten reagerar knappt på (a) och (c), högst 0,015 kommandon. På (b) reagerar den svagt och åt olika håll i olika frön (−0,07, −0,21, +0,07). Ett mönster av ettor just på rutor i målets färg förekommer aldrig i dess träning, så det kan ändra beteendet utan att agenten läser kanalen som en beskrivning av sig själv. 1a:s effekt är i varje frö minst fyra gånger större.
- **(c) är ett oväntat fynd.** 1a siktar mer sällan mot ett falskt "fönster förra steget". I piloten gjorde det ingen skillnad att nolla den kanalen, så agenten läser den men behöver den inte. Det är inget kriterium.
- **H1 får inte stöd,** som piloten förutsade. Bland hittade mål söker 1a inte flackare än Baslinjen eller Blandat. Schemats vinst ligger i färre missar och återbesök, inte i en flackare söklutning.
- Ett godkänt H2 visar att mekanismen beter sig som AST förutsäger. Det säger inget om upplevelse.

**Skärpt tolkning efter grind 4 (produktägaren och Claude, 23 okt).** Den här tolkningen gäller framför punkterna ovan:

- **MVP har prövat ett orakel, inte AST:s kärna.** I 1a räknas schemat ut ur den verkliga spotlighten och har aldrig fel under träningen. Därför är det rationellt för agenten att lita på det. H2 visar att agenten behandlar schemat som auktoritet. Det visar inte att schemat är en förenklad modell. AST:s kärna, att kunskapen om den egna uppmärksamheten kommer från en förenklad modell, prövas först i H5 och 1c.
- **H2 (b) bär resultatet.** Agenten litar mer på schemat än på sitt eget minne av vart den siktade.
- **H3 tillför lite eget.** Träffandelen i H3a är samma jämförelse som F, med samma siffror. Återbesöken är delvis givna per konstruktion, eftersom återbesök är vad kartan registrerar. H3b bygger på en akut lesion utanför träningsfördelningen.
- **Asymmetrin i H2 är oförklarad.** Ett falskt "redan tittat här" följs nästan alltid, ett falskt "inte tittat här" bara delvis. Asymmetrin visar vilken information agenten behandlar som auktoritet, och det är vad H5 handlar om. Min förklaring ovan, att agenten minns var den siktat, är en av flera möjliga. De prövas i sprint 5a.

### Oberoende granskning

**Oberoende granskning (23 okt 01:20): godkänd med anmärkningar. Ingen anmärkning ändrar utfallet för något kriterium.** Hela rapporten: `v2/results/sprint4_review.md`.

- **Bekräftat:** fröbanken och H2 är implementerade som de låsta definitionerna anger, utan informationsläckor. Granskaren skrev en egen version av bankepisoden och H2-måttet och fick exakt samma siffror. Förloppen med och utan lögn var identiska till och med steg 3 även hos tränade agenter. Bedömningen tillämpar rätt riktning och kräver alla tre frön. Alla nio körningar nådde 1 000 192 episoder med de redovisade inställningarna, ingen startades om eller kasserades. Ingen H2-kod har körts på pilotens agenter.
- **Stor anmärkning, tolkning: Blandat-kontrollen för H2 (a) var nästan tom.** Rutorna i (a) är agentens egna täckta rutor. I det schema Blandat-agenten får, från en annan episod, är de sällan markerade, så lögnen ändrade i genomsnitt 0,76 rutor mot 6,1 för 1a. Det är en följd av mitt tekniska val. Jämförelsen med Blandat i (a) var därför nästan automatisk. Kontrollen i (b) är rättvis, 2,6 mot 2,8 ändrade rutor.
- **Liten: (a) beror på vilken checkpoint som används.** På slutcheckpointerna är effekten i (a) ungefär dubbelt så stor i frö 11 och 12 (+0,30 och +0,19). (b) ändras knappt.
- **Liten: en del av effekten i (b) är mekanisk.** Lögnen hjälper agenten att hitta målet, så episoderna blir kortare och färre kommandon räknas. Räknat per kommando är effekten fortfarande stor: −0,29 ± 0,008.
- **Liten, process:** H2-siffrorna för åtta körningar fanns innan den nionde var klar. Inget ändrades efteråt.
- **Liten, kod:** `eval_mvp.py` återanvänder sparade resultat om en checkpoint skulle bytas ut. Bör få en kontroll.
- **Tolkningsnoter:** inget utfall ändras med parat i stället för oparat standardfel. Halvan om återbesök i H3 gäller delvis per konstruktion, eftersom återbesök är just vad kartan registrerar. Att Blandat reagerar på (b) utan att använda kanalen visar att en reaktion på kanalen inte i sig bevisar en modell.

**Tilläggsanalys efter granskningen (inte ett kriterium, de låsta utfallen gäller oförändrade).** H2 (a) för Blandat gjordes om med de rutor som schemat agenten faktiskt får visar som täckta efter steg 3. Lögnen tar då bort verkliga markeringar ur dess indata. För 1a är det identiskt med den låsta analysen. Skript: `v2/supp_h2a.py`, resultat: `v2/results/sprint4_supp_h2a.json`.

| Frö | Blandat, med − utan lögn | 1a:s effekt − Blandats |
| --- | --- | --- |
| 10 | +0,018 (4,6 SE) | +0,20 (16,8 SE) |
| 11 | +0,045 (9,2 SE) | +0,10 (10,1 SE) |
| 12 | −0,019 (5,4 SE) | +0,09 (11,9 SE) |

Med en verklig lögn reagerar Blandat svagt och åt olika håll. 1a:s effekt är fortfarande större med mer än 10 standardfel i varje frö. Slutsatsen om H2 (a) står sig alltså även mot den skarpare kontrollen. (a) bör ändå beskrivas som en delvis effekt: agenten följer nästan alltid ett falskt "redan tittat här", men bara delvis ett falskt "inte tittat här".

### Grind 4: dina beslut

**Sammanfattning:** F, H2, H3a och H3b är klarade och H1 är inte klarat, enligt de låsta kriterierna. Granskningen godkänner med anmärkningar som gäller tolkning, inte utfall.

**Beslut för produktägaren vid grind 4** (skriv svar: under respektive punkt)

- [x] Godkänner du resultatet av sprint 4 enligt de låsta kriterierna?
- [x] svar (i chatten 23 okt 10:48, infört av Claude): ja, grind 4 godkänd.
- [x] Nästa steg enligt sprintplanen är sprint 5, H5: schemamodul med lokal matchningsinformation och en egen pilot innan H5:s tröskel låses. Jag skriver planen i specen och väntar på ditt godkännande innan något körs. Går det bra?
- [x] svar (i chatten 23 okt 10:48, infört av Claude): ja till att planen skrivs, med villkor. Se nedan.

**Tekniska val (Claudes) som rör grinden:** jag föreslår ingen omkörning. De låsta utfallen gäller med de checkpoints som valdes på validering, även om slutcheckpointerna ger större effekt i (a). Tilläggsanalysen ovan redovisas bredvid, inte i stället.

**Beslut 23 okt 2026 kl. 10:48 (produktägaren, i chatten): grind 4 godkänd.** Skäl: proceduren höll hela vägen. Trösklarna var låsta före körning, frön och fröbank var nya, ingen körning gjordes om, och granskaren fick exakt samma H2-siffror med egen kod. F och H2 klaras med mycket stor marginal i alla tre frön.

**Produktägarens bedömning av resultatet:**

- **H2 (b) är det starkaste fyndet.** Agenten litar mer på schemat än på sitt eget minne av vart den siktade. Det är ett litet men verkligt steg mot kärnfrågan. Begränsning: i 1a har schemat aldrig haft fel under träningen, så det är rationellt att lita på det. Visat är att agenten behandlar schemat som auktoritet. Inte visat är att schemat är en förenklad modell. Det är H5:s och 1c:s uppgift.
- **H3 är svagare än det ser ut.** Halva H3a, träffandelen, är samma jämförelse som F. Återbesöken är delvis givna per konstruktion. H3b bygger på en akut lesion utanför träningsfördelningen. H3 är formellt klarat, men det är H2 som bär resultatet.
- **Asymmetrin i H2 är det intressantaste fyndet och ännu inte förklarad.** Förklaringen i specen är en av flera möjliga. En billig diagnostik före H5-piloten är väl använd tid.
- **Lärdom om kontroller.** Blandat-kontrollen för (a) var nästan tom på grund av ett tekniskt val, och varken granskningen vid grind 3 eller testerna fångade det. Framöver ska testerna före låsning kontrollera att ett ingrepp ändrar ungefär lika mycket i kontrollen som i huvudvillkoret.

**Villkor för sprint 5 (produktägaren):** planen ska uttryckligen hantera risken att H5 blir falsk per konstruktion. En schemamodul som får en lokal matchningssignal och tränas mot den faktiska platsen lär sig troligen att peka på den faktiska platsen. Då säger ett nej om H5 ingenting om AST. Planen ska jämföra med en variant där schemat har en flaskhals och måste förenkla, och den ska ta in asymmetrin från H2 som underlag.

**Claudes kommentar:** jag delar bedömningen. H3a:s träffhalva har exakt samma siffror som F mot Baslinjen i tabellen ovan, så den tillför inget eget. Invändningen om att schemat aldrig haft fel gäller också asymmetrin. Även (a) har en motsvarighet i träningen: "siktade dit men täckte det inte" är precis vad en fångst ger. Skillnaden kan alltså ligga i vilka lögner som liknar verkliga fångster. Det prövas som förklaring E3 i sprint 5-planen nedan.

## Sprint 5: H5

### Plan 5a, reviderad 23 okt (godkänd med villkor 23 okt 12:04)

Sprint 5 ska avgöra om ett schema som måste förenkla anger den avsedda platsen när spotlighten fångas, och inte bara för att det saknar information. Den har två delar. 5a: diagnostik av H2-asymmetrin, bygge och pilot. Den slutar med att H5:s trösklar låses vid grind 5a. 5b: huvudkörning med nya frön, bedömd mot de låsta trösklarna. Planen nedan gäller 5a. Plan 5b skrivs in när 5a är klar. Inget körs förrän du har godkänt planen.

**Revision 23 okt efter nio kommentarer från diskussionstråden** (förslag, inte beslut; svar i varje tråd):

- **Flaskhalsen flyttas** från modulens indata till dess inre representation, det modulen lämnar till agenten. Modulen får full indata.
- **Nivåerna anges som andelar**, 50, 25 och 12,5 %, av den informationsmängd varianten utan kostnad använder, räknat per steg och inte per ruta.
- **H5 får en nollmodell som låses vid grind 5a.** Den föreslagna formen behövde en justering, annars hade provet klarats automatiskt. Se punkt 3 nedan.
- **Diagnostiken:** E1 och E2 slås ihop. E3:s prov delas efter fångst. E4, att asymmetrin sitter i måttet, läggs till. Alla jämförelser görs bara på rutor i målets färg.
- **Beteendemåttet byts** mot en jämförelse inom samma agent: fångster där schemat hade fel mot fångster där det hade rätt.
- **H2 (b) körs på 1b-agenterna i 5b**, med lögnen i modulens karta. Det svarar på invändningen om oraklet. Ingen 1a-variant med brus.

**Varför H5 kan avgöras i förväg (ditt villkor)**

| Schemamodulens information | Vad den troligen lär sig | Vad ett utfall säger om AST |
| --- | --- | --- |
| Ingen matchningssignal, som i sprint 3 | Den avsedda platsen, eftersom fångsten inte syns | Inget, H5 är sann per konstruktion |
| Full indata, ingen kostnad | Den faktiska platsen, eftersom den tränas mot den | Inget, H5 är falsk per konstruktion. Varianten är förutsättningskontroll |
| Full indata, kostnad på den inre representationen | En grov bild av platsen. Eftersom de flesta steg saknar fångst är "där jag siktade" billigast, så även ren komprimering väntas dra mot den avsedda platsen | Det informativa fallet. Samlas felen på den avsedda platsen mer än nollmodellen säger, visar det en följd av AST, men inte något unikt för AST |

Så hanterar planen det:

1. **Varianten utan kostnad är en förutsättningskontroll, inte H5-provet.** Den ska visa att informationen räcker för att se fångsten. Klarar den inte det är H5 omätbar i den här formen, och vi stannar.
2. **H5 prövas på tre nivåer med kostnad, som bestäms nu.** Nivåerna är 50, 25 och 12,5 % av den informationsmängd per steg som varianten utan kostnad faktiskt använder. Ingen nivå väljs efter resultat.
3. **Förutsägelsen prövas mot en nollmodell som låses vid grind 5a.** Nollmodellen är en försvagad ideal betraktare. Den får samma träffsäkerhet vid fångst som schemat, genom att varje signal den använder med en viss sannolikhet byts mot ett slumpvärde (villkor a). Då hamnar en del av dess fel enligt förhandssannolikheterna, som redan gynnar den avsedda platsen. Signaturen är att schemats fel ligger på den avsedda platsen oftare än nollmodellens. En modul som bara blir allmänt sämre klarar inte provet. En som faller tillbaka på styrsignalen gör det.
4. **Ett beteendemått kopplar H5 till H2 (b).** Inom samma agent jämförs fångster där schemat hade fel med fångster där det hade rätt. Går agenten mer sällan tillbaka till den avsedda platsen när schemat felaktigt säger att den redan är sedd? Då har det förenklade schemat en kostnad i beteendet.

Hur ett ja på H5 ska läsas (villkor c, produktägarens beslut 23 okt): Ren komprimering väntas också dra mot den avsedda platsen. Avkodaren ser inte efferenskopian, så båda platserna kostar bitar. Men de flesta steg sker utan fångst, och då är avsedd plats densamma som faktisk plats. Den billigaste beskrivningen är därför "där jag siktade", och det första en komprimering släpper är fångstinformationen, som bara gäller en minoritet av stegen. Modulen får dessutom ingen gradient från styrningen, så dess fel speglar bara komprimeringens ekonomi. Ett ja på H5 visar att en kapacitetsbegränsad modell av uppmärksamheten behåller avsikten och släpper undantagen. Det följer av AST men är inte unikt för AST. Det AST-specifika, att schemat formas av sin användning för styrning, prövas först i 1c. Jag skriver det rakt ut, så att ett ja på H5 inte läses som mer än det är. Var bytet sker i dosserien redovisas, men det är nollmodellen som avgör.

**Steg**

1. **Diagnostik av asymmetrin i H2, ca 1 timme, ingen träning.** Körs på sprint 4:s 1a- och Blandat-agenter. Den är beskrivande och ändrar inget låst utfall. Alla jämförelser görs bara på rutor i målets färg, eftersom tomma rutor och rutor i annan färg aldrig är värda ett besök. Rutor med ett iögonfallande föremål utesluts också (teknisk rättelse 23 okt, före hela körningen). Ett sådant föremål är aldrig målet, periferin visar dess salience, och agenten siktar aldrig dit. En kontrollkörning på 500 episoder visade att rutor som täckts bara genom fångst nästan alltid var just det iögonfallande föremålet. E1-provet hade då visat noll effekt av det skälet ensamt. Tre förklaringar prövas, var och en med ett prov och en förutsägelse som bestäms nu:
   - **E1, agentens eget minne av sikte och synintryck.** Agenten minns vart den siktat och vad den såg där, och litar på det framför schemat. Prov: (a)-rutorna delas i dem agenten siktade mot och dem som täcktes bara genom fångst. E1 förutsäger större effekt för fångstrutorna, eftersom agenten saknar eget minne av dem.
   - **E2, rutornas värde, slås ihop med E1.** Den första förutsägelsen, effekt nästan bara i målets färg, följer av alla förklaringar och skiljer inte ut E2. Inom målets färg sammanfaller E2 med E1: agenten vet vad en ruta innehöll bara genom remappningen, som lägger synintrycket på den ruta den siktade mot. Jag hittar ingen förutsägelse där de skiljer sig, så de prövas tillsammans.
   - **E3, rimlighet.** (b) liknar en verklig fångst. I (a) försvinner tre stegs täckning på en gång, något som nästan aldrig händer i träningen. Prov: en variant av (a) där bara steg 3:s fönster blir falskt obesökt, som efter en enda fångst. Effekten per ruta jämförs med samma rutor från steg 3 när hela (a) ljuger, delat efter om steg 3 var en fångst. Inom varje grupp skiljer sig då bara hur mycket av kartan som ljuger, inte minnet. E3 förutsäger större effekt per ruta när bara steg 3 ljuger, i båda grupperna.
   - **E4, asymmetrin sitter i måttet.** (b) mäter undvikande från en hög nivå, eftersom agenten var på väg till rutorna. (a) mäter dragning till rutor som konkurrerar med alla verkligt obesökta rutor i målets färg. Prov: (a) delas efter antalet kvarvarande obesökta rutor i målets färg. Dessutom räknas båda ingreppen per ruta, i förhållande till en verkligt obesökt ruta i målets färg i samma episod. E4 förutsäger att (a)-effekten krymper med antalet konkurrenter, och att asymmetrin minskar med det normerade måttet.
   - Dessutom, efter granskningen: (a) och (b) på slutcheckpointerna, och (b) räknat per kommando och bara i episoder där alla tre kommandona hann ges.

   Förklaringarna utesluter inte varandra. De redovisas med standardfel, utan godkänt eller underkänt. Resultatet används när beteendemåttet låses vid grind 5a. Stöder det E1 måste beteendemåttet skilja på rutor agenten siktat mot och rutor den bara fångats till. Stöder det E4 måste måttet normeras mot konkurrerande rutor.
2. **Bygge, ca 5 timmar.** Allt pushas innan något tränas.
   - **Matchningssignal.** För varje ruta i det avsedda fönstret anger den om det foveán visade stämmer med vad periferin visar där, upptaget och färg. Den går bara till schemamodulen, inte till styrningen. Biologiskt motsvarar det en jämförelse mellan förväntad och faktisk syninformation.
   - **Flaskhals** (rekommenderad form och plats, se beslut nedan): kostnad per informationsmängd på modulens inre representation, nära 1c. Modulen får full indata: efferenskopia, matchning och periferi. Det den lämnar till agenten passerar en brusig kanal, där det kostar i träningen att släppa igenom information. Schemat som agenten får räknas bara ur det som passerat kanalen, så efferenskopian kan inte gå runt flaskhalsen. AST förutsäger förenkling just för att schemat är resursbegränsat, och enligt AST är det modellen som är karikatyren, inte sinnena. Samma kanal kan återanvändas i 1c.
   - **Nivåer som andelar av det som behövs.** Informationsmängden räknas per steg, summerad över alla rutor. Varianten utan kostnad ger referensen: den informationsmängd den faktiskt använder. De tre nivåerna är 50, 25 och 12,5 % av referensen och låses nu. Under träningen regleras kostnadens vikt så att den uppmätta informationsmängden håller nivån. Regleringen tittar bara på informationsmängden, inte på något H5-mått.
   - **Ändring av H5:s mätvillkor (ingår i godkännandet av planen).** "Foveán utesluter avsedd plats" avgörs med upptaget och färg, det agenten kan jämföra mot periferin. Formen ingår inte, eftersom periferin aldrig visar form. Den ideala betraktaren får samma begränsning. Annars ser den mer än agenten kan.
   - **Doskontroll enligt din lärdom.** Ett test visar att matchningssignalen skiljer sig vid fångst i samma andel fall som den ideala betraktaren kan upptäcka fångsten, och att alla nivåer får exakt samma indata. Piloten räknar dessutom hur många fångster som hamnar i varje grupp för beteendemåttet, så att båda grupperna är tillräckligt stora innan måttet låses.
   - **Nollmodellen och beteendemåttet** byggs och testas på otränade agenter, liksom H2 (b) med lögnen i modulens karta, som används först i 5b.
3. **Pilot, ca 2,5 timmar träning i bakgrunden.** Fem villkor: Baslinje, 1b utan kostnad och 1b med 50, 25 och 12,5 % av referensens informationsmängd. Varianten utan kostnad tränas först, eftersom den ger referensen. Frö 20 och 21, 400 000 episoder, tio körningar. Testet görs på en egen pilotbank, inte på MVP:s fröbank. Piloten mäter följande, blint för H5:s utfall:
   - **Förutsättningen:** modulen utan kostnad lägger sin mest sannolika ruta i det faktiska fönstret vid minst hälften av de H5-händelser där den ideala betraktaren kan se att spotlighten fångats.
   - **Att flaskhalsen biter:** andelen rätt vid fångst sjunker när informationsmängden minskar. Det visar hur ofta schemat har fel, inte vart det pekar.
   - **Schemats nytta:** träffandel för varje nivå, jämförd med Baslinjen.
   - Uppmätt informationsmängd per nivå, antal H5-händelser, antal fångster per grupp för beteendemåttet, spridning mellan frön och tid per körning.

   Vart felen pekar, och beteendemåttet, tittar jag inte på förrän trösklarna är låsta.
4. **Förslag till trösklar och körplan för 5b, oberoende granskning, grind 5a.** Nollmodellen, beteendemåttet och deras trösklar låses då. Du godkänner, och trösklarna låses.

**Om svaret blir nej**

| Utfall | Vad det betyder | Vad som händer |
| --- | --- | --- |
| Förutsättningen klaras inte | Matchningssignalen räcker inte för att se fångsten | H5 är omätbar i den här formen. Jag föreslår en annan signal, du beslutar |
| Flaskhalsen biter inte på någon nivå | Uppgiften är för lätt att komprimera | Rapporteras. Jag föreslår smalare nivåer, du beslutar innan något körs |
| Inget schema hjälper agenten | Modulen ger ingen användbar självkunskap, som i sprint 3 | Rapporteras. H5 kan ändå mätas, men tolkningen blir svagare. Beteendemåttet och H2 (b) på 1b blir tomma, eftersom agenten inte läser schemat |
| Diagnostiken stöder ingen av E1, E3 och E4 | Asymmetrin är oförklarad | Rapporteras. Beteendemåttet räknas både för siktade och fångade rutor |

**Beslut för produktägaren** (skriv svar: under respektive punkt)

- [x] Godkänner du den reviderade plan 5a: diagnostik, bygge, pilot och förslag till trösklar, inklusive ändringen av H5:s mätvillkor?
- [x] svar (i chatten 23 okt 12:04, infört av Claude): godkänd, inklusive ändringen av H5:s mätvillkor, med villkor a–c nedan.
- [x] Flaskhalsens form och plats. Jag rekommenderar kostnad per informationsmängd på modulens inre representation, med full indata.
- [x] svar (i chatten 23 okt 12:04): kostnad per informationsmängd på modulens inre representation, med full indata.
- [x] Ska beteendemåttet bli ett kriterium bredvid H5, med en tröskel som låses vid grind 5a?
- [x] svar (i chatten 23 okt 12:04): ja, kriterium med tröskel som låses vid grind 5a.
- [x] H2 (b) på 1b-agenterna, med lögnen i modulens karta: redovisas eller kriterium?
- [x] svar (i chatten 23 okt 12:04): byggs i 5a. Produktägaren beslutar vid grind 5a om det redovisas eller blir kriterium.
- [x] Nollmodellen för H5: en försvagad ideal betraktare med samma träffsäkerhet som schemat?
- [x] svar (i chatten 23 okt 12:04): ja, med villkor a nedan.

**Beslut 23 okt 2026 kl. 12:04 (produktägaren, i chatten): plan 5a godkänd med villkor.** Diagnostiken i steg 1 får starta direkt. Bygget startar när villkor a–c står i specen. Villkoren kommer från tre nya kommentarer från diskussionstråden.

**Villkor a–c (produktägaren), införda före bygget, och hur de genomförs:**

- **a. Nollmodellen försvagas genom blandning, inte genom dämpning.** Varje signal som betraktaren använder, alltså foveáns innehåll jämfört med periferin och de iögonfallande föremålens salience, byts med sannolikheten q mot ett slumpvärde ur sin egen fördelning. Samma q gäller alla signaler. q väljs per frö och nivå så att betraktarens träffsäkerhet vid fångst blir densamma som schemats. Skäl: dämpas sannolikheterna med en exponent förblir en utesluten plats utesluten, eftersom noll upphöjt till något fortfarande är noll. Ett test på syntetiska fall, där foveán utesluter den avsedda platsen, visar före låsningen att nollmodellen lägger fel på den avsedda platsen när q är större än noll. En jämn försvagning är en svagare nollmodell än optimal komprimering, som släpper fångstbevisen först. Att slå den visar ojämn förenkling, inte något AST-specifikt.
- **b. Referensens fasta kostnad låses som tal.** Kostnaden är 0,001 per bit och steg, räknad i samma enhet som modulens förutsägelsefel. Referensen körs också med 0,0001, en faktor tio lägre. Skiljer sig den uppmätta informationsmängden mer än en faktor 1,5 mellan de två, är referensen inte väldefinierad. Då används i stället den minsta informationsmängd där förutsägelsens träffsäkerhet över alla steg ligger inom 1 procentenhet av varianten med 0,0001. Den hittas med körningar som styrs mot halverade informationsmängder. Allt tittar bara på träffsäkerheten över alla steg, inte på vart felen pekar vid fångst, så H5 förblir blint. Varianten med 0,0001 är förutsättningskontrollen.
- **c. Riskstycket är omskrivet.** Ren komprimering väntas också dra mot den avsedda platsen. Ett ja på H5 visar en följd av AST, inte något unikt för AST. Det AST-specifika prövas i 1c. Se stycket under "Så hanterar planen det".

**Tekniska val (Claudes, kräver inte beslut)**

- Schemamodulen är ett separat nätverk som i sprint 3 (`--schema module`), tränat mot förra stegets faktiska fönster. Ingen gradient går mellan modul och styrning.
- Kanalen: modulen lämnar ett tal per ruta, med brus vars styrka den lär sig. Informationsmängden räknas i bitar, som avståndet mellan den brusiga signalen och rent brus, summerat över rutorna. Avkodaren som gör schemakartan ser bara kanalens utdata. Kanalen per ruta, inte en gemensam vektor, eftersom resten av agenten är uppbyggd ruta för ruta.
- Referensen: helt utan kostnad kan bruset krympa mot noll och informationsmängden växa utan gräns. Referensen får därför en liten fast kostnad, låst enligt villkor b: 0,001 och 0,0001 per bit och steg. Referensvärdet är medelvärdet över de två pilotfröna. Det lägger fyra körningar till piloten, och ungefär en timme till om regeln med halverade informationsmängder behövs.
- Regleringen: kostnadens vikt justeras under träningen, uppåt när informationsmängden ligger över nivån och nedåt när den ligger under.
- Nollmodellen: blandning enligt villkor a. Med sannolikheten q ersätts foveáns innehåll med ett slumpvis valt kandidatfönsters innehåll, och varje salience med ett nytt värde ur samma fördelning som i miljön. Betraktarens svar är medelvärdet över många sådana dragningar. q väljs per frö och nivå så att träffsäkerheten vid fångst blir densamma som schemats. Bara träffsäkerheten används för valet, inte vart felen pekar.
- Schemats fel vid en H5-händelse räknas som den del av dess sannolikhet som ligger utanför det faktiska fönstret. H5 mäter hur stor del av den som ligger på det avsedda fönstret.
- Piloten använder frö 20 och 21 och 400 000 episoder, som sprint 3:s pilot. 5b får nya frön.
- Pilotbanken: episod i har frö 4 000 000 + i, 4 000 episoder, samma uppsättning strömmar som MVP:s fröbank. MVP:s fröbank används igen först i 5b.
- Arbetspunkt och miljö är oförändrade från sprint 4.
- Diagnostiken i steg 1 räknas parat som H2, med standardfel. Den redovisas utan godkänt eller underkänt.

### Steg 1: diagnostik av H2-asymmetrin – resultat

**Resultat (23 okt ca 12:40): E3 och E4 förklarar inte asymmetrin, och E1 gick inte att avgöra (rättat efter granskningen 23 okt). Asymmetrin finns kvar i det normerade måttet och sitter alltså i agenten, inte i måttet.** Ett falskt "redan tittat här" stänger 93–95 % av gapet till fullt förtroende, ett falskt "inte tittat här" bara 10–33 %. Diagnostiken återskapar de låsta H2-siffrorna exakt. Beskrivande, inget godkänt eller underkänt. Skript: `v2/diag_h2.py`, resultat: `v2/results/sprint5_diag_h2.json`.

Mått: antal av de tre följande kommandona som siktar mot rutan, med lögn minus utan, medel per ruta och episod. Bara rutor i målets färg, inte målet och inte iögonfallande föremål. Fast schema 1a, bästa checkpoint, MVP:s fröbank.

| Prov | Förutsägelse | Frö 10 | Frö 11 | Frö 12 | Utfall |
| --- | --- | --- | --- | --- | --- |
| E1: (a), rutor agenten siktade mot | – | +0,12 ± 0,01 | +0,08 ± 0,01 | +0,04 ± 0,01 | – |
| E1: (a), rutor täckta bara genom fångst | Större än siktade | −0,01 ± 0,01 | 0,00 | 0,00 | Golv vid noll, ej avgjort |
| E3: steg 3:s rutor, bara steg 3 ljuger | Större än hela (a) | +0,30 ± 0,02 | +0,22 ± 0,01 | +0,09 ± 0,01 | Lika, inget stöd |
| E3: steg 3:s rutor, hela (a) ljuger | – | +0,29 ± 0,02 | +0,21 ± 0,01 | +0,09 ± 0,01 | – |
| E4: (a), 0–2 / 3–5 / 6+ konkurrerande rutor | Krymper | +0,12 / +0,12 / +0,11 | +0,09 / +0,07 / +0,06 | +0,05 / +0,03 / +0,01 | Svagt i två frön |
| E4: andel av gapet som stängs, (a) | Asymmetrin minskar | 33 ± 2 % | 23 ± 2 % | 10 ± 2 % | Inget stöd |
| E4: andel av gapet som stängs, (b) | Asymmetrin minskar | 93 ± 2 % | 95 ± 2 % | 93 ± 2 % | Inget stöd |

E3 i gruppen där steg 3 var en fångst har bara 32–35 episoder och säger inget. Fullt förtroende betyder för (a) att de täckta rutorna får samma besöksfrekvens som verkligt obesökta, ca 0,37. För (b) betyder det samma frekvens som täckta rutor, ca 0,04.

**Övriga mått (efter granskningen):**

- (b) per kommando: −0,29 i alla tre frön. I episoder där alla tre kommandona hann ges: −0,95 till −0,97 av 3. Effekten i (b) är alltså inte mekanisk.
- Slutcheckpoints: (a) +0,20 / +0,30 / +0,19 mot +0,22 / +0,15 / +0,07 på bästa checkpoint. (b) oförändrad, −0,80 till −0,82. (a) beror på träningsläge, (b) gör det inte.
- Blandat schema: ingen effekt av (a). Svag effekt av (b) med växlande tecken, −8 till +24 % av gapet.

**Min tolkning (Claudes, inte ett beslut):**

- **E1-provet är svagare än planerat.** Även efter att iögonfallande föremål uteslutits ligger rutor som bara fångats direkt intill ett sådant föremål, eftersom de kommer från 2×2-fönster runt det. Agenten undviker troligen fönster som överlappar iögonfallande föremål. Nollan kan därför ha det skälet. E1 är inte motbevisad, men provet kunde inte avgöra den.
- **Ett oväntat mönster:** effekten per ruta är 2–3 gånger större för rutor som täcktes i steg 3 än för alla täckta rutor. Det går inte att skilja från närhet, eftersom agenten söker lokalt och steg 3:s rutor ligger nära det aktuella fokuset. Jag drar ingen slutsats.
- **Asymmetrin är verklig och oförklarad.** Agenten behandlar en markering "sedd" som ett nästan absolut veto, medan en saknad markering bara delvis återuppväcker intresset. En möjlig förklaring som inte prövades är att agenten minns vad den såg i rutan, alltså föremålets form, och att det minnet väger tyngre än schemats "inte sedd". Det är en variant av E1 som provet inte kunde fånga.
- **Följd för beteendemåttet:** E1 och E4 fick inget stöd. Enligt planen behöver beteendemåttet därför varken delas efter siktade och fångade rutor eller normeras mot konkurrerande rutor. Det mäts som det står. Asymmetrin talar för att måttet är välvalt. Det gäller fallet där schemat felaktigt säger "redan sedd", alltså den riktning agenten följer nästan helt.

### Steg 2: bygget (klart 23 okt 12:22, inget tränat före push)

Bygget följer den godkända planen och villkor a–c. Allt pushades innan piloten startade (commit 003308e). Inga låsta kriterier är ändrade.

- **Matchningssignal** (`v2/agent_rl.py`): +1 där foveáns innehåll, upptaget och färg, stämmer med periferin i det avsedda fönstret, −1 där det inte stämmer, 0 annars. Den går bara till schemamodulen.
- **Schemamodul med kostsam kanal** (`v2/schema_channel.py`): full indata, eget nät, ett brusigt tal per ruta, informationsmängd i bitar, avkodare som bara ser kanalen. Tränad mot förra stegets faktiska fönster, ingen gradient från styrningen.
- **Reglering** (`v2/train_rl.py`): kostnadens vikt justeras varje träningsomgång mot målnivån.
- **Begränsad betraktare och nollmodell med blandning** (`v2/h5.py`, villkor a).
- **Körverktyg för modulagenter på fröbanken** (`v2/h5_bank.py`): H5-händelser, träffsäkerhet, informationsmängd, beteendemåttets grupper och H2 (b) i modulens karta.
- **Pilotens utvärdering och körskript** (`v2/pilot_5a.py`, `v2/run_sprint5a.sh`), med referensregeln i villkor b.

**Tester, alla godkända** (`v2/test_schema_channel.py`, `v2/test_h5_5a.py`; de gamla testerna går också igenom):

- Kanalens gradienter stämmer med finita differenser (största relativa fel 7 × 10⁻⁶).
- Schemakartan beror på indata bara genom kanalens värde.
- **Doskontroll:** matchningssignalen visar avvikelse vid exakt de fångster där den begränsade betraktaren kan upptäcka fångsten (1 256 av 1 386 fångster i provet), och aldrig utan fångst.
- **Villkor a, syntetiska fall:** den odämpade betraktaren ger den avsedda platsen exakt 0 vid alla 201 H5-händelser. Nollmodellen med blandning ger den i medel 0,09 vid q = 0,3. Träffsäkerheten sjunker med q (0,81, 0,66 och 0,41 vid q = 0, 0,3 och 0,8), så q går att matcha mot schemats.
- H2 (b) i modulens karta: identiskt före fjärde kommandot, och bara de valda rutorna ändras.
- Agenterna från sprint 4 är bitidentiska med före bygget.

**Tekniska val i bygget (Claudes):**

- **Blindhet i doskontrollen för beteendemåttet.** Antalet fångster per grupp skulle visa hur ofta schemat pekar på den avsedda platsen, alltså H5:s utfall. Piloten rapporterar därför bara om varje grupp når minst 200 fall per frö och nivå, som ja eller nej. Antalen lagras inte.
- **Kanalens brus** dras ur agentens egen slumpström i fröbanken, före styrningens dragningar. Det håller episoderna oberoende av varandra.
- **Träffsäkerhet** mäts som andelen steg där kartans mest sannolika ruta ligger i det faktiska fönstret. Det gäller både över alla steg och vid H5-händelser.
- **Informationsmängden** för referensen mäts på pilotbanken, i bitar per steg, bara för steg där schemat används, alltså från andra steget.
- **Regleringens steg** är 0,05 per omgång i logaritmisk skala. Startvärdet för nivåerna är 0,001.
- **Tidsbedömning:** kanalvarianterna är långsammare att träna än Baslinjen. Piloten kan ta längre än 2,5 timmar. Det redovisas när första fasen är klar.

### Steg 3: pilot – resultat

**Resultat fas A (23 okt 13:20): förutsättningen klaras inte. Piloten är stoppad enligt planen, och fas B har inte startats.** Schemamodulen med full information och nästan ingen kostnad lägger sin mest sannolika ruta i det faktiska fönstret vid 20–21 % av H5-händelserna. Kravet var minst 50 %. Den begränsade betraktaren lägger 94–95 % av sin sannolikhet där. Referensen är dessutom inte väldefinierad enligt villkor b. Vart schemats fel pekar har jag inte tittat på.

Alla siffror på pilotbanken. Frö 20 och 21, 400 000 episoder, bästa checkpoint. Rätt betyder att kartans mest sannolika ruta ligger i förra stegets faktiska fönster.

| Körning | Frö | Träffar målet | Bitar per steg | Rätt utan fångst | Rätt vid fångst | Rätt vid H5-händelser | Betraktarens massa på faktiskt fönster |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Baslinje | 20 | 86,8 % | – | – | – | – | – |
| Baslinje | 21 | 86,6 % | – | – | – | – | – |
| 1b, kostnad 0,001 | 20 | 87,2 % | 79 | 99,9 % | 10,8 % | 19,9 % | 0,94 |
| 1b, kostnad 0,001 | 21 | 86,3 % | 76 | 99,9 % | 11,5 % | 19,8 % | 0,94 |
| 1b, kostnad 0,0001 | 20 | 86,9 % | 186 | 99,9 % | 11,5 % | 20,1 % | 0,95 |
| 1b, kostnad 0,0001 | 21 | 86,6 % | 181 | 100 % | 10,9 % | 20,6 % | 0,94 |

- **Förutsättningen:** 20–21 % mot kravet 50 %, i båda fröna. Ej klarad.
- **Referensen (villkor b):** 78 bitar per steg vid 0,001 och 183 vid 0,0001, en faktor 2,4. Gränsen var 1,5. Halveringsregeln skulle därför gälla, men den har ingen mening förrän förutsättningen är löst. De extra bitarna köper ingenting, eftersom träffsäkerheten är densamma.
- **Schemats nytta:** ingen. 1b träffar målet lika ofta som Baslinjen, som i sprint 3.
- **Övrigt:** fångster sker i ungefär en tredjedel av stegen. 2 553–2 662 H5-händelser per körning. Båda grupperna för beteendemåttet når minst 200 fall i alla körningar. Modulens förlust planade ut redan vid ca 200 000 episoder.
- **Kontroll av koden:** felet sitter inte i mätningen. Utan fångst träffar modulen rätt i 99,9–100 % av stegen, som sprint 3:s modul. Skript: `v2/pilot_5a_accsplit.py`, resultat: `v2/results/sprint5a_accsplit.txt` och `v2/results/sprint5a_pilot.json`.

**Min tolkning (Claudes, inte ett beslut):**

- **Modulen har informationen men använder den inte.** Matchningssignalen visar avvikelse vid varje upptäckbar fångst, och den begränsade betraktaren hittar rätt plats med 94 % säkerhet. Ändå hittar modulen rätt plats vid bara 11–12 % av fångsterna. En trolig men oprövad förklaring är att den säger "där jag siktade". Vart felen pekar är inte mätt (rättat efter granskningen 23 okt). Det som saknas är alltså beräkning, inte information.
- **En trolig orsak är arkitekturen.** Avvikelsen syns i de rutor agenten siktade mot, men den faktiska platsen ligger på ett annat ställe, hos ett iögonfallande föremål. Rutorna kommunicerar bara via ett globalt medelvärde. Att flytta sannolikheten från en ruta till en annan kräver samordning som den arkitekturen klarar dåligt. Det är diskussionstrådens fråga 2, schemat som samordning.
- **Det är inte ett H5-resultat.** Men det är ett informativt negativt fynd: ett självschema med full tillgång till bevisen hittar ändå inte rätt plats vid fångst, troligen av arkitektoniska skäl och inte på grund av komprimering. H5 mätt på den här modulen skulle mest spegla arkitekturen, som befarades redan i sprint 3.

**Förslag, du beslutar (inget körs före ditt svar):**

- **A (rekommenderas): en lokal signal där spotlighten hamnade.** För varje ruta anger den om glimten, upptaget och färg, stämmer med periferin i ett fönster centrerat där. Då får modulen lokala bevis på den faktiska platsen, samma bevis som den ideala betraktaren använder. Biologiskt motsvarar det att glimten jämförs med den perifera kartan överallt, en igenkänningskarta. Ca 1 timmes bygge och 1 timme för att köra fas A igen. Risk: det närmar sig en handbyggd detektor, även om modulen fortfarande måste lära sig använda den.
- **B: bättre samordning mellan rutorna,** till exempel ett globalt maximum utöver medelvärdet. Det prövar fråga 2 direkt men är osäkrare. Ca 2 timmars bygge och en körning.
- **C: längre träning.** Avråds, eftersom förlusten planade ut tidigt.
- **D: avsluta H5 i den här formen.** Rapportera det negativa fyndet och gå vidare till 1c.

Med A eller B mäts referensen om, och villkor b gäller som det står.

**Beslut för produktägaren** (skriv svar: under punkten)

- [x] Väg vidare för H5: A, B, C eller D?
- [x] svar (i chatten 23 okt 14:03, infört av Claude): E, ett femte alternativ som togs fram i chatten. Se nedan.

**Beslut 23 okt 2026 kl. 14:03 (produktägaren, i chatten): alternativ E, global larmsignal.** Bakgrund: produktägaren frågade vilket alternativ som mest sannolikt kunde ha drivits fram evolutionärt, som en silvertejpslösning. Claude föreslog E, som inte fanns bland A–D.

- **Vad E är:** matchningssignalen räknas ihop till en enda siffra per steg, "något stämde inte där jag siktade", och sänds till alla rutor i schemamodulen. Modulen behåller den lokala matchningssignalen och sin vanliga indata. Varje ruta har redan sin egen brusiga salience från periferin. Modulen kan då lära sig tumregeln att om larmet går och rutan är iögonfallande, hamnade uppmärksamheten troligen här.
- **Varför evolutionärt sannolik:** den återanvänder det som redan finns, jämförelsen via efferenskopian och prioritetskartans salience. Det enda nya är en spridningskanal. Globala fel- och överraskningssignaler är vanliga i hjärnan, till exempel via noradrenalin från locus coeruleus. Larmet säger bara att något gick fel, inte var, så med flera iögonfallande föremål måste modulen gissa.
- **Jämfört med A och B:** A säger var glimten passar, vilket är generösare än biologin. B ändrar arkitekturen. E är en indatakanal till, beräknad som matchningssignalen, utan nya gradienter.
- **För H5:** vid en fångst konkurrerar tre gissningar: den avsedda platsen enligt styrsignalen, det mest iögonfallande föremålet enligt larmet och saliencen, och rätt plats.
- **Oförändrat:** H5:s mätning, nollmodellen, villkor a–c, förutsättningen på 50 %, nivåerna och beteendemåttet. Fas A körs om med E: referensvarianterna 0,001 och 0,0001, frö 20 och 21. Baslinjens körningar återanvänds, eftersom Baslinjen inte påverkas. Resultaten från den första fas A står kvar ovan.

### Steg 3b: pilot med alternativ E – resultat

**Fas A med E (23 okt 14:49): förutsättningen klaras.** Modulen med larmsignal och kostnad 0,0001 lägger sin mest sannolika ruta i det faktiska fönstret vid 57 % av H5-händelserna. Kravet var minst 50 %. Referensen är fortfarande inte väldefinierad, så halveringsregeln i villkor b gäller. Doskontrollen för beteendemåttet klaras inte, se nedan.

| Körning | Frö | Träffar målet | Bitar per steg | Rätt över alla steg | Rätt vid H5-händelser | Betraktarens massa på faktiskt fönster | Beteendegrupper ≥ 200 (rätt / fel) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E, kostnad 0,001 | 20 | 87,9 % | 74 | 76,1 % | 55,6 % | 0,94 | ja / nej |
| E, kostnad 0,001 | 21 | 87,9 % | 75 | 76,1 % | 55,0 % | 0,95 | ja / nej |
| E, kostnad 0,0001 | 20 | 87,4 % | 183 | 76,2 % | 57,2 % | 0,94 | ja / nej |
| E, kostnad 0,0001 | 21 | 87,9 % | 171 | 76,4 % | 57,4 % | 0,95 | ja / nej |

Baslinjen träffar 86,8 och 86,6 %. Larmsignalen höjer alltså modulens träffsäkerhet vid fångst från ca 20 till ca 56 %. Agenten träffar målet ungefär 1 procentenhet oftare än Baslinjen, en liten skillnad nära gränsen för vad piloten kan säkerställa.

- **Förutsättningen:** 57,2 och 57,4 % mot kravet 50 %. Klarad i båda fröna.
- **Referensen (villkor b):** 75 bitar vid 0,001 och 177 vid 0,0001, en faktor 2,4. Halveringsregeln gäller. Träffsäkerheten över alla steg är densamma vid 75 som vid 177 bitar, så referensen hamnar lägre.
- **Doskontrollen för beteendemåttet klaras inte.** Gruppen där schemat felaktigt pekar på det avsedda fönstret, och den avsedda rutan har ett osett föremål i målets färg, når inte 200 fall i någon körning. Beteendemåttet kan därför inte låsas i den form det har nu. Det är en fråga till grind 5a.
- **En brist i blindheten.** Ja/nej-svaret i doskontrollen avslöjar något om H5 som jag inte förutsåg. Med ungefär 1 100 fel vid H5-händelser per körning kan felen inte till största delen ligga på den avsedda platsen. Annars hade gruppen blivit större. Antalet har jag inte sett. Bristen redovisas här, så att du och granskaren kan väga in den när trösklarna för H5 låses.

**Halveringsregeln (villkor b), tekniskt genomförande:** körningar som regleras mot halveringar av 0,0001-variantens informationsmängd, alltså 177, 88, 44, 22, 11, 5,5 och 2,8 bitar per steg, båda fröna. Referensen blir den minsta nivå där träffsäkerheten över alla steg ligger inom 1 procentenhet av 0,0001-varianten. Eftersom nivåerna är 50, 25 och 12,5 % av referensen sammanfaller de med de tre följande halveringarna. Samma körningar är därför både svep och nivåer. 44 och 22 bitar körs först, start 14:55, och utfallet avgör om svepet fortsätter nedåt. Träffsäkerheten vid 75 bitar ligger redan inom 1 procentenhet, så 88 hoppas över.

**Resultat (23 okt 16:17): flaskhalsen biter tydligt, och regleringen höll nivåerna.** Träffsäkerheten vid H5-händelser faller från 57 % vid 177 bitar per steg till 14 % vid 5,5 bitar, jämnt i båda fröna. Agentens träffandel påverkas knappt. Vart felen pekar har jag inte tittat på.

| Bitar per steg (mål) | Uppmätt, frö 20 / 21 | Rätt över alla steg | Rätt vid H5-händelser | Schemats massa på faktiskt fönster vid H5 | Träffar målet |
| --- | --- | --- | --- | --- | --- |
| 177 (kostnad 0,0001) | 183 / 171 | 76,2 / 76,4 % | 57,2 / 57,4 % | 0,26 / 0,27 | 87,4 / 87,9 % |
| 75 (kostnad 0,001) | 74 / 75 | 76,1 / 76,1 % | 55,6 / 55,0 % | 0,28 / 0,27 | 87,9 / 87,9 % |
| 44,2 | 44,9 / 43,6 | 76,0 / 75,2 % | 54,1 / 51,5 % | 0,29 / 0,24 | 86,6 / 87,9 % |
| 22,1 | 21,8 / 22,3 | 73,4 / 73,2 % | 44,1 / 44,8 % | 0,23 / 0,24 | 87,2 / 87,7 % |
| 11,05 | 10,7 / 11,4 | 63,4 / 64,9 % | 24,7 / 26,5 % | 0,14 / 0,15 | 87,2 / 87,2 % |
| 5,53 | 5,5 / 6,0 | 46,7 / 48,5 % | 14,6 / 14,1 % | 0,09 / 0,09 | 86,8 / 86,5 % |
| Baslinje | – | – | – | – | 86,8 / 86,6 % |

Standardfelet för träffandelen är ca 0,5 procentenheter. Den begränsade betraktaren lägger 93–95 % av sin sannolikhet på det faktiska fönstret i alla körningar. Varje körning har 2 494–2 658 H5-händelser.

**Referensen enligt villkor b:** träffsäkerheten över alla steg för 0,0001-varianten är 76,3 % i medel. Vid 44 bitar ligger medelvärdet 0,7 procentenheter lägre, alltså inom gränsen. Vid 22 bitar ligger det 3,0 lägre. Villkoret säger inte om gränsen gäller medelvärdet eller varje frö. I frö 21 ligger 44 bitar 1,2 procentenheter under. Det ger två läsningar:

| Läsning | Referens | Nivåer (50 / 25 / 12,5 %) |
| --- | --- | --- |
| Medelvärdet över fröna (rekommenderas) | 44 bitar | 22 / 11 / 5,5 bitar |
| Varje frö för sig | 88 bitar | 44 / 22 / 11 bitar |

Jag rekommenderar medelvärdet, eftersom referensen i planen redan är definierad som medelvärdet över fröna. Tvetydigheten upptäcktes efter att siffrorna setts, så du beslutar. Alla nivåer finns körda för båda läsningarna.

**Iakttagelser:**

- **Schemats nytta är liten.** E-modulerna träffar målet 0–1 procentenhet oftare än Baslinjen, nära gränsen för vad piloten kan säkerställa. Agenten läser alltså schemat lite. Det försvagar beteendemåttet och H2 (b) på 1b, men inte H5, som mäts på modulen själv.
- **Vid den lägsta nivån blir modulen också sämre utan fångst.** Vid 5,5 bitar är den rätt i bara 47–49 % av alla steg. Nollmodellen matchas mot schemats träffsäkerhet och hanterar det, men tolkningen vid den nivån blir svårare.
- **Doskontrollen för beteendemåttet klaras inte vid någon nivå.** Felgruppen når inte 200 fall. Bristen i blindheten som beskrivs ovan gäller alla nivåer.

### Steg 4: förslag till trösklar och körplan för 5b (låses vid grind 5a)

**H5, förslag.** Vid H5-händelser jämförs schemats andel fel på det avsedda fönstret med nollmodellens. Andelen är schemats sannolikhetsmassa på det avsedda fönstret delad med dess massa utanför det faktiska fönstret. Båda summeras över händelserna inom fröt. Nollmodellen är blandningsbetraktaren (villkor a). Dess q väljs per frö och nivå så att dess massa på det faktiska fönstret blir densamma som schemats, och samma andel räknas för den. Godkänt: schemats andel är större än nollmodellens med mer än 2 standardfel, i alla tre frön. Standardfelet tas fram med bootstrap över händelserna (1 000 omdragningar). H5 bedöms per nivå och redovisas för alla tre. Varianten utan kostnad redovisas också, men den är inget kriterium.

**Beteendemåttet, förslag.** Det kan inte låsas som det står, eftersom felgruppen är för liten. Två vägar:

- **(i) Större bank (rekommenderas):** 5b använder 20 000 testepisoder per frö för beteendemåttet, i stället för 4 000. Regeln om minst 200 fall per grupp låses. Nås den inte i 5b redovisas måttet som omätbart, utan godkänt eller underkänt.
- **(ii) Beskrivande:** måttet blir inget kriterium och redovisas bara.

**H2 (b) på 1b:** jag rekommenderar att det redovisas utan att bli kriterium, eftersom agenten läser schemat så lite.

**Körplan för 5b, förslag.** Baslinje, 1b utan kostnad (0,0001) och 1b vid de tre nivåerna, frö 30, 31 och 32, 15 körningar à 400 000 episoder. Modulens förlust planar ut vid ca 200 000 episoder, och H5 mäts på modulen. Ca 21 minuter per körning, två åt gången, ca 2,7 timmar. Test på MVP:s fröbank, 4 000 episoder, och för beteendemåttet en utökad bank om 20 000 episoder.

**Om svaret blir nej (5b)**

| Utfall | Vad det betyder | Vad som händer |
| --- | --- | --- |
| H5 klaras vid alla nivåer | En kapacitetsbegränsad modell behåller avsikten och släpper undantagen, mer än en jämnt försvagad betraktare | En följd av AST, inte unikt för AST (villkor c). Det AST-specifika prövas i 1c |
| H5 klaras vid några nivåer | Beror på hur hårt modulen måste förenkla | Rapporteras per nivå |
| H5 klaras inte | Komprimeringen fördelar felen som en opartisk betraktare | Ett negativt resultat för H5 i den här formen. Rapporteras som det är |

### Oberoende granskning av 5a

**Oberoende granskning (23 okt 16:40): godkänd med anmärkningar. Kod, pilot och villkor b är godkända. H5-kriteriet i steg 4 kan inte låsas som det står.** Hela rapporten: `v2/results/sprint5a_review.md`. Granskaren höll blindheten. Nollmodellens siffror nedan kommer från fångster hos Baslinjen och en slumpagent, aldrig från schemamodulen.

- **Bekräftat:** alla testsviter går igenom, och granskarens eget gradienttest på E-modulen ger största relativa fel 4 × 10⁻⁵. Avkodaren ser bara kanalen, och inget läcker det faktiska fönstret eller målet. Träningen, fröna och bankerna är som de redovisats, och regleringen nådde målen. Pilotens siffror stod sig när granskaren räknade om dem med egen kod. Diagnostiken återskapar de låsta H2-siffrorna exakt, och uteslutningen av iögonfallande föremål är motiverad och öppet redovisad.
- **1, blockerande: q går inte att matcha som jag föreslog.** Även när all evidens ersätts (q = 1) behåller nollmodellen 0,33 av sin massa på det faktiska fönstret. Schemat ligger under det på alla nivåer. Att matcha på massa avviker dessutom från specens egen definition av träffsäkerhet, som gäller den mest sannolika rutan.
- **2, blockerande: schemat och nollmodellen jämförs på olika rutor.** Nollmodellen sprider sin massa bara över kandidatfönstren, medan schemats karta är diffus över hela rutnätet. Vid händelser med bara ett iögonfallande föremål, 18–20 % av händelserna, är nollmodellens andel alltid 1, och där kan inget schema slå den. Kriteriet lutade alltså mot underkänt per konstruktion.
- **3, stor: hur nollmodellen ersätter glimten sätter ribban.** Koden väljer ett kandidatfönster jämnt. Villkor a:s lydelse, "ur sin egen fördelning", och dess motivering pekar mot att glimten dras efter fångstens förhandssannolikheter. Nollmodellens andel fel på den avsedda platsen blir 0,54–0,58 med jämnt val och 0,72–0,77 med förhandssannolikheter.
- **4, stor: koden som ska beräkna H5 finns inte än.** Den ska skrivas, provas på otränade agenter och pushas före låsningen. Bootstrap ska göras över episoder, med q matchat på nytt i varje omdragning, och regeln för hur nivåerna vägs samman ska bestämmas i förväg.
- **5, måttlig: beteendemåttet.** Läckan från doskontrollen är verklig men svag. Granskaren rekommenderar att måttet blir beskrivande, eftersom agenten bara träffar 0–1 procentenhet oftare med schemat och ett nollresultat därför inte säger något. Samma sak gäller H2 (b) på 1b.
- **6, måttlig: referensens läsning.** Texten stöder medelvärdet, men avgörandet, 0,70 mot 1 procentenhet, ligger inom bruset. Vid 5,5 bitar är modulen fel i hälften av alla steg, och där går ingen nollmodell att matcha.
- **7 och 8, mindre: två formuleringar sa mer än vad som mätts.** E1 var inte prövad, bara oavgjord. Att den första modulen "nästan alltid säger där jag siktade" gäller vart felen pekar, vilket ingen tittat på. Båda är rättade ovan.
- **9, mindre: körplanen för 5b.** Lås målen i bitar som tal. Pröva förutsättningen igen i de nya fröna, med en följd bestämd i förväg. Använd slutcheckpointen, eftersom den bästa väljs på agentens träffandel, inte på modulen.
- **Tolkningsnot:** vid varje H5-händelse går larmet per definition. All massa på den avsedda platsen betyder alltså att modulen släpper en helt pålitlig signal. Det är vad komprimering väntas släppa först, och det stärker villkor c. Schemat hjälper agenten med bara 0–1 procentenhet, så vad H5 än visar gäller en modell av uppmärksamheten som agenten knappt använder för styrning.

### Reviderat H5-förslag efter granskningen (23 okt, Claude)

Förslaget i steg 4 ersätts av det här. Det låses inte förrän koden är skriven, provad på otränade agenter och pushad (granskningens punkt 4), och förrän det är visat att matchningen går att genomföra på varje nivå. Grind 5a tas därför i två steg: dina beslut nu, sedan kod och genomförbarhet, sedan låsning.

- **Samma rutor för schema och nollmodell (punkt 2).** Båda jämförs bara över kandidatfönstren: det avsedda, det faktiska och övriga iögonfallande föremåls fönster. Schemats massa normeras om över dem. Massan utanför kandidaterna redovisas för sig.
- **Bara händelser med minst ett annat iögonfallande föremål (punkt 2).** Annars finns det bara en felaktig plats, och provet blir tomt.
- **Måttet:** schemats andel av felmassan som ligger på det avsedda fönstret, jämfört med nollmodellens. Felmassan är massan på kandidaterna utom det faktiska fönstret. Andelen summeras över händelserna inom fröt.
- **Matchning (punkt 1):** q väljs så att nollmodellens massa på det faktiska fönstret, omnormerad över kandidaterna, blir densamma som schemats. Alternativet är den mest sannolika rutan. Den har ett golv som gör låga nivåer omöjliga att matcha. Går matchningen inte att genomföra på en nivå, redovisas nivån som ej bedömbar. Det bestäms nu.
- **Glimtens ersättning i nollmodellen (punkt 3):** dragen efter fångstens förhandssannolikheter, inte jämnt. Det motsvarar villkor a:s lydelse och ger den hårdare ribban, 0,72–0,77 mot 0,54–0,58.
- **Statistik (punkt 4):** bootstrap över episoder, 1 000 omdragningar, med q matchat på nytt i varje. Godkänt: schemats andel är större än nollmodellens med mer än 2 standardfel, i alla tre frön.
- **Nivåer:** H5 bedöms per nivå, utan samlat godkännande. Med medelvärdesläsningen bedöms 22 och 11 bitar, och 5,5 bitar redovisas utan bedömning, eftersom modulen där är fel i hälften av alla steg (punkt 6). Varianten utan kostnad redovisas också.
- **Beteendemåttet och H2 (b) på 1b:** beskrivande (punkt 5). Agenten använder schemat så lite att ett nollresultat inte skulle säga något.
- **5b (punkt 9):** målen låses i bitar per steg som tal från piloten. Förutsättningen prövas igen i frö 30–32, och klaras den inte bedöms H5 inte i 5b. Mätningen görs på slutcheckpointen.

### Grind 5a: dina beslut

**Beslut för produktägaren** (skriv svar: under respektive punkt)

- [ ] Referensens läsning: medelvärdet över fröna (rekommenderas, stöds av texten; nivåer 22 / 11 / 5,5 bitar, där 5,5 bara redovisas) eller varje frö för sig (nivåer 44 / 22 / 11 bitar)?
- [ ] Glimtens ersättning i nollmodellen: efter fångstens förhandssannolikheter (rekommenderas, hårdare ribba) eller jämnt?
- [ ] Matchning: omnormerad massa på det faktiska fönstret (rekommenderas) eller den mest sannolika rutan?
- [ ] Beteendemåttet och H2 (b) på 1b blir beskrivande, inte kriterier?
- [ ] Godkänner du resten av det reviderade förslaget: samma rutor för schema och nollmodell, händelser med minst ett annat iögonfallande föremål, bootstrap över episoder, bedömning per nivå och körplanen för 5b med 400 000 episoder, frö 30–32 och slutcheckpoint?

Efter dina svar skriver och provar jag H5-koden och redovisar om matchningen går att genomföra på varje nivå. Genomförbarheten avgörs bara på schemats träffsäkerhet, inte på vart felen pekar. Därefter låses trösklarna.

## Efter MVP: 1c, social fas och begreppsbildning

Beslut och skisser från 22 okt om vad som kommer efter MVP. De flyttades hit från sprint 2-avsnittet för att det ska gå att följa sprintarna.

**Beslut 22 okt 2026 (produktägaren):** nästa steg efter MVP blir en fas med flera agenter som måste förutsäga varandras uppmärksamhet. Den prövar schemats sociala, spelteoretiska funktion, som MVP inte testar, och om ett schema växer fram lättare under socialt tryck än under ren styrning. Produktägarens tes: självmodellen är en föreställning som alla andra, men nödvändig av spelteoretiska skäl.

Skiss för den sociala fasen (förslag 22 okt, ej beslutad), i Conways anda: minsta möjliga regler, inget facit, och se vad som växer fram.

1. Två agenter söker i samma rutnät. Var och en ser grovt var den andras spotlight låg. Tävling: den som hittar målet först vinner. Samarbete: de delar belöningen.
2. Varje agent har en fri intern kanal med kostnad per informationsmängd (1c), tränad bara mot belöningen.
3. Jämförelse 2 × 2: ensam eller social, med eller utan kanal. Tesen att schemat är spelteoretiskt nödvändigt förutsäger att kanalen beskriver den egna uppmärksamheten tidigare och tydligare i det sociala villkoret.
4. Mät om kanalen beskriver den egna uppmärksamheten, den andras, eller båda med samma kod. Graziano förutsäger samma kod.
5. Möjligt: urval och mutation (evolution) i stället för gradientinlärning, i linje med den evolutionära principen, om det ryms på CPU.

**Beslut 22 okt 2026 (produktägaren):** ordningen blir MVP, sedan 1c för en ensam agent, sedan den sociala fasen. Den sociala fasen startar först när 1c visat något i enkel form, så att ett uteblivet resultat kan skiljas från en inlärning som inte fungerar.

**Inriktning 22 okt 2026 (produktägaren):** medvetande är i hög grad knutet till att vi kan tala om det, och dess evolutionära fördel bör sökas i ett socialt sammanhang med begreppsbildning, inte nödvändigtvis språk i snäv mening. Begreppsbildning ska därför ingå i det som driver fram ett schema. Förslag till form (ej beslutad): schemakanalen i 1c och den sociala fasen går genom ett fåtal diskreta symboler i stället för en numerisk karta. I den sociala fasen beskriver agenterna sin uppmärksamhet för varandra med symbolerna, och samma symboler matas tillbaka till agenten själv, som inre tal. Förutsägelse: symbolerna blir grova kategorier för egen och andras uppmärksamhet, med samma kod för båda, de styr agentens egen uppmärksamhet (lesion och falska symboler försämrar respektive flyttar den), och de är fel på det sätt AST förutsäger.

Koppling till fas 2 (22 okt): en språkmodell är formad av den sociala vägen, att förutsäga människor som modellerar varandras uppmärksamhet, men har inte belönats för en korrekt modell av sin egen. Om tesen att det sociala spelet räcker stämmer, kan ett schema ha vuxit fram där som bieffekt. Modellens egna utsagor bevisar inget, eftersom den lärt sig människors schemaspråk. Därför mäter fas 2 de inre tillstånden, inte vad modellen säger. Ordningen ändras inte: fas 2 körs om fas 1 visar effekt.

### Förslag från diskussionstråden 22–23 okt (Anders och Claude, ej beslutat)

Infört 23 okt på produktägarens begäran. Inget av det här är beslutat, och det påverkar varken sprint 4:s låsta utfall eller plan 5a.

**Sammanhållande fråga:** när behöver en helhet av delar en gemensam, förenklad modell av sin egen uppmärksamhet, och vad händer när modellen och verkligheten skiljer sig åt?

**Inspiration:**

- **Fåglars lateralisering (Rogers).** Ena hjärnhalvan fokuserar och kategoriserar, den andra vakar.
- **Minskys Society of Mind.** En B-hjärna övervakar en A-hjärna, och självkännedomen är förenklad. Nexis-agenten, med en enhet per ruta och kommunikation bara via ett globalt medelvärde, är ett sådant samhälle.
- **Dennett och Harari.** Jaget och samhällets fiktioner fungerar som berättelser. Anders poäng är att berättelsen inte existerar i sig men är verklig genom det den handlar om, alltså genom sin intentionalitet. Språket är ett av flera sätt att modellera. Det ger en icke-alienerande läsning av AST: schemat är förenklat, men ingen illusion.

**Tre frågor som går att pröva:**

1. **Inre kapning.** En agent med en fokuserande och en vaksam kanal, där fångsten kommer inifrån i stället för från en slumpmekanism. Anger schemat den avsedda platsen när den vaksamma sidan tagit över? Det är H5 med fågelns arkitektur.
2. **Schemat som samordning.** Piloten antyder att 1b misslyckades delvis för att rutorna bara kommunicerar via ett globalt medelvärde. Ger bättre kommunikation mellan rutorna ett fungerande eget schema, eller behövs en separat modul, en B-hjärna?
3. **Från modell till berättelse.** Uppstår vilseledning om den egna uppmärksamheten i den sociala fasen först när schemakanalen får diskreta symboler? Ett ordlöst schema kan ha fel men inte ljuga. En symbolisk modell kan frikopplas och ljuga.

**Förbehåll:** tänkarna tillhör i stort samma funktionalistiska tradition, så att de stämmer överens är ingen oberoende bekräftelse.

**Claudes kommentar:** fråga 1 ligger närmast det som redan finns. Plan 5a:s matchningssignal och flaskhals kan återanvändas, och fångstmekanismen byts mot en inre vaksam kanal. Fråga 2 går delvis att besvara redan i 5a, eftersom schemamodulen där är just en separat B-hjärna.

## Öppna frågor och beslut för produktägaren

- [x] Fas 1 (agenten) körs först och avgör huvudfrågan. Fas 2 (GPT-2) körs bara om fas 1 visar effekt.
- [x] Fas 2 använder engelsk textkorpus.
- [x] Agenten i fas 1 tränas med förstärkningsinlärning: belöning vid träff, avdrag per steg.
- [x] Rutnätets storlek, fönster, stegbudget och antal distraktorer varierar slumpmässigt mellan episoder.
- [x] Tröskelvärden för verifieringskriterierna fastställs innan första körningen.
- [x] Spec och resultat förs i detta dokument.
- [x] Zoom och bredd tas med som första utbyggnad efter MVP. MVP har fast fönster, och schemat anger läge, förflyttning och besökta områden.
- [x] svar:zoom och bredd verkar vara viktuga faktorer
- [x] vi väntar med detta om detta leder till alltför äkad komplexitet
- [x] Ledtrådarnas styrka: tre nivåer, ingen, svag och stark. MVP kör nivån svag, övriga senare.
- [x] Svar: förstår inte

*Förklaring av ledtrådsnivåerna:* nivåerna är olika svåra versioner av samma uppgift. Vid *ingen* visar periferin bara att det finns objekt, inte deras färg. Vid *svag* visar periferin färg, men de flesta distraktorer har målets färg, så färgen hjälper lite. Vid *stark* har få distraktorer målets färg, så färgen pekar nästan ut målet. MVP kör den svaga nivån, där agenten måste hålla reda på många likadana kandidater och schemat har mest att göra.
