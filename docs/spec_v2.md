<!-- Ögonblicksbild av det levande dokumentet "Nexis v2 – spec" (https://claude.ai/artifact/GsEmfNtb1vs2bhXxwc6vsW), exporterad 22 okt 2026 22:35, rev 262. Vid skillnad gäller det levande dokumentet. -->

# Nexis v2 – spec

Oct 20, 2026 · @Anders Norell

## Läge just nu

Vi är i fas 1, agentexperimentet. Sprint 3, piloten, är klar och grind 3 godkänd, så trösklarna är låsta. Planen för sprint 4, MVP, väntar på ditt godkännande. Fas 2, GPT-2-proben, körs bara om fas 1 visar effekt.

Begreppen förklaras i [Nexis – ordlista](https://claude.ai/artifact/2syuVpzf2GcAfaP9DgSgjC).

| Sprint | Innehåll | Status |
| --- | --- | --- |
| 0 | Spec | Klar |
| 1 | Miljön | Klar, grind 1 godkänd 21 okt |
| 2 | Kompetens: baslinjeagent utan schema | Klar, grind 2 godkänd 22 okt |
| 3 | Pilot: arbetspunkt, spridning, trösklar | Klar, grind 3 godkänd 22 okt, trösklar låsta |
| 4 | MVP: Baslinje, 1a och Blandat; F, H1, H2, H3 | Plan väntar på ditt godkännande |
| 5 | H5: schemamodul med lokal matchning, egen pilot | Ej påbörjad |
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

### Plan (väntar på produktägarens godkännande)

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

- [ ] Godkänner du sprint 4-planen?

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
